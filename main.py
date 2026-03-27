# orarpro-solver/main.py
#
# FastAPI microservice for schedule generation using Google OR-Tools CP-SAT.
# Two endpoints:
#   POST /solve/shifts  — shift scheduling for HoReCa, factories, retail
#   POST /solve/school  — timetable scheduling for schools and universities
#
# Deploy on Railway: railway up
# Local dev: uvicorn main:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from solvers.shifts import solve_shifts, ShiftsRequest, ShiftsResponse
from solvers.school import solve_school, SchoolRequest, SchoolResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OrarPro Solver",
    description="CP-SAT schedule generation engine using Google OR-Tools",
    version="1.0.0",
)

# Allow requests from Vercel (Next.js) and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Vercel URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check — used by Railway and Next.js fallback logic."""
    return {"status": "ok", "solver": "OR-Tools CP-SAT"}


@app.post("/solve/shifts", response_model=ShiftsResponse)
async def solve_shifts_endpoint(payload: ShiftsRequest) -> ShiftsResponse:
    """
    Shift scheduling for HoReCa, factories, retail, clinics.

    Input: employees, shift definitions, working dates, constraints.
    Output: assignments (employee × shift × date) + violations + stats.

    Uses CP-SAT to find a feasible solution that:
    - Covers the required number of employees per shift per day
    - Respects all hard constraints (rest, consecutive days, pairs, seniority)
    - Minimizes soft constraint violations (balance, consistency)
    """
    logger.info(
        f"Solving shifts: {len(payload.employees)} employees, "
        f"{len(payload.shift_definitions)} shifts, "
        f"{len(payload.working_dates)} working days"
    )
    try:
        return solve_shifts(payload)
    except Exception as e:
        logger.error(f"Shifts solver error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve/school", response_model=SchoolResponse)
async def solve_school_endpoint(payload: SchoolRequest) -> SchoolResponse:
    """
    Timetable scheduling for schools and universities.

    Input: teachers, subjects, classes, rooms, periods, constraints.
    Output: timetable (teacher × subject × class × room × period) + violations.

    Uses CP-SAT to find a feasible timetable that:
    - Each subject gets the required number of periods per week
    - No teacher is in two rooms simultaneously
    - No room has two classes simultaneously
    - No class has two subjects at the same time
    - Minimizes teacher free periods (windows) between lessons
    """
    logger.info(
        f"Solving school timetable: {len(payload.teachers)} teachers, "
        f"{len(payload.subjects)} subjects, "
        f"{len(payload.classes)} classes, "
        f"{len(payload.rooms)} rooms, "
        f"{payload.periods_per_day} periods/day"
    )
    try:
        return solve_school(payload)
    except Exception as e:
        logger.error(f"School solver error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
