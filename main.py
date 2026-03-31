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
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    Timetable scheduling for schools (v3 — class×subject model, no teachers/rooms).
    Input: class_subjects with periods_per_week, config.
    Output: timetable (class × subject × day × period) + violations + debug_log.
    """
    logger.info(
        f"Solving school timetable: {len(payload.lessons)} lessons, "
        f"{len(payload.classes)} classes, "
        f"{payload.slots_per_day} slots/day"
    )
    try:
        return solve_school(payload)
    except Exception as e:
        logger.error(f"School solver error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from solvers.business import solve_business

# 2. Pydantic models (adaugă în main.py lângă celelalte modele):
from pydantic import BaseModel
from typing import Optional, Any

class HardConfig(BaseModel):
    min_employees_per_shift: int = 2
    max_consecutive_days: int = 6
    min_rest_hours: float = 11.0
    max_weekly_hours: float = 48.0
    max_night_shifts_per_week: int = 2
    enforce_legal_limits: bool = True

class SoftWeights(BaseModel):
    balance: int = 90
    nightWeekend: int = 70
    preferences: int = 80
    daysOff: int = 75
    continuity: int = 40

class BusinessSoftRules(BaseModel):
    balanceHours: bool = True
    avoidNightWeekend: bool = True
    respectPreferences: bool = True
    consecutiveDaysOff: bool = True
    shiftContinuity: bool = False
    weights: SoftWeights = SoftWeights()

class BusinessSolveRequest(BaseModel):
    schedule: dict[str, Any]
    employees: list[dict[str, Any]]
    shift_definitions: list[dict[str, Any]]
    leaves: list[dict[str, Any]] = []
    hard_config: HardConfig = HardConfig()
    soft_rules: BusinessSoftRules = BusinessSoftRules()
    solver_time_limit_seconds: int = 50

# 3. Route (adaugă în main.py):
@app.post("/solve/business")
async def solve_business_route(req: BusinessSolveRequest):
    try:
        result = solve_business(req.model_dump())
        return result
    except Exception as e:
        logger.exception("Business solver error")
        raise HTTPException(status_code=500, detail=str(e))