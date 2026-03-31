# orarpro-solver/main.py
#
# FastAPI microservice pentru generare orare — Google OR-Tools CP-SAT.
#
# Endpoints:
#   GET  /health           — health check (Railway + Next.js wake-up)
#   POST /solve/shifts     — orar ture business (horeca, fabrici, retail)
#   POST /solve/business   — alias pentru /solve/shifts (compatibilitate)
#   POST /solve/school     — orar școlar CP-SAT v4
#
# Deploy: Railway → railway up
# Local:  uvicorn main:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from solvers.business import solve_shifts, ShiftsRequest, ShiftsResponse
from solvers.school   import solve_school,  SchoolRequest,  SchoolResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OrarPro Solver",
    description="CP-SAT schedule generation — Google OR-Tools",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check — Railway și Next.js wake-up call."""
    return {"status": "ok", "solver": "OR-Tools CP-SAT", "version": "3.0.0"}


@app.post("/solve/business", response_model=ShiftsResponse)
async def solve_business_endpoint(payload: ShiftsRequest) -> ShiftsResponse:
    """
    Orar ture business (horeca, fabrici, retail, clinici) — CP-SAT v2.
    Input:  employees, shift_definitions, working_dates, config, soft_rules.
    Output: assignments[], violations[], stats, debug_log[].
    """
    logger.info(
        f"/solve/business: {len(payload.employees)} angajați, "
        f"{len(payload.shift_definitions)} ture, "
        f"{len(payload.working_dates)} zile"
    )
    try:
        return solve_shifts(payload)
    except Exception as e:
        logger.error(f"Business solver error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve/school", response_model=SchoolResponse)
async def solve_school_endpoint(payload: SchoolRequest) -> SchoolResponse:
    """
    Orar școlar CP-SAT v4 — class×subject model.
    Input:  lessons[], teachers[], classes[], rooms[], soft_rules.
    Output: timetable[], violations[], stats, debug_log[].
    """
    logger.info(
        f"/solve/school: {len(payload.lessons)} lecții, "
        f"{len(payload.classes)} clase, "
        f"{payload.slots_per_day} sloturi/zi"
    )
    try:
        return solve_school(payload)
    except Exception as e:
        logger.error(f"School solver error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))