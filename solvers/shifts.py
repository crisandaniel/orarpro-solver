# orarpro-solver/solvers/shifts.py
#
# CP-SAT shift scheduling solver for HoReCa, factories, retail, clinics.
#
# Model overview:
#   Variables:
#     x[e][s][d] ∈ {0, 1}  — employee e works shift s on day d
#
#   Hard constraints (must be satisfied):
#     1. Slots coverage       — each shift needs exactly N employees per day
#     2. One shift per day    — each employee works at most 1 shift per day
#     3. Min rest             — min hours between consecutive shifts
#     4. Max consecutive days — limit on consecutive working days
#     5. Max weekly hours     — cap on hours per calendar week
#     6. Max night shifts     — cap on night shifts per week
#     7. Pair required        — two employees must work the same shift on same day
#     8. Pair forbidden       — two employees cannot work the same shift on same day
#     9. Min seniority        — at least one senior per shift per day
#    10. Fixed shift          — employee always works the same shift type
#    11. Leave / unavailability — employee cannot work on specific dates
#
#   Soft constraints (minimized in objective):
#     - Imbalance in total hours across employees (fairness)
#     - Shift switching (consistency — same shift on consecutive days preferred)

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
from pydantic import BaseModel
from ortools.sat.python import cp_model
import logging

logger = logging.getLogger(__name__)

# ── Input models ──────────────────────────────────────────────────────────────

class Employee(BaseModel):
    id: str
    name: str
    experience_level: str          # 'junior' | 'mid' | 'senior'
    color: Optional[str] = None

class ShiftDefinition(BaseModel):
    id: str
    name: str
    shift_type: str                # 'morning' | 'afternoon' | 'night' | 'custom'
    start_time: str                # "06:00"
    end_time: str                  # "14:00"
    crosses_midnight: bool = False
    duration_hours: float          # pre-computed by Next.js

class Constraint(BaseModel):
    type: str
    employee_id: Optional[str] = None
    target_employee_id: Optional[str] = None
    shift_definition_id: Optional[str] = None
    value: Optional[float] = None  # numeric value (hours, days, count)
    note: Optional[str] = None

class GenerationConfig(BaseModel):
    min_employees_per_shift: int = 1
    max_consecutive_days: int = 6
    min_rest_hours_between_shifts: float = 11.0
    max_weekly_hours: float = 48.0
    max_night_shifts_per_week: int = 3
    enforce_legal_limits: bool = True
    balance_shift_distribution: bool = True
    shift_consistency: int = 2     # 0=off, 1=mild, 2=strong

class ShiftsRequest(BaseModel):
    schedule_id: str
    employees: list[Employee]
    shift_definitions: list[ShiftDefinition]
    working_dates: list[str]       # ISO date strings ["2026-04-01", ...]
    slots_per_shift: dict[str, int]  # shift_id → slots needed per day
    constraints: list[Constraint] = []
    leaves: list[dict] = []        # [{employee_id, start_date, end_date}]
    unavailability: list[dict] = [] # [{employee_id, date or day_of_week}]
    config: GenerationConfig = GenerationConfig()
    solver_time_limit_seconds: int = 30

# ── Output models ─────────────────────────────────────────────────────────────

class Assignment(BaseModel):
    employee_id: str
    shift_definition_id: str
    date: str
    is_manual_override: bool = False

class Violation(BaseModel):
    type: str
    employee_id: str
    employee_name: str
    date: str
    message: str

class SolverStats(BaseModel):
    total_slots: int
    filled_slots: int
    hours_per_employee: dict[str, float]
    solver_status: str             # 'OPTIMAL' | 'FEASIBLE' | 'INFEASIBLE' | 'UNKNOWN'
    solve_time_seconds: float

class ShiftsResponse(BaseModel):
    assignments: list[Assignment]
    violations: list[Violation]
    stats: SolverStats

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date(s: str) -> date:
    return date.fromisoformat(s)

def week_number(d: str) -> int:
    """ISO week number for grouping by week."""
    return parse_date(d).isocalendar()[1]

def is_on_leave(employee_id: str, d: str, leaves: list[dict]) -> bool:
    dt = parse_date(d)
    for leave in leaves:
        if leave.get("employee_id") != employee_id:
            continue
        start = parse_date(leave["start_date"])
        end = parse_date(leave["end_date"])
        if start <= dt <= end:
            return True
    return False

def is_unavailable(employee_id: str, d: str, unavailability: list[dict]) -> bool:
    dt = parse_date(d)
    dow = dt.weekday() + 1  # 1=Mon, 7=Sun (0-indexed weekday → 1-indexed)
    if dow == 0:
        dow = 7
    for u in unavailability:
        if u.get("employee_id") != employee_id:
            continue
        if u.get("date") == d:
            return True
        if u.get("day_of_week") == dow:
            return True
    return False

def parse_time(t: str) -> tuple[int, int]:
    """Parse HH:MM or HH:MM:SS into (hours, minutes)."""
    parts = t.split(":")
    return int(parts[0]), int(parts[1])

def shift_end_hour(shift: ShiftDefinition) -> float:
    """End time as decimal hours, adjusted for midnight crossing."""
    h, m = parse_time(shift.end_time)
    end = h + m / 60
    if shift.crosses_midnight:
        end += 24
    return end

def shift_start_hour(shift: ShiftDefinition) -> float:
    h, m = parse_time(shift.start_time)
    return h + m / 60

def rest_hours_between(prev_shift: ShiftDefinition, prev_date: str,
                        next_shift: ShiftDefinition, next_date: str) -> float:
    """Hours between end of prev shift and start of next shift."""
    prev_end = shift_end_hour(prev_shift)
    next_start = shift_start_hour(next_shift)
    days_apart = (parse_date(next_date) - parse_date(prev_date)).days
    return days_apart * 24 + next_start - prev_end

# ── Main solver ───────────────────────────────────────────────────────────────

def solve_shifts(payload: ShiftsRequest) -> ShiftsResponse:
    employees     = payload.employees
    shifts        = payload.shift_definitions
    dates         = payload.working_dates
    slots         = payload.slots_per_shift
    constraints   = payload.constraints
    config        = payload.config

    E = len(employees)
    S = len(shifts)
    D = len(dates)

    emp_idx   = {e.id: i for i, e in enumerate(employees)}
    shift_idx = {s.id: i for i, s in enumerate(shifts)}
    date_idx  = {d: i for i, d in enumerate(dates)}

    # Build leave / unavailability lookup for fast access
    blocked: set[tuple[int, int]] = set()  # (emp_idx, date_idx)
    for ei, emp in enumerate(employees):
        for di, d in enumerate(dates):
            if is_on_leave(emp.id, d, payload.leaves):
                blocked.add((ei, di))
            if is_unavailable(emp.id, d, payload.unavailability):
                blocked.add((ei, di))

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────────────
    # x[e][s][d] = 1 if employee e works shift s on day d
    x = [[[model.new_bool_var(f"x_e{ei}_s{si}_d{di}")
            for di in range(D)]
            for si in range(S)]
            for ei in range(E)]

    # ── Hard constraint 1: slots coverage ────────────────────────────────────
    # Each shift on each working day must have exactly N employees
    for si, shift in enumerate(shifts):
        required = slots.get(shift.id, config.min_employees_per_shift)
        for di in range(D):
            model.add(sum(x[ei][si][di] for ei in range(E)) == required)

    # ── Hard constraint 2: one shift per day per employee ─────────────────────
    for ei in range(E):
        for di in range(D):
            model.add(sum(x[ei][si][di] for si in range(S)) <= 1)

    # ── Hard constraint 3: blocked days (leave / unavailability) ─────────────
    for (ei, di) in blocked:
        for si in range(S):
            model.add(x[ei][si][di] == 0)

    # ── Hard constraint 4: min rest between shifts ────────────────────────────
    min_rest = config.min_rest_hours_between_shifts
    for ei in range(E):
        for di in range(D - 1):
            for si1, s1 in enumerate(shifts):
                for si2, s2 in enumerate(shifts):
                    rest = rest_hours_between(s1, dates[di], s2, dates[di + 1])
                    if rest < min_rest:
                        # Cannot work s1 on day d AND s2 on day d+1
                        model.add(x[ei][si1][di] + x[ei][si2][di + 1] <= 1)

    # ── Hard constraint 5: max consecutive working days ───────────────────────
    max_consec = config.max_consecutive_days
    if max_consec < D:
        for ei in range(E):
            for di in range(D - max_consec):
                # Cannot work ALL days in window of size max_consec+1
                model.add(
                    sum(
                        sum(x[ei][si][di + k] for si in range(S))
                        for k in range(max_consec + 1)
                    ) <= max_consec
                )

    # ── Hard constraint 6: max weekly hours ───────────────────────────────────
    # Group dates by ISO week
    weeks: dict[int, list[int]] = {}
    for di, d in enumerate(dates):
        wk = week_number(d)
        weeks.setdefault(wk, []).append(di)

    max_weekly = config.max_weekly_hours
    for ei in range(E):
        for wk_dates in weeks.values():
            # Sum of hours worked this week ≤ max_weekly
            # We use scaled integers (×10) for CP-SAT (no floats)
            hours_scaled = [
                int(shifts[si].duration_hours * 10)
                for si in range(S)
            ]
            model.add(
                sum(
                    x[ei][si][di] * hours_scaled[si]
                    for si in range(S)
                    for di in wk_dates
                ) <= int(max_weekly * 10)
            )

    # ── Hard constraint 7: max night shifts per week ──────────────────────────
    night_shifts = [si for si, s in enumerate(shifts) if s.shift_type == "night"]
    if night_shifts:
        max_night = config.max_night_shifts_per_week
        for ei in range(E):
            for wk_dates in weeks.values():
                model.add(
                    sum(x[ei][si][di] for si in night_shifts for di in wk_dates)
                    <= max_night
                )

    # ── Custom constraints from constraints table ─────────────────────────────
    for c in constraints:
        if not c.employee_id:
            continue

        ei = emp_idx.get(c.employee_id)
        if ei is None:
            continue

        if c.type == "fixed_shift" and c.shift_definition_id:
            # Employee must ONLY work this shift (never others)
            fixed_si = shift_idx.get(c.shift_definition_id)
            if fixed_si is not None:
                for si in range(S):
                    if si != fixed_si:
                        for di in range(D):
                            model.add(x[ei][si][di] == 0)

        elif c.type == "max_consecutive" and c.value:
            mc = int(c.value)
            if mc < D:
                for di in range(D - mc):
                    model.add(
                        sum(
                            sum(x[ei][si][di + k] for si in range(S))
                            for k in range(mc + 1)
                        ) <= mc
                    )

        elif c.type == "max_weekly_hours" and c.value:
            for wk_dates in weeks.values():
                hours_scaled = [int(shifts[si].duration_hours * 10) for si in range(S)]
                model.add(
                    sum(
                        x[ei][si][di] * hours_scaled[si]
                        for si in range(S) for di in wk_dates
                    ) <= int(c.value * 10)
                )

        elif c.type == "max_night_shifts" and c.value and night_shifts:
            for wk_dates in weeks.values():
                model.add(
                    sum(x[ei][si][di] for si in night_shifts for di in wk_dates)
                    <= int(c.value)
                )

    # Pair constraints (applied per day)
    pair_required = [(c.employee_id, c.target_employee_id, c.shift_definition_id)
                     for c in constraints
                     if c.type == "pair_required" and c.employee_id and c.target_employee_id]
    pair_forbidden = [(c.employee_id, c.target_employee_id, c.shift_definition_id)
                      for c in constraints
                      if c.type == "pair_forbidden" and c.employee_id and c.target_employee_id]

    for di in range(D):
        # Pair required: both work same shift on same day
        for (ea_id, eb_id, shift_id) in pair_required:
            ea = emp_idx.get(ea_id)
            eb = emp_idx.get(eb_id)
            if ea is None or eb is None:
                continue
            if shift_id:
                # Specific shift
                si = shift_idx.get(shift_id)
                if si is not None:
                    model.add(x[ea][si][di] == x[eb][si][di])
            else:
                # Any shift — both must work the same shift on this day
                for si in range(S):
                    model.add(x[ea][si][di] == x[eb][si][di])

        # Pair forbidden: cannot work same shift on same day
        for (ea_id, eb_id, shift_id) in pair_forbidden:
            ea = emp_idx.get(ea_id)
            eb = emp_idx.get(eb_id)
            if ea is None or eb is None:
                continue
            target_shifts = [shift_idx[shift_id]] if shift_id and shift_id in shift_idx else range(S)
            for si in target_shifts:
                model.add(x[ea][si][di] + x[eb][si][di] <= 1)

        # Min seniority: at least one senior per shift per day
        seniors = [ei for ei, e in enumerate(employees) if e.experience_level == "senior"]
        min_seniority_constraints = [
            c for c in constraints
            if c.type == "min_seniority"
        ]
        for c in min_seniority_constraints:
            target_shifts = [shift_idx[c.shift_definition_id]] if c.shift_definition_id and c.shift_definition_id in shift_idx else range(S)
            for si in target_shifts:
                if seniors:
                    model.add(sum(x[ei][si][di] for ei in seniors) >= 1)

    # ── Objective: minimize soft constraint violations ────────────────────────
    objective_terms = []

    # 1. Balance: minimize max-min hours difference across employees
    #    We use a linearization: minimize the maximum hours worked
    if config.balance_shift_distribution:
        max_hours_var = model.new_int_var(0, int(max_weekly * 10 * 6), "max_hours")
        hours_scaled = [int(shifts[si].duration_hours * 10) for si in range(S)]
        for ei in range(E):
            total_hours = sum(
                x[ei][si][di] * hours_scaled[si]
                for si in range(S) for di in range(D)
            )
            model.add(max_hours_var >= total_hours)
        objective_terms.append(max_hours_var)

    # 2. Consistency: penalize shift changes between consecutive days
    consistency = config.shift_consistency
    if consistency > 0:
        shift_change_penalties = []
        for ei in range(E):
            for di in range(D - 1):
                for si in range(S):
                    # worked[di] on shift si but NOT worked[di+1] on same shift
                    # = shift change penalty
                    switched = model.new_bool_var(f"switch_e{ei}_s{si}_d{di}")
                    model.add(x[ei][si][di] - x[ei][si][di + 1] <= switched)
                    shift_change_penalties.append(switched)
        # Weight by consistency level - repeat the list instead of multiply
        # to avoid LinearExpr * int type issues in some OR-Tools versions
        for _ in range(consistency):
            for p in shift_change_penalties:
                objective_terms.append(p)

    if objective_terms:
        # Use OR-Tools LinearExpr.sum() to avoid Python sum() type issues
        from ortools.sat.python.cp_model import LinearExpr
        model.minimize(cp_model.LinearExpr.sum(objective_terms))

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload.solver_time_limit_seconds
    solver.parameters.num_search_workers = 4  # parallel search
    solver.parameters.log_search_progress = False
    # solver.parameters.stop_after_first_solution = True # quicker but not optimised

    status = solver.solve(model)
    _STATUS_NAMES = {
        cp_model.UNKNOWN: 'UNKNOWN',
        cp_model.MODEL_INVALID: 'MODEL_INVALID',
        cp_model.FEASIBLE: 'FEASIBLE',
        cp_model.INFEASIBLE: 'INFEASIBLE',
        cp_model.OPTIMAL: 'OPTIMAL',
    }
    status_name = _STATUS_NAMES.get(status, f'STATUS_{status}')
    solve_time = getattr(solver, 'wall_time', 0) or getattr(solver, 'WallTime', lambda: 0)()
    logger.info(f"Solver status: {status_name} in {solve_time:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning(f"No feasible solution found: {status_name}")
        return ShiftsResponse(
            assignments=[],
            violations=[Violation(
                type="infeasible",
                employee_id="",
                employee_name="",
                date=dates[0] if dates else "",
                message=f"CP-SAT could not find a feasible solution ({status_name}). "
                        f"Try relaxing constraints (fewer slots, fewer constraints, longer time limit)."
            )],
            stats=SolverStats(
                total_slots=sum(slots.get(s.id, 1) for s in shifts) * D,
                filled_slots=0,
                hours_per_employee={e.id: 0.0 for e in employees},
                solver_status=status_name,
                solve_time_seconds=solve_time,
            )
        )

    # ── Extract solution ───────────────────────────────────────────────────────
    assignments = []
    hours_per_employee: dict[str, float] = {e.id: 0.0 for e in employees}

    for ei, emp in enumerate(employees):
        for si, shift in enumerate(shifts):
            for di, d in enumerate(dates):
                if solver.value(x[ei][si][di]) == 1:
                    assignments.append(Assignment(
                        employee_id=emp.id,
                        shift_definition_id=shift.id,
                        date=d,
                    ))
                    hours_per_employee[emp.id] += shift.duration_hours

    # Check for violations (slots not fully covered — shouldn't happen if FEASIBLE)
    violations = []
    for si, shift in enumerate(shifts):
        required = slots.get(shift.id, config.min_employees_per_shift)
        for di, d in enumerate(dates):
            covered = sum(solver.value(x[ei][si][di]) for ei in range(E))
            if covered < required:
                violations.append(Violation(
                    type="uncovered_slot",
                    employee_id="",
                    employee_name="",
                    date=d,
                    message=f"Shift '{shift.name}' on {d}: {covered}/{required} employees assigned"
                ))

    return ShiftsResponse(
        assignments=assignments,
        violations=violations,
        stats=SolverStats(
            total_slots=sum(slots.get(s.id, 1) for s in shifts) * D,
            filled_slots=len(assignments),
            hours_per_employee=hours_per_employee,
            solver_status=status_name,
            solve_time_seconds=solve_time,
        )
    )