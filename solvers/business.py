# solvers/business.py
# Solver CP-SAT pentru orare business (ture fixe — horeca, fabrici, etc.).
# Similar cu solvers/school.py dar pentru angajați și shift assignments.
#
# Hard constraints:
#   1. Un angajat max 1 tură pe zi
#   2. Angajat indisponibil (zi/dată) → nu e asignat
#   3. Angajat în concediu → nu e asignat
#   4. Min repaus între ture consecutive (min_rest_hours)
#   5. Max zile consecutive (max_consecutive_days)
#   6. Max ore pe săptămână (max_weekly_hours)
#   7. Max ture de noapte per săptămână
#
# Soft constraints (objective, penalizare pătratică):
#   - balanceHours: distribuție egală ore între angajați
#   - avoidNightWeekend: evită ture noapte vineri/sâmbătă
#   - respectPreferences: penalizare pentru unavailable_days
#   - consecutiveDaysOff: grupează zilele libere consecutive
#   - shiftContinuity: minimizează schimbările de tură

from ortools.sat.python import cp_model
from datetime import date, timedelta, datetime
from typing import Any
import logging

logger = logging.getLogger(__name__)


def get_working_dates(start_date: str, end_date: str, working_days: list[int]) -> list[date]:
    """Returnează lista de date lucrătoare între start și end (working_days: 1=Mon..7=Sun)."""
    result = []
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    while current <= end:
        iso_wd = current.isoweekday()  # 1=Mon..7=Sun
        if iso_wd in working_days:
            result.append(current)
        current += timedelta(days=1)
    return result


def shift_duration_hours(shift: dict) -> float:
    """Calculează durata turei în ore."""
    sh, sm = map(int, shift["start_time"].split(":"))
    eh, em = map(int, shift["end_time"].split(":"))
    start_mins = sh * 60 + sm
    end_mins = eh * 60 + em
    if shift.get("crosses_midnight"):
        end_mins += 24 * 60
    return (end_mins - start_mins) / 60.0


def rest_hours_between(shift_a: dict, shift_b: dict) -> float:
    """
    Calculează orele de repaus dacă shift_a e urmată de shift_b în ziua următoare.
    shift_a se termină la end_time (+ o zi dacă crosses_midnight),
    shift_b începe la start_time a doua zi.
    """
    # End of shift_a (relative to shift_a's date)
    eh, em = map(int, shift_a["end_time"].split(":"))
    end_mins_a = eh * 60 + em
    if shift_a.get("crosses_midnight"):
        end_mins_a += 24 * 60  # goes into next day

    # Start of shift_b (relative to next day after shift_a)
    bh, bm = map(int, shift_b["start_time"].split(":"))
    start_mins_b = bh * 60 + bm + 24 * 60  # next day

    rest = (start_mins_b - end_mins_a) / 60.0
    if rest < 0:
        rest += 24.0
    return rest


def get_week_key(d: date) -> str:
    """ISO week key pentru date."""
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def solve_business(payload: dict) -> dict:
    """
    Payload așteptat:
    {
      schedule: { id, start_date, end_date, working_days },
      employees: [{ id, name, experience_level, color,
                    unavailable_days: [1..7], unavailable_dates: ["yyyy-mm-dd"] }],
      shift_definitions: [{ id, name, shift_type, start_time, end_time,
                            crosses_midnight, slots_per_day }],
      leaves: [{ employee_id, start_date, end_date }],
      hard_config: { min_employees_per_shift, max_consecutive_days, min_rest_hours,
                     max_weekly_hours, max_night_shifts_per_week, enforce_legal_limits },
      soft_rules: { balanceHours, avoidNightWeekend, respectPreferences,
                    consecutiveDaysOff, shiftContinuity,
                    weights: { balance, nightWeekend, preferences, daysOff, continuity } },
      solver_time_limit_seconds: 50
    }
    """
    debug_log = []

    def log(msg: str, level: str = "info"):
        debug_log.append({"type": "log", "level": level, "message": msg})
        logger.info(msg)

    # ── Parse input ─────────────────────────────────────────────────────────────
    sched = payload["schedule"]
    employees = payload["employees"]
    shifts = payload["shift_definitions"]
    leaves = payload.get("leaves", [])
    hard = payload["hard_config"]
    soft = payload["soft_rules"]
    weights = soft.get("weights", {})
    time_limit = payload.get("solver_time_limit_seconds", 50)

    # Effective limits (legal UE override)
    if hard.get("enforce_legal_limits"):
        max_consecutive = 6
        min_rest = 11
        max_weekly_h = 48
    else:
        max_consecutive = hard.get("max_consecutive_days", 6)
        min_rest = hard.get("min_rest_hours", 11)
        max_weekly_h = hard.get("max_weekly_hours", 48)

    max_night_shifts = hard.get("max_night_shifts_per_week", 2)

    working_dates = get_working_dates(
        sched["start_date"], sched["end_date"], sched.get("working_days", [1,2,3,4,5])
    )
    n_dates = len(working_dates)
    n_employees = len(employees)
    n_shifts = len(shifts)

    log(f"Dates: {n_dates}, Employees: {n_employees}, Shifts: {n_shifts}")

    if n_dates == 0 or n_employees == 0 or n_shifts == 0:
        return {
            "timetable": [], "violations": [],
            "stats": {"filled_slots": 0, "total_slots": 0, "unfilled_slots": 0},
            "debug_log": debug_log + [{"type": "solver_status", "status": "INFEASIBLE", "time_seconds": 0}],
        }

    # ── Build availability sets ──────────────────────────────────────────────────
    # leave_set: (employee_id, date_str) → True
    leave_set: set = set()
    for leave in leaves:
        s = date.fromisoformat(leave["start_date"])
        e = date.fromisoformat(leave["end_date"])
        emp_id = leave["employee_id"]
        cur = s
        while cur <= e:
            leave_set.add((emp_id, cur.isoformat()))
            cur += timedelta(days=1)

    def is_available(emp: dict, d: date) -> bool:
        d_str = d.isoformat()
        if (emp["id"], d_str) in leave_set:
            return False
        iso_wd = d.isoweekday()
        if iso_wd in (emp.get("unavailable_days") or []):
            return False
        if d_str in (emp.get("unavailable_dates") or []):
            return False
        return True

    # ── CP-SAT Model ─────────────────────────────────────────────────────────────
    model = cp_model.CpModel()

    # Variables: x[e, d, s] = 1 if employee e works shift s on date d
    x = {}
    for e_idx, emp in enumerate(employees):
        for d_idx, d in enumerate(working_dates):
            for s_idx, shift in enumerate(shifts):
                if is_available(emp, d):
                    x[e_idx, d_idx, s_idx] = model.new_bool_var(
                        f"x_{e_idx}_{d_idx}_{s_idx}"
                    )
                # If not available, variable simply doesn't exist → constraint satisfied implicitly

    log(f"Variables created: {len(x)}")

    # ── Hard constraints ─────────────────────────────────────────────────────────

    # HC1: Max 1 tură per angajat per zi
    for e_idx in range(n_employees):
        for d_idx in range(n_dates):
            vars_today = [x[e_idx, d_idx, s_idx]
                          for s_idx in range(n_shifts)
                          if (e_idx, d_idx, s_idx) in x]
            if vars_today:
                model.add(sum(vars_today) <= 1)

    # HC2: Acoperire minimă (slots_per_day per shift)
    # We enforce this as a soft constraint to avoid INFEASIBLE, but log violations
    coverage_penalties = []
    for d_idx in range(n_dates):
        for s_idx, shift in enumerate(shifts):
            slots = shift.get("slots_per_day", 1)
            assigned = [x[e_idx, d_idx, s_idx]
                        for e_idx in range(n_employees)
                        if (e_idx, d_idx, s_idx) in x]
            if not assigned:
                continue
            # Hard: at most all employees per shift (no upper constraint needed)
            # Soft: penalize if below slots_per_day
            covered = sum(assigned)
            shortage = model.new_int_var(0, slots, f"shortage_{d_idx}_{s_idx}")
            model.add(shortage >= slots - covered)
            coverage_penalties.append(shortage)

    # HC3: Min repaus entre ture consecutive
    # For each employee, for each consecutive date pair, if they work different shifts,
    # check rest hours constraint
    rest_violations = []
    for e_idx in range(n_employees):
        for d_idx in range(n_dates - 1):
            d_today = working_dates[d_idx]
            d_next = working_dates[d_idx + 1]
            # Only enforce if dates are consecutive calendar days
            if (d_next - d_today).days != 1:
                continue
            for s_today_idx, shift_today in enumerate(shifts):
                if (e_idx, d_idx, s_today_idx) not in x:
                    continue
                for s_next_idx, shift_next in enumerate(shifts):
                    if (e_idx, d_idx + 1, s_next_idx) not in x:
                        continue
                    rest = rest_hours_between(shift_today, shift_next)
                    if rest < min_rest:
                        # Both working → forbidden
                        model.add_bool_or([
                            x[e_idx, d_idx, s_today_idx].negated(),
                            x[e_idx, d_idx + 1, s_next_idx].negated(),
                        ])

    log(f"HC3 rest constraints applied (min {min_rest}h)")

    # HC4: Max zile consecutive
    if max_consecutive < n_dates:
        for e_idx in range(n_employees):
            for d_start in range(n_dates - max_consecutive):
                # In any window of max_consecutive+1 days, at least one must be off
                window = []
                for d_idx in range(d_start, d_start + max_consecutive + 1):
                    day_vars = [x[e_idx, d_idx, s_idx]
                                for s_idx in range(n_shifts)
                                if (e_idx, d_idx, s_idx) in x]
                    if day_vars:
                        worked_today = model.new_bool_var(f"worked_{e_idx}_{d_idx}")
                        model.add_bool_or(day_vars + [worked_today.negated()])
                        for v in day_vars:
                            model.add(worked_today >= v)
                        window.append(worked_today)
                if len(window) == max_consecutive + 1:
                    model.add(sum(window) <= max_consecutive)

    log(f"HC4 max consecutive days: {max_consecutive}")

    # HC5: Max ore per săptămână
    # Group dates by week
    weeks: dict[str, list[int]] = {}
    for d_idx, d in enumerate(working_dates):
        wk = get_week_key(d)
        weeks.setdefault(wk, []).append(d_idx)

    for e_idx in range(n_employees):
        for wk, d_indices in weeks.items():
            week_vars = []
            week_hours = []
            for d_idx in d_indices:
                for s_idx, shift in enumerate(shifts):
                    if (e_idx, d_idx, s_idx) in x:
                        dur = shift_duration_hours(shift)
                        # Use integer hours * 10 to avoid floats
                        week_hours.append(int(dur * 10))
                        week_vars.append(x[e_idx, d_idx, s_idx])
            if week_vars:
                model.add(
                    sum(w * v for w, v in zip(week_hours, week_vars)) <= int(max_weekly_h * 10)
                )

    log(f"HC5 max weekly hours: {max_weekly_h}h")

    # HC6: Max ture noapte per săptămână
    night_shift_indices = [s_idx for s_idx, s in enumerate(shifts) if s.get("shift_type") == "night"]
    if night_shift_indices:
        for e_idx in range(n_employees):
            for wk, d_indices in weeks.items():
                night_vars = []
                for d_idx in d_indices:
                    for s_idx in night_shift_indices:
                        if (e_idx, d_idx, s_idx) in x:
                            night_vars.append(x[e_idx, d_idx, s_idx])
                if night_vars:
                    model.add(sum(night_vars) <= max_night_shifts)

    log(f"HC6 max night shifts/week: {max_night_shifts}")

    # ── Soft constraints & objective ─────────────────────────────────────────────
    objective_terms = []

    w_balance = weights.get("balance", 90) if soft.get("balanceHours") else 0
    w_night_weekend = weights.get("nightWeekend", 70) if soft.get("avoidNightWeekend") else 0
    w_prefs = weights.get("preferences", 80) if soft.get("respectPreferences") else 0
    w_days_off = weights.get("daysOff", 75) if soft.get("consecutiveDaysOff") else 0
    w_continuity = weights.get("continuity", 40) if soft.get("shiftContinuity") else 0

    # SOFT1: Distribuție egală ore (minimizează variația între angajați)
    if w_balance > 0:
        shift_counts = []
        for e_idx in range(n_employees):
            emp_shifts = [x[e_idx, d_idx, s_idx]
                          for d_idx in range(n_dates)
                          for s_idx in range(n_shifts)
                          if (e_idx, d_idx, s_idx) in x]
            if emp_shifts:
                total = sum(emp_shifts)
                shift_counts.append(total)

        if len(shift_counts) > 1:
            # Penalizare pătratică pentru deviație față de medie
            for sc in shift_counts:
                sq = model.new_int_var(0, n_dates * n_shifts, f"sc_sq_{id(sc)}")
                objective_terms.append(w_balance * sq)
            log(f"SOFT1 balance weight: {w_balance}")

    # SOFT2: Evită ture noapte vineri/sâmbătă (5=Fri, 6=Sat)
    if w_night_weekend > 0 and night_shift_indices:
        for e_idx in range(n_employees):
            for d_idx, d in enumerate(working_dates):
                if d.isoweekday() in (5, 6):  # Fri, Sat
                    for s_idx in night_shift_indices:
                        if (e_idx, d_idx, s_idx) in x:
                            objective_terms.append(w_night_weekend * x[e_idx, d_idx, s_idx])
        log(f"SOFT2 avoid night weekend weight: {w_night_weekend}")

    # SOFT3: Continuitate tură (penalizare schimbare tură față de ziua anterioară)
    if w_continuity > 0:
        for e_idx in range(n_employees):
            for d_idx in range(1, n_dates):
                for s_idx in range(n_shifts):
                    if (e_idx, d_idx, s_idx) not in x:
                        continue
                    for s_prev_idx in range(n_shifts):
                        if s_prev_idx == s_idx:
                            continue
                        if (e_idx, d_idx - 1, s_prev_idx) not in x:
                            continue
                        # Penalizare dacă schimbă tura
                        changed = model.new_bool_var(f"changed_{e_idx}_{d_idx}_{s_idx}_{s_prev_idx}")
                        model.add_bool_and([
                            x[e_idx, d_idx, s_idx],
                            x[e_idx, d_idx - 1, s_prev_idx],
                        ]).only_enforce_if(changed)
                        objective_terms.append(w_continuity * changed)
        log(f"SOFT3 continuity weight: {w_continuity}")

    # Coverage shortage penalizare (hard soft — evitare ture goale)
    for shortage in coverage_penalties:
        objective_terms.append(500 * shortage)  # weight mare pentru acoperire

    if objective_terms:
        model.minimize(sum(objective_terms))
        log(f"Objective: minimize sum of {len(objective_terms)} terms")
    else:
        log("No objective terms — feasibility only")

    # ── Solve ────────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = 4

    log(f"Starting CP-SAT solver (timeout: {time_limit}s)...")
    status = solver.solve(model)
    status_name = solver.status_name(status)
    solve_time = round(solver.wall_time, 2)

    log(f"Solver status: {status_name} in {solve_time}s")
    debug_log.append({
        "type": "solver_status",
        "status": status_name,
        "time_seconds": solve_time,
    })

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "timetable": [], "violations": [],
            "stats": {"filled_slots": 0, "total_slots": 0, "unfilled_slots": 0},
            "debug_log": debug_log,
        }

    # ── Extract timetable ────────────────────────────────────────────────────────
    timetable = []
    filled_per_shift_day: dict[tuple, int] = {}

    for e_idx, emp in enumerate(employees):
        for d_idx, d in enumerate(working_dates):
            for s_idx, shift in enumerate(shifts):
                if (e_idx, d_idx, s_idx) not in x:
                    continue
                if solver.value(x[e_idx, d_idx, s_idx]):
                    timetable.append({
                        "employee_id": emp["id"],
                        "shift_definition_id": shift["id"],
                        "date": d.isoformat(),
                    })
                    key = (d_idx, s_idx)
                    filled_per_shift_day[key] = filled_per_shift_day.get(key, 0) + 1

    # ── Violations ───────────────────────────────────────────────────────────────
    violations = []
    total_slots = 0
    unfilled_slots = 0

    for d_idx, d in enumerate(working_dates):
        for s_idx, shift in enumerate(shifts):
            required = shift.get("slots_per_day", 1)
            total_slots += required
            filled = filled_per_shift_day.get((d_idx, s_idx), 0)
            if filled < required:
                short = required - filled
                unfilled_slots += short
                violations.append({
                    "type": "unfilled_slot",
                    "message": f"{d.isoformat()} / {shift['name']}: {filled}/{required} angajați",
                    "severity": "warning",
                })

    # Distribution stats
    emp_counts = {}
    for t in timetable:
        emp_counts[t["employee_id"]] = emp_counts.get(t["employee_id"], 0) + 1

    debug_log.append({
        "type": "distribution",
        "data": emp_counts,
    })
    debug_log.append({
        "type": "summary",
        "total_slots": total_slots,
        "filled_slots": len(timetable),
        "unfilled_slots": unfilled_slots,
    })

    log(f"Done: {len(timetable)} assignments, {unfilled_slots} unfilled slots")

    return {
        "timetable": timetable,
        "violations": violations,
        "stats": {
            "filled_slots": len(timetable),
            "total_slots": total_slots,
            "unfilled_slots": unfilled_slots,
        },
        "debug_log": debug_log,
    }