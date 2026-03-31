# orarpro-solver/solvers/business.py  (v2)
# Endpoint: POST /solve/business
#
# Loguri structurate identic cu school.py:
#   === Business solver v2 ===
#   ANGAJAT / TURĂ info (echivalent TEACHER / CLASS din school)
#   [CHECk] disponibilitate per angajat
#   [HARD1..7] constrângeri hard cu număr constrângeri
#   === CONSTRAINTS SUMMARY ===
#   [SOFT1..4] cu termeni și weights
#   === PRE-SOLVE CHECK ===
#   === POST-SOLVE VALIDATION ===
#   [RESULT] per angajat: X ture → TurăA:Nz, TurăB:Mz
#   === Done ===

from __future__ import annotations
from ortools.sat.python import cp_model
from datetime import date, timedelta
from collections import defaultdict, Counter
from typing import Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


# ── Input models ──────────────────────────────────────────────────────────────

class EmployeeConfig(BaseModel):
    id: str
    name: str = ''
    experience_level: str = 'mid'
    color: Optional[str] = None
    unavailable_days: list[int] = []      # ISO weekday 1=Mon..7=Sun
    unavailable_dates: list[str] = []     # "yyyy-mm-dd"

class ShiftDefinition(BaseModel):
    id: str
    name: str
    shift_type: str = 'custom'
    start_time: str
    end_time: str
    crosses_midnight: bool = False
    slots_per_day: int = 1
    duration_hours: Optional[float] = None

class LeaveEntry(BaseModel):
    employee_id: str
    start_date: str
    end_date: str

class UnavailabilityEntry(BaseModel):
    employee_id: str
    date: Optional[str] = None
    day_of_week: Optional[int] = None

class SoftRules(BaseModel):
    balanceHours: bool = True
    avoidNightWeekend: bool = True
    respectPreferences: bool = False
    consecutiveDaysOff: bool = True
    shiftContinuity: bool = False
    weights: dict = {
        'balance': 90, 'nightWeekend': 70,
        'preferences': 80, 'daysOff': 75, 'continuity': 40,
    }

class ShiftsConfig(BaseModel):
    min_employees_per_shift: int = 1
    max_consecutive_days: int = 6
    min_rest_hours_between_shifts: float = 11.0
    max_weekly_hours: float = 48.0
    max_night_shifts_per_week: int = 3
    enforce_legal_limits: bool = True
    balance_shift_distribution: bool = True
    shift_consistency: int = 2

class ShiftsRequest(BaseModel):
    schedule_id: str = ''
    employees: list[EmployeeConfig] = []
    shift_definitions: list[ShiftDefinition] = []
    working_dates: list[str] = []
    slots_per_shift: dict[str, int] = {}   # legacy override
    constraints: list[dict] = []           # pair_required/pair_forbidden legacy
    leaves: list[LeaveEntry] = []
    unavailability: list[UnavailabilityEntry] = []
    config: ShiftsConfig = ShiftsConfig()
    soft_rules: SoftRules = SoftRules()
    solver_time_limit_seconds: int = 50


# ── Output models ─────────────────────────────────────────────────────────────

class AssignmentResult(BaseModel):
    employee_id: str
    shift_definition_id: str
    date: str

class ShiftsViolation(BaseModel):
    type: str
    message: str
    employee_id: Optional[str] = None
    date: Optional[str] = None
    severity: str = 'warning'

class ShiftsStats(BaseModel):
    total_slots: int
    filled_slots: int
    unfilled_slots: int
    solver_status: str
    solve_time_seconds: float
    objective_value: Optional[float] = None

class ShiftsResponse(BaseModel):
    assignments: list[AssignmentResult]
    violations: list[ShiftsViolation]
    stats: ShiftsStats
    debug_log: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_hhmm(t: str) -> tuple[int, int]:
    """Parsează HH:MM sau HH:MM:SS → (hour, minute)."""
    parts = t.split(':')
    return int(parts[0]), int(parts[1])


def shift_duration_hours(shift: ShiftDefinition) -> float:
    if shift.duration_hours is not None:
        return shift.duration_hours
    sh, sm = parse_hhmm(shift.start_time)
    eh, em = parse_hhmm(shift.end_time)
    start_mins = sh * 60 + sm
    end_mins   = eh * 60 + em
    if shift.crosses_midnight:
        end_mins += 24 * 60
    return (end_mins - start_mins) / 60.0


def rest_hours_between(shift_a: ShiftDefinition, shift_b: ShiftDefinition) -> float:
    """Ore de repaus dacă shift_a se termină și shift_b începe a doua zi."""
    eh, em       = parse_hhmm(shift_a.end_time)
    end_mins_a   = eh * 60 + em
    if shift_a.crosses_midnight:
        end_mins_a += 24 * 60
    bh, bm         = parse_hhmm(shift_b.start_time)
    start_mins_b   = bh * 60 + bm + 24 * 60
    rest = (start_mins_b - end_mins_a) / 60.0
    if rest < 0:
        rest += 24.0
    return rest


def get_week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ── Solver ────────────────────────────────────────────────────────────────────

def solve_shifts(payload: ShiftsRequest) -> ShiftsResponse:
    debug_log: list[dict] = []
    cfg     = payload.config
    soft    = payload.soft_rules
    weights = soft.weights

    # ── identic cu school: prima linie de log ─────────────────────────────────
    logger.info("=== Business solver v2 ===")
    logger.info(f"  {len(payload.employees)} angajați, "
                f"{len(payload.shift_definitions)} ture, "
                f"{len(payload.working_dates)} zile")

    def log(msg: str, level: str = "info"):
        debug_log.append({"type": "log", "level": level, "message": msg})
        logger.info(msg)

    # ── Effective limits ──────────────────────────────────────────────────────
    if cfg.enforce_legal_limits:
        max_consecutive = 6
        min_rest        = 11.0
        max_weekly_h    = 48.0
    else:
        max_consecutive = cfg.max_consecutive_days
        min_rest        = cfg.min_rest_hours_between_shifts
        max_weekly_h    = cfg.max_weekly_hours
    max_night = cfg.max_night_shifts_per_week

    log(f"Config: max_consecutive={max_consecutive}, min_rest={min_rest}h, "
        f"max_weekly={max_weekly_h}h, max_night/săpt={max_night}, "
        f"legal_UE={'DA' if cfg.enforce_legal_limits else 'NU'}")

    # ── Parse dates ───────────────────────────────────────────────────────────
    working_dates: list[date] = []
    for ds in payload.working_dates:
        try:
            working_dates.append(date.fromisoformat(ds))
        except ValueError:
            log(f"Data invalidă ignorată: {ds}", "warn")

    n_dates     = len(working_dates)
    n_employees = len(payload.employees)
    n_shifts    = len(payload.shift_definitions)

    log(f"Date: {n_dates}, Angajați: {n_employees}, Ture: {n_shifts}")

    # ── Effective slots (legacy merge) ────────────────────────────────────────
    effective_slots: dict[str, int] = {}
    for shift in payload.shift_definitions:
        legacy = payload.slots_per_shift.get(shift.id)
        effective_slots[shift.id] = legacy if legacy is not None else shift.slots_per_day

    # ── Log angajați — identic cu school log teachers ─────────────────────────
    for e in payload.employees:
        log(f"  ANGAJAT {e.name}: unavail_days={e.unavailable_days}, "
            f"unavail_dates={len(e.unavailable_dates)} date")

    # ── Log ture — identic cu school log classes ──────────────────────────────
    for s in payload.shift_definitions:
        dur = shift_duration_hours(s)
        log(f"  TURĂ {s.name}: {s.start_time}–{s.end_time} "
            f"({'trece miezul nopții' if s.crosses_midnight else 'normală'}) "
            f"{dur:.1f}h, tip={s.shift_type}, slots={effective_slots[s.id]}/zi")
        debug_log.append({
            "type": "shift_info", "shift_id": s.id, "name": s.name,
            "start_time": s.start_time, "end_time": s.end_time,
            "duration_h": dur, "slots_per_day": effective_slots[s.id],
            "shift_type": s.shift_type,
        })

    if n_dates == 0 or n_employees == 0 or n_shifts == 0:
        log("Date/angajați/ture lipsă — returnez INFEASIBLE", "warn")
        return ShiftsResponse(
            assignments=[], violations=[],
            stats=ShiftsStats(total_slots=0, filled_slots=0, unfilled_slots=0,
                              solver_status='INFEASIBLE', solve_time_seconds=0),
            debug_log=debug_log,
        )

    # ── Availability ──────────────────────────────────────────────────────────
    leave_set: set[tuple[str, str]] = set()
    for leave in payload.leaves:
        s = date.fromisoformat(leave.start_date)
        e = date.fromisoformat(leave.end_date)
        cur = s
        while cur <= e:
            leave_set.add((leave.employee_id, cur.isoformat()))
            cur += timedelta(days=1)

    unavail_dates_set: set[tuple[str, str]] = set()
    unavail_days_map: dict[str, set[int]] = defaultdict(set)
    for u in payload.unavailability:
        if u.date:
            unavail_dates_set.add((u.employee_id, u.date))
        if u.day_of_week is not None:
            unavail_days_map[u.employee_id].add(u.day_of_week)

    def is_available(emp: EmployeeConfig, d: date) -> bool:
        d_str  = d.isoformat()
        iso_wd = d.isoweekday()
        if (emp.id, d_str) in leave_set:             return False
        if iso_wd in (emp.unavailable_days or []):   return False
        if d_str in (emp.unavailable_dates or []):   return False
        if (emp.id, d_str) in unavail_dates_set:     return False
        if iso_wd in unavail_days_map.get(emp.id, set()): return False
        return True

    # ── Variables ─────────────────────────────────────────────────────────────
    model = cp_model.CpModel()
    x: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for e_idx, emp in enumerate(payload.employees):
        for d_idx, d in enumerate(working_dates):
            for s_idx, shift in enumerate(payload.shift_definitions):
                if is_available(emp, d):
                    x[e_idx, d_idx, s_idx] = model.new_bool_var(f"x_{e_idx}_{d_idx}_{s_idx}")

    total_vars = len(x)
    log(f"Variabile create: {total_vars}")
    debug_log.append({"type": "variables", "count": total_vars})

    # ── Log disponibilitate — identic cu school log assignments ───────────────
    for e_idx, emp in enumerate(payload.employees):
        avail = sum(1 for d_idx in range(n_dates) for s_idx in range(n_shifts)
                    if (e_idx, d_idx, s_idx) in x)
        ok = avail >= n_shifts
        log(f"  [CHECK] {emp.name}: {avail} sloturi disponibile din "
            f"{n_dates * n_shifts} maxime "
            f"({'✓ OK' if ok else '✗ IMPOSIBIL — nicio disponibilitate'})")
        debug_log.append({
            "type": "employee_check", "employee_id": emp.id, "name": emp.name,
            "available_slots": avail, "max_slots": n_dates * n_shifts,
        })

    # ── HARD 1: Max 1 tură/angajat/zi ────────────────────────────────────────
    for e_idx in range(n_employees):
        for d_idx in range(n_dates):
            day_vars = [x[e_idx, d_idx, s_idx] for s_idx in range(n_shifts)
                        if (e_idx, d_idx, s_idx) in x]
            if len(day_vars) > 1:
                model.add(sum(day_vars) <= 1)
    log("[HARD1] max 1 tură/angajat/zi — aplicat")

    # ── HARD 2: Acoperire minimă (soft-penalty) ───────────────────────────────
    coverage_penalties: list[cp_model.IntVar] = []
    for d_idx in range(n_dates):
        for s_idx, shift in enumerate(payload.shift_definitions):
            slots = effective_slots[shift.id]
            assigned = [x[e_idx, d_idx, s_idx] for e_idx in range(n_employees)
                        if (e_idx, d_idx, s_idx) in x]
            if not assigned:
                continue
            shortage = model.new_int_var(0, slots, f"shortage_{d_idx}_{s_idx}")
            model.add(shortage >= slots - sum(assigned))
            coverage_penalties.append(shortage)
    log(f"[HARD2] acoperire minimă: {len(coverage_penalties)} perechi tură/zi")

    # ── HARD 3: Min repaus între ture consecutive ─────────────────────────────
    rest_cnt = 0
    for e_idx in range(n_employees):
        for d_idx in range(n_dates - 1):
            d_today = working_dates[d_idx]
            d_next  = working_dates[d_idx + 1]
            if (d_next - d_today).days != 1:
                continue
            for s_a, shift_a in enumerate(payload.shift_definitions):
                if (e_idx, d_idx, s_a) not in x:
                    continue
                for s_b, shift_b in enumerate(payload.shift_definitions):
                    if (e_idx, d_idx + 1, s_b) not in x:
                        continue
                    if rest_hours_between(shift_a, shift_b) < min_rest:
                        model.add_bool_or([x[e_idx, d_idx, s_a].negated(),
                                           x[e_idx, d_idx + 1, s_b].negated()])
                        rest_cnt += 1
    log(f"[HARD3] min repaus {min_rest}h: {rest_cnt} constrângeri")

    # ── HARD 4: Max zile consecutive ─────────────────────────────────────────
    consec_cnt = 0
    if max_consecutive < n_dates:
        for e_idx in range(n_employees):
            for d_start in range(n_dates - max_consecutive):
                window = []
                for d_idx in range(d_start, d_start + max_consecutive + 1):
                    day_vars = [x[e_idx, d_idx, s_idx] for s_idx in range(n_shifts)
                                if (e_idx, d_idx, s_idx) in x]
                    if day_vars:
                        worked = model.new_bool_var(f"worked_{e_idx}_{d_idx}_{d_start}")
                        model.add_bool_or(day_vars + [worked.negated()])
                        for v in day_vars: model.add(worked >= v)
                        window.append(worked)
                if len(window) == max_consecutive + 1:
                    model.add(sum(window) <= max_consecutive)
                    consec_cnt += 1
    log(f"[HARD4] max {max_consecutive} zile consecutive: {consec_cnt} ferestre")

    # ── HARD 5: Max ore/săptămână ─────────────────────────────────────────────
    weeks: dict[str, list[int]] = {}
    for d_idx, d in enumerate(working_dates):
        weeks.setdefault(get_week_key(d), []).append(d_idx)
    weekly_cnt = 0
    for e_idx in range(n_employees):
        for wk, d_indices in weeks.items():
            terms = [(int(shift_duration_hours(payload.shift_definitions[s_idx]) * 10),
                      x[e_idx, d_idx, s_idx])
                     for d_idx in d_indices for s_idx in range(n_shifts)
                     if (e_idx, d_idx, s_idx) in x]
            if terms:
                model.add(sum(d * v for d, v in terms) <= int(max_weekly_h * 10))
                weekly_cnt += 1
    log(f"[HARD5] max {max_weekly_h}h/săpt: {weekly_cnt} constrângeri")

    # ── HARD 6: Max ture noapte/săptămână ────────────────────────────────────
    night_indices = [s_idx for s_idx, s in enumerate(payload.shift_definitions)
                     if s.shift_type == 'night']
    night_cnt = 0
    if night_indices:
        for e_idx in range(n_employees):
            for wk, d_indices in weeks.items():
                night_vars = [x[e_idx, d_idx, s_idx]
                              for d_idx in d_indices for s_idx in night_indices
                              if (e_idx, d_idx, s_idx) in x]
                if night_vars:
                    model.add(sum(night_vars) <= max_night)
                    night_cnt += 1
    log(f"[HARD6] max {max_night} ture noapte/săpt: {night_cnt} constrângeri")

    # ── HARD 7: pair_forbidden (legacy) ──────────────────────────────────────
    emp_idx_map = {e.id: i for i, e in enumerate(payload.employees)}
    pair_forb = [c for c in payload.constraints
                 if c.get('type') == 'pair_forbidden' and c.get('is_active', True)]
    for c in pair_forb:
        ea = emp_idx_map.get(c.get('employee_id', ''))
        eb = emp_idx_map.get(c.get('target_employee_id', ''))
        if ea is None or eb is None: continue
        for d_idx in range(n_dates):
            for s_idx in range(n_shifts):
                va, vb = x.get((ea, d_idx, s_idx)), x.get((eb, d_idx, s_idx))
                if va and vb: model.add_bool_or([va.negated(), vb.negated()])
    if pair_forb:
        log(f"[HARD7] pair_forbidden: {len(pair_forb)} perechi interzise")

    # ── Constraints summary — identic cu school ───────────────────────────────
    log("=== CONSTRAINTS SUMMARY ===")
    log(f"  [HARD] max1TurăPeZi: fiecare angajat max 1 tură/zi")
    log(f"  [HARD] repausMinim: {min_rest}h → {rest_cnt} constrângeri")
    log(f"  [HARD] zileConsecutive: max {max_consecutive} → {consec_cnt} ferestre")
    log(f"  [HARD] oreSăptămână: max {max_weekly_h}h → {weekly_cnt} constrângeri")
    log(f"  [HARD] turăNoapte: max {max_night}/săpt → {night_cnt} constrângeri")
    log(f"  [HARD] acoperire: soft-penalty w=500 → {len(coverage_penalties)} variabile")
    for t in payload.employees:
        avail_cnt = sum(1 for d_idx in range(n_dates) for s_idx in range(n_shifts)
                        if (payload.employees.index(t), d_idx, s_idx) in x)
        log(f"  [HARD] {t.name}: {avail_cnt} sloturi disponibile")

    # ── SOFT objective ────────────────────────────────────────────────────────
    objective: list = []
    active_soft: list[str] = []

    w_balance    = weights.get('balance', 90)    if soft.balanceHours      else 0
    w_night_wknd = weights.get('nightWeekend', 70) if soft.avoidNightWeekend else 0
    w_continuity = weights.get('continuity', 40) if soft.shiftContinuity   else 0
    w_days_off   = weights.get('daysOff', 75)    if soft.consecutiveDaysOff else 0

    # SOFT1: Distribuție egală ─────────────────────────────────────────────────
    if w_balance > 0:
        totals = []
        for e_idx in range(n_employees):
            emp_vars = [x[e_idx, d_idx, s_idx]
                        for d_idx in range(n_dates) for s_idx in range(n_shifts)
                        if (e_idx, d_idx, s_idx) in x]
            if emp_vars: totals.append(sum(emp_vars))
        if len(totals) > 1:
            avg = model.new_int_var(0, n_dates * n_shifts, "avg_target")
            model.add(sum(totals) == avg * len(totals))
            for i, total in enumerate(totals):
                excess = model.new_int_var(0, n_dates * n_shifts, f"excess_{i}")
                model.add(excess >= total - avg)
                model.add(excess >= avg - total)
                objective.append(w_balance * excess)
            active_soft.append(f"balanceHours(w={w_balance})")
            log(f"  [SOFT1] balanceHours: w={w_balance}, {len(totals)} angajați, "
                f"{len(totals)} termeni excess")
    else:
        log(f"  [SOFT1] balanceHours: DEZACTIVAT")

    # SOFT2: Evită ture noapte vineri/sâmbătă ─────────────────────────────────
    nw_cnt = 0
    if w_night_wknd > 0 and night_indices:
        for e_idx in range(n_employees):
            for d_idx, d in enumerate(working_dates):
                if d.isoweekday() in (5, 6):
                    for s_idx in night_indices:
                        if (e_idx, d_idx, s_idx) in x:
                            objective.append(w_night_wknd * x[e_idx, d_idx, s_idx])
                            nw_cnt += 1
        active_soft.append(f"avoidNightWeekend(w={w_night_wknd})")
        log(f"  [SOFT2] avoidNightWeekend: w={w_night_wknd}, {nw_cnt} variabile, "
            f"max penalty={nw_cnt * w_night_wknd}")
    else:
        log(f"  [SOFT2] avoidNightWeekend: DEZACTIVAT")

    # SOFT3: Continuitate tură ────────────────────────────────────────────────
    cont_cnt = 0
    if w_continuity > 0 and n_shifts > 1:
        for e_idx in range(n_employees):
            for d_idx in range(1, n_dates):
                for s_idx in range(n_shifts):
                    if (e_idx, d_idx, s_idx) not in x: continue
                    for s_prev in range(n_shifts):
                        if s_prev == s_idx: continue
                        if (e_idx, d_idx - 1, s_prev) not in x: continue
                        changed = model.new_bool_var(f"chg_{e_idx}_{d_idx}_{s_idx}_{s_prev}")
                        model.add_bool_and([x[e_idx, d_idx, s_idx],
                                            x[e_idx, d_idx - 1, s_prev]]).only_enforce_if(changed)
                        model.add_bool_or([x[e_idx, d_idx, s_idx].negated(),
                                           x[e_idx, d_idx - 1, s_prev].negated(), changed])
                        objective.append(w_continuity * changed)
                        cont_cnt += 1
        active_soft.append(f"shiftContinuity(w={w_continuity})")
        log(f"  [SOFT3] shiftContinuity: w={w_continuity}, {cont_cnt} variabile")
    else:
        log(f"  [SOFT3] shiftContinuity: DEZACTIVAT")

    # SOFT4: Zile libere consecutive ──────────────────────────────────────────
    iso_cnt = 0
    if w_days_off > 0 and n_dates >= 3:
        for e_idx in range(n_employees):
            for d_idx in range(1, n_dates - 1):
                vp = [x[e_idx, d_idx-1, s] for s in range(n_shifts) if (e_idx, d_idx-1, s) in x]
                vt = [x[e_idx, d_idx,   s] for s in range(n_shifts) if (e_idx, d_idx,   s) in x]
                vn = [x[e_idx, d_idx+1, s] for s in range(n_shifts) if (e_idx, d_idx+1, s) in x]
                if not (vp and vt and vn): continue
                wp = model.new_bool_var(f"wp_{e_idx}_{d_idx}")
                wt = model.new_bool_var(f"wt_{e_idx}_{d_idx}")
                wn = model.new_bool_var(f"wn_{e_idx}_{d_idx}")
                model.add_bool_or(vp + [wp.negated()])
                model.add_bool_or(vt + [wt.negated()])
                model.add_bool_or(vn + [wn.negated()])
                for v in vp: model.add(wp >= v)
                for v in vt: model.add(wt >= v)
                for v in vn: model.add(wn >= v)
                isolated = model.new_bool_var(f"iso_{e_idx}_{d_idx}")
                model.add_bool_and([wp, wt.negated(), wn]).only_enforce_if(isolated)
                model.add_bool_or([wp.negated(), wt, wn.negated(), isolated])
                objective.append(w_days_off * isolated)
                iso_cnt += 1
        active_soft.append(f"consecutiveDaysOff(w={w_days_off})")
        log(f"  [SOFT4] consecutiveDaysOff: w={w_days_off}, {iso_cnt} variabile")
    else:
        log(f"  [SOFT4] consecutiveDaysOff: DEZACTIVAT")

    for shortage in coverage_penalties:
        objective.append(500 * shortage)

    log(f"  Soft constraints active: {active_soft if active_soft else ['none']}")
    log(f"  Total termeni objective: {len(objective)}")
    log("=== END CONSTRAINTS SUMMARY ===")

    if objective:
        model.minimize(sum(objective))
        log(f"  model.minimize: {len(objective)} termeni")
    else:
        log("  Fără objective — pure feasibility")

    proto = model.proto
    log(f"  Model: {len(proto.variables)} variabile, {len(proto.constraints)} constrângeri")

    # ── Pre-solve check — identic cu school ───────────────────────────────────
    log("=== PRE-SOLVE CHECK ===")
    total_needed = sum(effective_slots[s.id] for s in payload.shift_definitions) * n_dates
    log(f"  Asignări necesare: {total_needed} total "
        f"({n_shifts} ture × {n_dates} zile × media slots/zi)")
    for e_idx, emp in enumerate(payload.employees):
        avail = sum(1 for d_idx in range(n_dates) for s_idx in range(n_shifts)
                    if (e_idx, d_idx, s_idx) in x)
        log(f"  [{emp.name}] {avail} sloturi disponibile "
            f"({'✓' if avail >= 1 else '✗ ZERO DISPONIBILITATE'})")

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload.solver_time_limit_seconds
    solver.parameters.num_search_workers  = 4
    solver.parameters.log_search_progress = False

    log(f"Pornesc CP-SAT (timeout: {payload.solver_time_limit_seconds}s, workers: 4)...")
    status      = solver.solve(model)
    status_name = solver.status_name(status)
    solve_time  = round(solver.wall_time, 2)
    obj_val     = solver.objective_value if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None

    log(f"Status: {status_name} în {solve_time}s, objective={obj_val}")
    debug_log.append({
        "type": "solver_status", "status": status_name,
        "time_seconds": solve_time,
        "feasible": status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        "objective": obj_val,
    })

    # ── INFEASIBLE ────────────────────────────────────────────────────────────
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        log("INFEASIBLE — analiză cauze...", "warn")
        reasons = _analyze_infeasibility(
            payload, working_dates, n_dates, n_shifts, n_employees,
            effective_slots, min_rest, max_consecutive, max_weekly_h, x
        )
        for r in reasons:
            log(f"  ► {r}", "warn")
        return ShiftsResponse(
            assignments=[],
            violations=[ShiftsViolation(type="infeasible", message=r) for r in reasons],
            stats=ShiftsStats(
                total_slots=total_needed, filled_slots=0, unfilled_slots=total_needed,
                solver_status=status_name, solve_time_seconds=solve_time),
            debug_log=debug_log,
        )

    # ── Extract ───────────────────────────────────────────────────────────────
    assignments: list[AssignmentResult] = []
    filled_per: dict[tuple[int, int], int] = {}

    for e_idx, emp in enumerate(payload.employees):
        for d_idx, d in enumerate(working_dates):
            for s_idx, shift in enumerate(payload.shift_definitions):
                if (e_idx, d_idx, s_idx) not in x: continue
                if solver.value(x[e_idx, d_idx, s_idx]):
                    assignments.append(AssignmentResult(
                        employee_id=emp.id,
                        shift_definition_id=shift.id,
                        date=d.isoformat(),
                    ))
                    key = (d_idx, s_idx)
                    filled_per[key] = filled_per.get(key, 0) + 1

    # ── Post-solve validation — identic cu school ────────────────────────────
    log("=== POST-SOLVE VALIDATION ===")
    emp_day: dict[tuple[str, str], list[str]] = defaultdict(list)
    for a in assignments:
        emp_day[(a.employee_id, a.date)].append(a.shift_definition_id)
    conflicts = {k: v for k, v in emp_day.items() if len(v) > 1}

    if conflicts:
        for (eid, dt), sl in conflicts.items():
            ename = next((e.name for e in payload.employees if e.id == eid), eid[:8])
            log(f"  ANGAJAT CONFLICT {ename}: {dt}: {sl}", "error")
    else:
        log("  ✓ Niciun conflict angajat/zi detectat")

    # Per-angajat distribuție — echivalentul teacher_result din school
    emp_shift_dates: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for a in assignments:
        emp_shift_dates[a.employee_id][a.shift_definition_id].append(a.date)

    emp_counts: dict[str, int] = Counter(a.employee_id for a in assignments)

    for e_idx, emp in enumerate(payload.employees):
        cnt = emp_counts.get(emp.id, 0)
        shift_summary = []
        for shift in payload.shift_definitions:
            dates = emp_shift_dates[emp.id].get(shift.id, [])
            if dates:
                shift_summary.append(f"{shift.name}:{len(dates)}z")
        # Identic cu school: "teacher X: N lessons → [(day, period), ...]"
        log(f"  {emp.name}: {cnt} ture → {', '.join(shift_summary) if shift_summary else 'nicio tură'}")
        debug_log.append({
            "type": "employee_result", "employee_id": emp.id, "name": emp.name,
            "assignments": cnt,
            "by_shift": {s.id: len(emp_shift_dates[emp.id].get(s.id, []))
                         for s in payload.shift_definitions},
        })

    debug_log.append({"type": "distribution", "data": dict(emp_counts)})

    # ── Violations ────────────────────────────────────────────────────────────
    violations: list[ShiftsViolation] = []
    total_slots   = 0
    unfilled_slots = 0

    for d_idx, d in enumerate(working_dates):
        for s_idx, shift in enumerate(payload.shift_definitions):
            required = effective_slots[shift.id]
            total_slots += required
            filled = filled_per.get((d_idx, s_idx), 0)
            if filled < required:
                unfilled_slots += required - filled
                violations.append(ShiftsViolation(
                    type="unfilled_slot", date=d.isoformat(),
                    message=f"{d.isoformat()} / {shift.name}: {filled}/{required} angajați",
                    severity="warning",
                ))
                debug_log.append({"type": "violation",
                                  "message": f"{d.isoformat()} / {shift.name}: {filled}/{required}",
                                  "severity": "warning"})

    # ── Score breakdown — identic cu school ──────────────────────────────────
    if obj_val is not None:
        log(f"  Objective value: {obj_val:.0f}")
        by_shift: dict[str, int] = Counter(a.shift_definition_id for a in assignments)
        for shift in payload.shift_definitions:
            cnt    = by_shift.get(shift.id, 0)
            target = effective_slots[shift.id] * n_dates
            log(f"  [RESULT] {shift.name}: {cnt} asignări (target: {target}, "
                f"{'✓ OK' if cnt >= target else f'⚠ lipsesc {target - cnt}'})")

    debug_log.append({
        "type": "summary", "total_slots": total_slots,
        "filled_slots": len(assignments), "unfilled_slots": unfilled_slots,
        "violations": len(violations),
    })

    log(f"=== Done: {len(assignments)} asignări, "
        f"{unfilled_slots} neacoperite, {len(violations)} violări ===")

    return ShiftsResponse(
        assignments=assignments, violations=violations,
        stats=ShiftsStats(
            total_slots=total_slots, filled_slots=len(assignments),
            unfilled_slots=unfilled_slots, solver_status=status_name,
            solve_time_seconds=solve_time, objective_value=obj_val,
        ),
        debug_log=debug_log,
    )


# ── Infeasibility analyzer ────────────────────────────────────────────────────

def _analyze_infeasibility(payload, working_dates, n_dates, n_shifts, n_employees,
                            effective_slots, min_rest, max_consecutive, max_weekly_h, x):
    reasons = []
    total_needed = sum(effective_slots[s.id] for s in payload.shift_definitions) * n_dates
    total_avail  = len(x)

    reasons.append(
        f"[SUMAR] {n_employees} angajați · {n_shifts} ture · {n_dates} zile · "
        f"{total_needed} asignări necesare · {total_avail} variabile disponibile"
    )
    for e_idx, emp in enumerate(payload.employees):
        avail = sum(1 for d_idx in range(n_dates) for s_idx in range(n_shifts)
                    if (e_idx, d_idx, s_idx) in x)
        if avail == 0:
            reasons.append(f"[ANGAJAT] {emp.name}: ZERO sloturi — concediu/indisponibilitate acoperă toată perioada")
    if total_avail < total_needed:
        reasons.append(f"[ACOPERIRE] Max posibil={total_avail} < necesar={total_needed}. Adaugă angajați sau reduce slots/zi.")
    for i, s1 in enumerate(payload.shift_definitions):
        for j, s2 in enumerate(payload.shift_definitions):
            if i == j: continue
            rest = rest_hours_between(s1, s2)
            if 0 < rest < min_rest:
                reasons.append(f"[REPAUS] {s1.name}→{s2.name}: {rest:.1f}h < {min_rest}h minim.")
    for shift in payload.shift_definitions:
        dur = shift_duration_hours(shift)
        if dur * 5 > max_weekly_h:
            reasons.append(f"[ORE] {shift.name} ({dur:.1f}h) × 5 = {dur*5:.0f}h > max_weekly={max_weekly_h}h.")
    for e_idx, emp in enumerate(payload.employees):
        total_h = sum(shift_duration_hours(payload.shift_definitions[s_idx])
                      for d_idx in range(n_dates) for s_idx in range(n_shifts)
                      if (e_idx, d_idx, s_idx) in x)
        reasons.append(f"[ANGAJAT] {emp.name}: ore potențiale maxime = {total_h:.0f}h")
    if len(reasons) <= 1:
        reasons.append("[NECUNOSCUT] Verifică nr. angajați activi, slots/zi, concedii.")
    return reasons