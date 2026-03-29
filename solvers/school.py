# orarpro-solver/solvers/school.py  (v3 — simplified)
#
# Variables: x[class, subject, day, period] ∈ {0,1}
# No teachers, no rooms in the model — assigned as metadata post-solve.
#
# HARD constraints:
#   1. Coverage: exactly periods_per_week lessons per (class, subject)
#   2. Class once per slot: at most one subject per (class, day, period)
#   3. Max 1 same subject per day (or 2 if consecutive)
#   4. Consecutive pairs on same day, adjacent periods
#   5. No windows for classes: lessons compact, no gaps between first and last
#
# SOFT (objective):
#   - Start from period 0 each day (penalize gaps at start)
#   - Hard subjects in morning

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from ortools.sat.python import cp_model
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# ── Input models ──────────────────────────────────────────────────────────────

class ClassSubject(BaseModel):
    class_id: str
    subject_id: str
    periods_per_week: int
    requires_consecutive: bool = False
    preferred_morning: bool = False   # soft: schedule early

class SchoolConfig(BaseModel):
    avoid_windows: bool = True
    hard_subjects_morning: bool = True
    start_from_first_period: bool = True
    max_periods_per_day: int = 7
    min_periods_per_day: int = 1

class SchoolRequest(BaseModel):
    schedule_id: str
    class_ids: list[str]
    subject_ids: list[str]
    class_subjects: list[ClassSubject]
    days_per_week: int = 5
    periods_per_day: int = 7
    config: Optional[SchoolConfig] = None
    solver_time_limit_seconds: int = 50


# ── Output models ─────────────────────────────────────────────────────────────

class Lesson(BaseModel):
    class_id: str
    subject_id: str
    day: int
    period: int

class SchoolViolation(BaseModel):
    type: str
    message: str

class SchoolStats(BaseModel):
    total_lessons: int
    scheduled_lessons: int
    solver_status: str
    solve_time_seconds: float

class SchoolResponse(BaseModel):
    timetable: list[Lesson]
    violations: list[SchoolViolation]
    stats: SchoolStats
    debug_log: list[dict] = []


# ── Solver ────────────────────────────────────────────────────────────────────

def solve_school(payload: SchoolRequest) -> SchoolResponse:
    D   = payload.days_per_week
    P   = payload.periods_per_day
    cfg = payload.config or SchoolConfig()

    debug_log: list[dict] = []

    logger.info(f"=== School solver v3 start ===")
    logger.info(f"  Classes: {len(payload.class_ids)}, Subjects: {len(payload.subject_ids)}")
    logger.info(f"  Assignments: {len(payload.class_subjects)}, D={D} P={P}")
    logger.info(f"  Config: start_first={cfg.start_from_first_period} avoid_win={cfg.avoid_windows}")

    for cs in payload.class_subjects:
        logger.info(f"  CS: class={cs.class_id[:8]} subj={cs.subject_id[:8]} n={cs.periods_per_week} consec={cs.requires_consecutive}")
        debug_log.append({"type": "assignment",
                          "class_id": cs.class_id, "subject_id": cs.subject_id,
                          "periods_per_week": cs.periods_per_week,
                          "consecutive": cs.requires_consecutive})

    model = cp_model.CpModel()

    # x[(ci, si, d, p)] = 1 if class ci has subject si at day d period p
    x: dict[tuple, cp_model.IntVar] = {}
    cs_map = {(cs.class_id, cs.subject_id): cs for cs in payload.class_subjects}

    for cs in payload.class_subjects:
        ci = cs.class_id
        si = cs.subject_id
        for d in range(D):
            for p in range(P):
                x[(ci, si, d, p)] = model.new_bool_var(f"x_{ci[:6]}_{si[:6]}_d{d}_p{p}")

    logger.info(f"  Variables: {len(x)}")
    debug_log.append({"type": "variables", "count": len(x)})

    def slots(ci=None, si=None, d=None, p=None):
        return [v for (c2, s2, d2, p2), v in x.items()
                if (ci is None or c2 == ci) and (si is None or s2 == si)
                and (d is None or d2 == d) and (p is None or p2 == p)]

    # ── HARD 1: Coverage ─────────────────────────────────────────────────────
    for cs in payload.class_subjects:
        s = slots(ci=cs.class_id, si=cs.subject_id)
        if s:
            model.add(sum(s) == cs.periods_per_week)

    # ── HARD 2: Class once per slot ───────────────────────────────────────────
    for ci in payload.class_ids:
        for d in range(D):
            for p in range(P):
                s = slots(ci=ci, d=d, p=p)
                if s: model.add(sum(s) <= 1)

    # ── HARD 3: Max same subject per day ──────────────────────────────────────
    for cs in payload.class_subjects:
        max_per_day = 2 if cs.requires_consecutive else 1
        for d in range(D):
            s = slots(ci=cs.class_id, si=cs.subject_id, d=d)
            if s: model.add(sum(s) <= max_per_day)

    # ── HARD 4: Consecutive pairs ─────────────────────────────────────────────
    for cs in payload.class_subjects:
        if not cs.requires_consecutive:
            continue
        for d in range(D):
            for p in range(P - 1):
                at_p  = slots(ci=cs.class_id, si=cs.subject_id, d=d, p=p)
                at_p1 = slots(ci=cs.class_id, si=cs.subject_id, d=d, p=p+1)
                if at_p and at_p1:
                    bp  = model.new_bool_var(f"cp_{cs.class_id[:6]}_{cs.subject_id[:6]}_d{d}_p{p}")
                    bp1 = model.new_bool_var(f"cp_{cs.class_id[:6]}_{cs.subject_id[:6]}_d{d}_p{p}1")
                    model.add(sum(at_p)  == bp)
                    model.add(sum(at_p1) == bp1)
                    model.add(bp == bp1)

    # ── HARD 5: No windows for classes ────────────────────────────────────────
    # If class has lessons at p1 and p2 (p1<p2) on day d, all periods between must be filled
    for ci in payload.class_ids:
        for d in range(D):
            for p1 in range(P):
                for p2 in range(p1 + 2, P):
                    at_p1 = slots(ci=ci, d=d, p=p1)
                    at_p2 = slots(ci=ci, d=d, p=p2)
                    for pmid in range(p1 + 1, p2):
                        at_mid = slots(ci=ci, d=d, p=pmid)
                        if not at_p1 or not at_p2 or not at_mid:
                            continue
                        b1   = model.new_bool_var(f"nw_{ci[:6]}_d{d}_{p1}_{p2}_b1")
                        b2   = model.new_bool_var(f"nw_{ci[:6]}_d{d}_{p1}_{p2}_b2")
                        bmid = model.new_bool_var(f"nw_{ci[:6]}_d{d}_{p1}_{p2}_m{pmid}")
                        model.add(sum(at_p1)  == b1)
                        model.add(sum(at_p2)  == b2)
                        model.add(sum(at_mid) == bmid)
                        both = model.new_bool_var(f"nw_{ci[:6]}_d{d}_{p1}_{p2}_both{pmid}")
                        model.add_bool_and([b1, b2]).only_enforce_if(both)
                        model.add_bool_or([b1.negated(), b2.negated()]).only_enforce_if(both.negated())
                        model.add(bmid >= both)

    # ── SOFT: start from period 0 ─────────────────────────────────────────────
    objective = []
    if cfg.start_from_first_period:
        for ci in payload.class_ids:
            for d in range(D):
                at_zero = slots(ci=ci, d=d, p=0)
                at_rest = [v for (c2,s2,d2,p2),v in x.items() if c2==ci and d2==d and p2>0]
                if not at_zero or not at_rest:
                    continue
                has_zero  = model.new_bool_var(f"sz_{ci[:6]}_d{d}")
                has_later = model.new_bool_var(f"sl_{ci[:6]}_d{d}")
                model.add(sum(at_zero) >= has_zero)
                model.add(sum(at_zero) <= len(at_zero) * has_zero)
                model.add(sum(at_rest) >= has_later)
                model.add(sum(at_rest) <= len(at_rest) * has_later)
                gap = model.new_bool_var(f"sg_{ci[:6]}_d{d}")
                model.add_bool_and([has_later, has_zero.negated()]).only_enforce_if(gap)
                model.add_bool_or([has_later.negated(), has_zero]).only_enforce_if(gap.negated())
                objective.append(gap * 15)

    # ── SOFT: hard subjects in morning ────────────────────────────────────────
    if cfg.hard_subjects_morning:
        for cs in payload.class_subjects:
            if cs.preferred_morning:
                for (c2,s2,d2,p2), v in x.items():
                    if c2 == cs.class_id and s2 == cs.subject_id and p2 >= P // 2:
                        objective.append(v)

    if objective:
        model.minimize(sum(objective))

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload.solver_time_limit_seconds
    solver.parameters.num_search_workers  = 4
    solver.parameters.log_search_progress = False

    status      = solver.solve(model)
    status_name = solver.status_name(status)
    logger.info(f"  Status: {status_name} in {solver.wall_time:.2f}s")
    debug_log.append({"type": "solver_status", "status": status_name,
                      "time_seconds": round(solver.wall_time, 2),
                      "feasible": status in (cp_model.OPTIMAL, cp_model.FEASIBLE)})

    total = sum(cs.periods_per_week for cs in payload.class_subjects)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        reasons = _analyze_infeasibility(payload, D, P, cfg)
        for r in reasons:
            logger.warning(f"  INFEASIBLE reason: {r}")
        return SchoolResponse(
            timetable=[], debug_log=debug_log,
            violations=[SchoolViolation(type="infeasible", message=r) for r in reasons],
            stats=SchoolStats(total_lessons=total, scheduled_lessons=0,
                              solver_status=status_name, solve_time_seconds=solver.wall_time)
        )

    # ── Extract ───────────────────────────────────────────────────────────────
    timetable = [
        Lesson(class_id=c2, subject_id=s2, day=d2, period=p2)
        for (c2, s2, d2, p2), v in x.items() if solver.value(v) == 1
    ]

    # Per-class summary log
    for ci in payload.class_ids:
        cl = [l for l in timetable if l.class_id == ci]
        slot_counts = Counter((l.day, l.period) for l in cl)
        dups = {k: v for k, v in slot_counts.items() if v > 1}
        if dups:
            logger.error(f"  CLASS CONFLICT {ci[:8]}: {dups}")
        logger.info(f"  class {ci[:8]}: {len(cl)} lessons → {sorted(slot_counts.keys())}")
        debug_log.append({"type": "class_result", "class_id": ci,
                          "lessons": len(cl), "slots": sorted(slot_counts.keys()),
                          "conflict": bool(dups)})

    # Violations
    violations = []
    for cs in payload.class_subjects:
        got = sum(1 for l in timetable if l.class_id == cs.class_id and l.subject_id == cs.subject_id)
        if got < cs.periods_per_week:
            msg = f"class={cs.class_id[:8]} subj={cs.subject_id[:8]}: {got}/{cs.periods_per_week} ore"
            violations.append(SchoolViolation(type="underscheduled", message=msg))
            logger.warning(f"  UNDERSCHEDULED: {msg}")

    debug_log.append({"type": "summary", "total": total, "scheduled": len(timetable),
                      "violations": len(violations)})
    logger.info(f"=== Done: {len(timetable)}/{total} lessons, {len(violations)} violations ===")

    return SchoolResponse(timetable=timetable, violations=violations, debug_log=debug_log,
        stats=SchoolStats(total_lessons=total, scheduled_lessons=len(timetable),
                          solver_status=status_name, solve_time_seconds=solver.wall_time))


def _analyze_infeasibility(payload: SchoolRequest, D: int, P: int, cfg: SchoolConfig) -> list[str]:
    reasons = []
    for cs in payload.class_subjects:
        max_possible = D * 2 if cs.requires_consecutive else D
        if cs.periods_per_week > max_possible:
            reasons.append(
                f"Clasa {cs.class_id[:8]}, materia {cs.subject_id[:8]}: "
                f"{cs.periods_per_week} ore/săpt > {max_possible} maxim posibil"
            )
    total_per_class: dict[str, int] = {}
    for cs in payload.class_subjects:
        total_per_class[cs.class_id] = total_per_class.get(cs.class_id, 0) + cs.periods_per_week
    for ci, total in total_per_class.items():
        if total > D * P:
            reasons.append(f"Clasa {ci[:8]}: total {total} ore/săpt > {D*P} sloturi disponibile")
    if not reasons:
        reasons.append(
            "Constrângerile combinate fac orarul imposibil. "
            "Verifică numărul de ore pe săptămână per clasă."
        )
    return reasons