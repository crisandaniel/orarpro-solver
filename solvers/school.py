# orarpro-solver/solvers/school.py
#
# HARD constraints:
#   1. Class once per period (no two subjects at same time for same class)
#   2. Teacher once per period (no two classes at same time for same teacher)
#   3. Room once per period (no two classes in same room at same time)
#   4. Coverage: exactly N periods/week per (class, subject) pair
#   5. Max 1 lesson of same subject per class per day (unless requires_consecutive)
#   6. Consecutive (2h): paired lessons on same day, rest single on other days
#   7. No windows for classes: if class has lessons on day d,
#      they must be consecutive (no free periods between first and last lesson)
#
# SOFT constraints (objective):
#   - Place lessons at earliest available slot for teacher (minimize late periods)
#   - Start class day from period 0 when possible

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from ortools.sat.python import cp_model
import logging

logger = logging.getLogger(__name__)

# ── Input models ──────────────────────────────────────────────────────────────

class Teacher(BaseModel):
    id: str
    name: str
    subject_ids: list[str]
    max_periods_per_day: int = 8
    max_periods_per_week: int = 40
    unavailable_periods: list[dict] = []

class Subject(BaseModel):
    id: str
    name: str
    periods_per_week: int
    requires_consecutive: bool = False
    room_type: str = "classroom"
    preferred_morning: bool = False

class SchoolClass(BaseModel):
    id: str
    name: str
    subject_ids: list[str]

class Room(BaseModel):
    id: str
    name: str
    room_type: str = "classroom"
    capacity: int = 30

class Assignment(BaseModel):
    id: str
    teacher_id: str
    subject_id: str
    class_id: str
    group_id: Optional[str] = None
    periods_per_week: int
    requires_consecutive: bool = False

class ClassHomeroom(BaseModel):
    class_id: str
    room_id: str

class SchoolConfig(BaseModel):
    avoid_teacher_windows: bool = True
    hard_subjects_morning: bool = True
    max_periods_per_day: int = 7
    min_periods_per_day: int = 1
    students_move_rooms: bool = False
    class_homerooms: list[ClassHomeroom] = []
    start_from_first_period: bool = True   # soft: try to start from period 0

class SchoolRequest(BaseModel):
    schedule_id: str
    teachers: list[Teacher]
    subjects: list[Subject]
    classes: list[SchoolClass]
    rooms: list[Room]
    days_per_week: int = 5
    periods_per_day: int = 8
    assignments: list[Assignment] = []
    config: Optional[SchoolConfig] = None
    constraints: list[dict] = []
    solver_time_limit_seconds: int = 60

# ── Output models ─────────────────────────────────────────────────────────────

class Lesson(BaseModel):
    teacher_id: str
    subject_id: str
    class_id: str
    room_id: str
    day: int
    period: int

class SchoolViolation(BaseModel):
    type: str
    entity_id: str
    entity_name: str
    message: str

class SchoolStats(BaseModel):
    total_lessons: int
    scheduled_lessons: int
    teacher_windows: int
    solver_status: str
    solve_time_seconds: float

class SchoolResponse(BaseModel):
    timetable: list[Lesson]
    violations: list[SchoolViolation]
    stats: SchoolStats


def analyze_infeasibility(payload, class_subjects, asgn_periods, subject_idx, class_idx,
                           teacher_can_teach, asgn_teacher, homeroom, room_for_subject,
                           D, P, cfg) -> list[str]:
    """
    Identify which constraints make the model infeasible.
    Strategy: try solving with each hard constraint relaxed individually.
    Returns human-readable reasons.
    """
    teachers = payload.teachers
    subjects = payload.subjects
    classes  = payload.classes
    rooms    = payload.rooms
    reasons  = []

    # Check 1: Coverage feasibility — can periods_per_week fit in D*P slots?
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            n = asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
            max_possible = D * P  # absolute max
            # With max 1/day constraint: max = D (one per day)
            is_consec = subjects[sui].requires_consecutive
            max_with_constraint = D * 2 if is_consec else D
            if n > max_with_constraint:
                reasons.append(
                    f"Clasa '{cls.name}', materia '{subjects[sui].name}': "
                    f"{n} ore/săpt > {max_with_constraint} maxim posibil "
                    f"({'dublu → max 2/zi' if is_consec else 'max 1/zi'} × {D} zile)"
                )

    # Check 2: Teacher availability — can teacher cover all their assignments?
    for ti, t in enumerate(teachers):
        total_needed = sum(
            asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
            for ci in range(len(classes))
            for sui in class_subjects[ci]
            if (ci, sui) in asgn_teacher and asgn_teacher[(ci, sui)] == ti
        )
        if total_needed > t.max_periods_per_week:
            reasons.append(
                f"Profesorul '{t.name}': are de predat {total_needed} ore/săpt "
                f"dar limita e {t.max_periods_per_week} ore/săpt"
            )
        per_day_needed = total_needed / D if D > 0 else 0
        if per_day_needed > t.max_periods_per_day:
            reasons.append(
                f"Profesorul '{t.name}': distribuție uniformă ar necesita "
                f"{per_day_needed:.1f} ore/zi dar limita e {t.max_periods_per_day}"
            )

    # Check 3: Teacher not assigned to subject
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            has_teacher = any(
                sui in teacher_can_teach.get(ti, set())
                for ti in range(len(teachers))
                if (ci, sui) not in asgn_teacher or asgn_teacher[(ci, sui)] == ti
            )
            if not has_teacher:
                reasons.append(
                    f"Clasa '{cls.name}', materia '{subjects[sui].name}': "
                    f"niciun profesor calificat disponibil"
                )

    # Check 4: No room available for subject type
    for sui, s in enumerate(subjects):
        matching_rooms = [r for r in rooms if r.room_type == s.room_type]
        if not matching_rooms and not rooms:
            reasons.append(f"Materia '{s.name}': nicio sală definită în instituție")
        elif not matching_rooms and s.room_type != 'classroom':
            reasons.append(
                f"Materia '{s.name}' necesită sală de tip '{s.room_type}' "
                f"dar nu există astfel de sală definită (se va folosi orice sală)"
            )

    # Check 5: No-window constraint + single lesson days
    if cfg.start_from_first_period or True:  # always check
        for ci, cls in enumerate(classes):
            total = sum(asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
                       for sui in class_subjects[ci])
            if total > D * P:
                reasons.append(
                    f"Clasa '{cls.name}': total {total} ore/săpt "
                    f"depășește capacitatea {D * P} sloturi disponibile"
                )

    if not reasons:
        reasons.append(
            "Constrângerile combinate fac orarul imposibil. "
            "Încearcă: mai mulți profesori, mai multe săli, "
            "sau reducerea numărului de ore pe săptămână."
        )

    return reasons


# ── Solver ────────────────────────────────────────────────────────────────────

def solve_school(payload: SchoolRequest) -> SchoolResponse:
    teachers = payload.teachers
    subjects = payload.subjects
    classes  = payload.classes
    rooms    = payload.rooms
    D        = payload.days_per_week
    P        = payload.periods_per_day
    cfg      = payload.config or SchoolConfig()

    teacher_idx = {t.id: i for i, t in enumerate(teachers)}
    subject_idx = {s.id: i for i, s in enumerate(subjects)}
    class_idx   = {c.id: i for i, c in enumerate(classes)}
    room_idx    = {r.id: i for i, r in enumerate(rooms)}

    logger.info(f"=== School solver start ===")
    logger.info(f"  Teachers: {[t.name for t in teachers]}")
    logger.info(f"  Subjects: {[s.name for s in subjects]}")
    logger.info(f"  Classes:  {[c.name for c in classes]}")
    logger.info(f"  Rooms:    {[r.name for r in rooms]}")
    logger.info(f"  Config:   days={D} periods={P} students_move={cfg.students_move_rooms} start_first={cfg.start_from_first_period}")

    # Per-class assignment data
    asgn_periods:     dict[tuple[int,int], int]  = {}
    asgn_consecutive: dict[tuple[int,int], bool] = {}
    asgn_teacher:     dict[tuple[int,int], int]  = {}

    for a in payload.assignments:
        if a.subject_id not in subject_idx or a.class_id not in class_idx:
            logger.warning(f"  Skipping assignment: subject={a.subject_id} class={a.class_id} (not found in index)")
            continue
        sui = subject_idx[a.subject_id]
        ci  = class_idx[a.class_id]
        asgn_periods[(ci, sui)]     = a.periods_per_week
        asgn_consecutive[(ci, sui)] = a.requires_consecutive
        if a.teacher_id in teacher_idx:
            asgn_teacher[(ci, sui)] = teacher_idx[a.teacher_id]
        tname = next((t.name for t in teachers if t.id == a.teacher_id), a.teacher_id)
        cname = classes[ci].name
        sname = subjects[sui].name
        logger.info(f"  Assignment: {tname} → {sname} → {cname} × {a.periods_per_week}h/w consec={a.requires_consecutive}")

    # Teacher → subjects
    teacher_can_teach: dict[int, set[int]] = {
        ti: {subject_idx[sid] for sid in t.subject_ids if sid in subject_idx}
        for ti, t in enumerate(teachers)
    }

    # Room assignment
    homeroom: dict[int, int] = {}
    if not cfg.students_move_rooms:
        for ch in cfg.class_homerooms:
            if ch.class_id in class_idx and ch.room_id in room_idx:
                homeroom[class_idx[ch.class_id]] = room_idx[ch.room_id]

    room_for_subject: dict[int, list[int]] = {}
    for sui, s in enumerate(subjects):
        matched = [ri for ri, r in enumerate(rooms) if r.room_type == s.room_type]
        room_for_subject[sui] = matched or list(range(len(rooms)))

    # Class → subjects
    class_subjects: dict[int, list[int]] = {}
    for ci, cls in enumerate(classes):
        subs = [subject_idx[sid] for sid in cls.subject_ids if sid in subject_idx]
        for (ci2, sui2) in asgn_periods:
            if ci2 == ci and sui2 not in subs:
                subs.append(sui2)
        class_subjects[ci] = subs

    model = cp_model.CpModel()

    # ── Variables ─────────────────────────────────────────────────────────────
    x: dict[tuple, cp_model.IntVar] = {}
    for ti in range(len(teachers)):
        for sui in teacher_can_teach[ti]:
            for ci in range(len(classes)):
                if sui not in class_subjects[ci]:
                    continue
                if (ci, sui) in asgn_teacher and asgn_teacher[(ci, sui)] != ti:
                    continue
                if ci in homeroom and not cfg.students_move_rooms:
                    room_list = [homeroom[ci]]
                else:
                    room_list = room_for_subject[sui]
                for ri in room_list:
                    for d in range(D):
                        for p in range(P):
                            x[(ti, sui, ci, ri, d, p)] = model.new_bool_var(
                                f"x_t{ti}_s{sui}_c{ci}_r{ri}_d{d}_p{p}"
                            )

    logger.info(f"  Variables created: {len(x)}")
    if len(x) == 0:
        logger.error("  ZERO variables! Check teacher subject_ids match assignment subject IDs")

    def slots(*, ti=None, sui=None, ci=None, ri=None, d=None, p=None):
        return [v for (t2,s2,c2,r2,d2,p2), v in x.items()
                if (ti is None or t2==ti) and (sui is None or s2==sui)
                and (ci is None or c2==ci) and (ri is None or r2==ri)
                and (d is None or d2==d) and (p is None or p2==p)]

    # ── HARD 1: Coverage (exact periods/week per class-subject) ──────────────
    for ci in range(len(classes)):
        for sui in class_subjects[ci]:
            n = asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
            s = slots(ci=ci, sui=sui)
            if s:
                model.add(sum(s) == n)

    # ── HARD 2: Teacher once per period ──────────────────────────────────────
    for ti in range(len(teachers)):
        for d in range(D):
            for p in range(P):
                s = slots(ti=ti, d=d, p=p)
                if s: model.add(sum(s) <= 1)

    # ── HARD 3: Class once per period ────────────────────────────────────────
    for ci in range(len(classes)):
        for d in range(D):
            for p in range(P):
                s = slots(ci=ci, d=d, p=p)
                if s: model.add(sum(s) <= 1)

    # ── HARD 4: Room once per period ─────────────────────────────────────────
    for ri in range(len(rooms)):
        for d in range(D):
            for p in range(P):
                s = slots(ri=ri, d=d, p=p)
                if s: model.add(sum(s) <= 1)

    # ── HARD 5: Teacher load ─────────────────────────────────────────────────
    for ti, t in enumerate(teachers):
        for d in range(D):
            s = slots(ti=ti, d=d)
            if s: model.add(sum(s) <= t.max_periods_per_day)
        s = slots(ti=ti)
        if s: model.add(sum(s) <= t.max_periods_per_week)

    # ── HARD 6: Max 1 same subject per class per day (unless consecutive) ────
    for ci in range(len(classes)):
        for sui in class_subjects[ci]:
            is_consec = asgn_consecutive.get((ci, sui), subjects[sui].requires_consecutive)
            max_per_day = 2 if is_consec else 1
            for d in range(D):
                s = slots(ci=ci, sui=sui, d=d)
                if s: model.add(sum(s) <= max_per_day)

    # ── HARD 7: Consecutive lessons must be paired (same day, adjacent) ──────
    for ci in range(len(classes)):
        for sui in class_subjects[ci]:
            if not asgn_consecutive.get((ci, sui), subjects[sui].requires_consecutive):
                continue
            n = asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
            # On days with 2 lessons, they must be at periods p and p+1
            for d in range(D):
                for p in range(P - 1):
                    at_p  = slots(ci=ci, sui=sui, d=d, p=p)
                    at_p1 = slots(ci=ci, sui=sui, d=d, p=p+1)
                    if at_p and at_p1:
                        bp  = model.new_bool_var(f"cp_c{ci}_s{sui}_d{d}_p{p}")
                        bp1 = model.new_bool_var(f"cp_c{ci}_s{sui}_d{d}_p{p}1")
                        model.add(sum(at_p)  == bp)
                        model.add(sum(at_p1) == bp1)
                        model.add(bp == bp1)

    # ── HARD 8: No windows for classes ───────────────────────────────────────
    # If a class has lessons at periods p1 and p2 (p1 < p2) on day d,
    # all periods between p1 and p2 must also be occupied by the class.
    for ci in range(len(classes)):
        for d in range(D):
            for p in range(P):
                for p2 in range(p + 2, P):
                    at_p  = slots(ci=ci, d=d, p=p)
                    at_p2 = slots(ci=ci, d=d, p=p2)
                    # For each period between p and p2, class must have a lesson
                    for pmid in range(p + 1, p2):
                        at_mid = slots(ci=ci, d=d, p=pmid)
                        if not at_p or not at_p2 or not at_mid:
                            continue
                        bp   = model.new_bool_var(f"nw_c{ci}_d{d}_p{p}_{p2}_has_p")
                        bp2  = model.new_bool_var(f"nw_c{ci}_d{d}_p{p}_{p2}_has_p2")
                        bmid = model.new_bool_var(f"nw_c{ci}_d{d}_p{p}_{p2}_mid{pmid}")
                        model.add(sum(at_p)   == bp)
                        model.add(sum(at_p2)  == bp2)
                        model.add(sum(at_mid) == bmid)
                        # bp AND bp2 → bmid
                        both = model.new_bool_var(f"nw_both_c{ci}_d{d}_{p}_{p2}_{pmid}")
                        model.add_bool_and([bp, bp2]).only_enforce_if(both)
                        model.add_bool_or([bp.negated(), bp2.negated()]).only_enforce_if(both.negated())
                        model.add(bmid >= both)

    # ── Teacher unavailability ────────────────────────────────────────────────
    for ti, t in enumerate(teachers):
        for u in t.unavailable_periods:
            d, p = u.get("day"), u.get("period")
            if d is not None and p is not None:
                for v in slots(ti=ti, d=d, p=p):
                    model.add(v == 0)

    # ── SOFT: minimize teacher windows ───────────────────────────────────────
    objective = []

    if cfg.avoid_teacher_windows:
        for ti in range(len(teachers)):
            for d in range(D):
                worked = []
                for p in range(P):
                    s = slots(ti=ti, d=d, p=p)
                    if s:
                        w = model.new_bool_var(f"tw_t{ti}_d{d}_p{p}")
                        model.add(sum(s) == w)
                        worked.append((p, w))
                for i, (p1, w1) in enumerate(worked):
                    for p2, w2 in worked[i+1:]:
                        gap = p2 - p1 - 1
                        if gap <= 0: continue
                        both = model.new_bool_var(f"tboth_t{ti}_d{d}_{p1}_{p2}")
                        model.add_bool_and([w1, w2]).only_enforce_if(both)
                        model.add_bool_or([w1.negated(), w2.negated()]).only_enforce_if(both.negated())
                        objective.append(both * gap * 10)

    # ── SOFT: start class day from period 0 ──────────────────────────────────
    # Penalize class-days where first lesson is not at period 0
    if cfg.start_from_first_period:
        for ci in range(len(classes)):
            for d in range(D):
                at_zero = slots(ci=ci, d=d, p=0)
                at_rest = slots(ci=ci, d=d)
                at_rest_nonzero = [v for (t2,s2,c2,r2,d2,p2), v in x.items()
                                   if c2==ci and d2==d and p2>0]
                if not at_zero or not at_rest_nonzero:
                    continue
                has_zero  = model.new_bool_var(f"sz_c{ci}_d{d}")
                has_later = model.new_bool_var(f"sl_c{ci}_d{d}")
                model.add(sum(at_zero) >= has_zero)
                model.add(sum(at_zero) <= len(at_zero) * has_zero)
                model.add(sum(at_rest_nonzero) >= has_later)
                model.add(sum(at_rest_nonzero) <= len(at_rest_nonzero) * has_later)
                gap = model.new_bool_var(f"sg_c{ci}_d{d}")
                model.add_bool_and([has_later, has_zero.negated()]).only_enforce_if(gap)
                model.add_bool_or([has_later.negated(), has_zero]).only_enforce_if(gap.negated())
                objective.append(gap * 15)

    # ── SOFT: hard subjects in morning ───────────────────────────────────────
    if cfg.hard_subjects_morning:
        for (ti, sui, ci, ri, d, p), v in x.items():
            if subjects[sui].preferred_morning and p >= P // 2:
                objective.append(v)

    if objective:
        model.minimize(sum(objective))

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload.solver_time_limit_seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.solve(model)
    status_name = solver.status_name(status)
    logger.info(f"School solver: {status_name} in {solver.wall_time:.2f}s")
    logger.info(f"  Objective value: {solver.objective_value if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 'N/A'}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        total = sum(asgn_periods.get((class_idx.get(a.class_id,-1),
                    subject_idx.get(a.subject_id,-1)), 0)
                    for a in payload.assignments)
        reasons = analyze_infeasibility(
            payload, class_subjects, asgn_periods, subject_idx, class_idx,
            teacher_can_teach, asgn_teacher, homeroom, room_for_subject,
            D, P, cfg
        )
        logger.warning(f"=== INFEASIBLE ({status_name}) in {solver.wall_time:.2f}s ===")
        logger.warning(f"  Variables: {len(x)}, total lessons needed: {total}")
        logger.warning(f"  Config: start_first={cfg.start_from_first_period} no_windows={True} max_day={cfg.max_periods_per_day}")
        for i, r in enumerate(reasons):
            logger.warning(f"  Reason {i+1}: {r}")
        # Log per-teacher load
        for t in payload.teachers:
            t_needed = sum(
                asgn_periods.get((class_idx.get(a.class_id,-1), subject_idx.get(a.subject_id,-1)), 0)
                for a in payload.assignments if a.teacher_id == t.id
            )
            logger.warning(f"  Teacher '{t.name}': needs {t_needed}h/w, limit={t.max_periods_per_week}h/w ({t.max_periods_per_day}h/day)")
        violations = [
            SchoolViolation(type="infeasible", entity_id="", entity_name="", message=r)
            for r in reasons
        ]
        return SchoolResponse(
            timetable=[],
            violations=violations,
            stats=SchoolStats(total_lessons=total, scheduled_lessons=0,
                teacher_windows=0, solver_status=status_name,
                solve_time_seconds=solver.wall_time)
        )

    # ── Extract ───────────────────────────────────────────────────────────────
    timetable = [
        Lesson(teacher_id=teachers[ti].id, subject_id=subjects[sui].id,
               class_id=classes[ci].id, room_id=rooms[ri].id, day=d, period=p)
        for (ti, sui, ci, ri, d, p), v in x.items() if solver.value(v) == 1
    ]

    logger.info(f"  Lessons generated: {len(timetable)}")
    # Summary per teacher
    for ti, t in enumerate(teachers):
        t_lessons = [l for l in timetable if l.teacher_id == t.id]
        logger.info(f"  {t.name}: {len(t_lessons)} lessons → {sorted(set((l.day, l.period) for l in t_lessons))}")

    # Post-solve validation — detect teacher conflicts
    teacher_slots: dict[tuple, list] = {}
    for lesson in timetable:
        slot = (lesson.teacher_id, lesson.day, lesson.period)
        teacher_slots.setdefault(slot, []).append(lesson.class_id)
    conflicts = {k: v for k, v in teacher_slots.items() if len(v) > 1}
    if conflicts:
        logger.error(f"TEACHER CONFLICT in solution! {len(conflicts)} slots with multiple classes:")
        for (tid, d, p), classes_list in list(conflicts.items())[:3]:
            tname = next((t.name for t in teachers if t.id == tid), tid)
            logger.error(f"  {tname} day={d} period={p}: {classes_list}")

    total_windows = 0
    for ti in range(len(teachers)):
        for d in range(D):
            busy = sorted(set(p2 for (t2,s2,c2,r2,d2,p2), v in x.items()
                              if t2==ti and d2==d and solver.value(v)==1))
            if len(busy) > 1:
                total_windows += busy[-1] - busy[0] - len(busy) + 1

    violations = []
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            n = asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
            got = sum(1 for l in timetable if l.class_id==cls.id and l.subject_id==subjects[sui].id)
            if got < n:
                violations.append(SchoolViolation(type="underscheduled",
                    entity_id=cls.id, entity_name=cls.name,
                    message=f"'{subjects[sui].name}' for '{cls.name}': {got}/{n}"))
                logger.warning(f"  UNDERSCHEDULED: {subjects[sui].name} for {cls.name}: {got}/{n}")

    total = sum(asgn_periods.get((ci, sui), subjects[sui].periods_per_week)
                for ci in range(len(classes)) for sui in class_subjects[ci])

    logger.info(f"=== School solver done: {len(timetable)}/{total} lessons, {len(violations)} violations ===")
    return SchoolResponse(timetable=timetable, violations=violations,
        stats=SchoolStats(total_lessons=total, scheduled_lessons=len(timetable),
            teacher_windows=total_windows, solver_status=status_name,
            solve_time_seconds=solver.wall_time))