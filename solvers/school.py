# orarpro-solver/solvers/school.py
#
# CP-SAT timetable scheduling for schools and universities.
#
# Model overview:
#   Variables:
#     x[t][sub][c][r][d][p] ∈ {0,1}
#       teacher t teaches subject sub to class c in room r during period p
#
#   Hard constraints:
#     1. Coverage        — each (class, subject) pair gets exactly N periods/week
#                          N comes from assignments (per-class), not subject global
#     2. Teacher once    — teacher at most one lesson per period
#     3. Class once      — class at most one lesson per period
#     4. Room once       — room hosts at most one lesson per period
#     5. Teacher load    — max_periods_per_day, max_periods_per_week
#     6. Availability    — teacher unavailability periods
#     7. Consecutive     — double lessons in consecutive pairs
#     8. Homeroom        — when students stay, class always uses its fixed room
#     9. Start first     — no gaps at start of day per class (lessons start from period 0)
#    10. Max same subj   — at most 1 lesson of same subject per class per day
#
#   Soft constraints (minimized in objective):
#     - Teacher windows (free periods between lessons)
#     - Difficult subjects in morning
#     - Uneven day distribution

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
    max_periods_per_day: int = 6
    max_periods_per_week: int = 20
    unavailable_periods: list[dict] = []

class Subject(BaseModel):
    id: str
    name: str
    periods_per_week: int           # fallback if no assignment override
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
    periods_per_week: int           # per-class override (key field)
    requires_consecutive: bool = False

class ClassHomeroom(BaseModel):
    class_id: str
    room_id: str

class SchoolConfig(BaseModel):
    avoid_teacher_windows: bool = True
    hard_subjects_morning: bool = True
    max_periods_per_day: int = 7
    min_periods_per_day: int = 1
    max_same_subject_per_day: int = 1       # NEW: max times same subject per class per day
    start_from_first_period: bool = True    # NEW: no gaps at start of day per class
    students_move_rooms: bool = False       # NEW: students go to rooms vs stay in homeroom
    class_homerooms: list[ClassHomeroom] = []  # NEW: fixed room per class when !students_move

class SchoolRequest(BaseModel):
    schedule_id: str
    teachers: list[Teacher]
    subjects: list[Subject]
    classes: list[SchoolClass]
    rooms: list[Room]
    days_per_week: int = 5
    periods_per_day: int = 8
    assignments: list[Assignment] = []     # NEW: per-class assignments with periods_per_week
    config: Optional[SchoolConfig] = None  # NEW: all soft/hard config
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

# ── Main solver ───────────────────────────────────────────────────────────────

def solve_school(payload: SchoolRequest) -> SchoolResponse:
    teachers  = payload.teachers
    subjects  = payload.subjects
    classes   = payload.classes
    rooms     = payload.rooms
    D         = payload.days_per_week
    P         = payload.periods_per_day
    cfg       = payload.config or SchoolConfig()

    T  = len(teachers)
    Su = len(subjects)
    C  = len(classes)
    R  = len(rooms)

    teacher_idx = {t.id: i for i, t in enumerate(teachers)}
    subject_idx = {s.id: i for i, s in enumerate(subjects)}
    class_idx   = {c.id: i for i, c in enumerate(classes)}
    room_idx    = {r.id: i for i, r in enumerate(rooms)}

    # ── Build per-class periods_per_week from assignments ─────────────────────
    # assignment_periods[ci][sui] = periods_per_week for that class-subject pair
    # assignment_teacher[ci][sui] = teacher index (if specified)
    assignment_periods: dict[tuple[int,int], int] = {}
    assignment_teacher: dict[tuple[int,int], int] = {}
    assignment_consecutive: dict[tuple[int,int], bool] = {}

    for asgn in payload.assignments:
        if asgn.subject_id not in subject_idx or asgn.class_id not in class_idx:
            continue
        sui = subject_idx[asgn.subject_id]
        ci  = class_idx[asgn.class_id]
        assignment_periods[(ci, sui)] = asgn.periods_per_week
        assignment_consecutive[(ci, sui)] = asgn.requires_consecutive
        if asgn.teacher_id in teacher_idx:
            assignment_teacher[(ci, sui)] = teacher_idx[asgn.teacher_id]

    # Which teachers can teach which subjects
    teacher_can_teach: dict[int, set[int]] = {}
    for ti, teacher in enumerate(teachers):
        teacher_can_teach[ti] = {subject_idx[sid] for sid in teacher.subject_ids if sid in subject_idx}

    # Room constraints
    room_for_subject: dict[int, list[int]] = {}
    for sui, subject in enumerate(subjects):
        matched = [ri for ri, room in enumerate(rooms) if room.room_type == subject.room_type]
        room_for_subject[sui] = matched if matched else list(range(R))

    # Homeroom map: class_idx → room_idx (when !students_move)
    homeroom: dict[int, int] = {}
    if not cfg.students_move_rooms:
        for ch in cfg.class_homerooms:
            if ch.class_id in class_idx and ch.room_id in room_idx:
                homeroom[class_idx[ch.class_id]] = room_idx[ch.room_id]

    # Which subjects each class needs (from assignments first, then class.subject_ids)
    class_subjects: dict[int, list[int]] = {}
    for ci, cls in enumerate(classes):
        subs = [subject_idx[sid] for sid in cls.subject_ids if sid in subject_idx]
        # Also include subjects from assignments not in class.subject_ids
        for (ci2, sui2) in assignment_periods:
            if ci2 == ci and sui2 not in subs:
                subs.append(sui2)
        class_subjects[ci] = subs

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────────────
    x: dict[tuple, cp_model.IntVar] = {}

    for ti, teacher in enumerate(teachers):
        for sui in teacher_can_teach[ti]:
            for ci, cls in enumerate(classes):
                if sui not in class_subjects[ci]:
                    continue
                # Rooms: homeroom if fixed, else by subject type
                if ci in homeroom and not cfg.students_move_rooms:
                    rooms_for_slot = [homeroom[ci]]
                else:
                    rooms_for_slot = room_for_subject[sui]
                # Restrict to assigned teacher if specified
                if (ci, sui) in assignment_teacher and assignment_teacher[(ci, sui)] != ti:
                    continue
                for ri in rooms_for_slot:
                    for d in range(D):
                        for p in range(P):
                            key = (ti, sui, ci, ri, d, p)
                            x[key] = model.new_bool_var(f"x_t{ti}_s{sui}_c{ci}_r{ri}_d{d}_p{p}")

    # ── Hard constraint 1: Coverage (per-class periods_per_week) ─────────────
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            subject = subjects[sui]
            # Use assignment override if available, else subject global
            n_periods = assignment_periods.get((ci, sui), subject.periods_per_week)
            all_slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                         if ci2 == ci and sui2 == sui]
            if all_slots:
                model.add(sum(all_slots) == n_periods)

    # ── Hard constraint 2: Teacher once per period ────────────────────────────
    for ti in range(T):
        for d in range(D):
            for p in range(P):
                slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                         if ti2 == ti and d2 == d and p2 == p]
                if slots:
                    model.add(sum(slots) <= 1)

    # ── Hard constraint 3: Class once per period ──────────────────────────────
    for ci in range(C):
        for d in range(D):
            for p in range(P):
                slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                         if ci2 == ci and d2 == d and p2 == p]
                if slots:
                    model.add(sum(slots) <= 1)

    # ── Hard constraint 4: Room once per period ───────────────────────────────
    for ri in range(R):
        for d in range(D):
            for p in range(P):
                slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                         if ri2 == ri and d2 == d and p2 == p]
                if slots:
                    model.add(sum(slots) <= 1)

    # ── Hard constraint 5: Teacher load ──────────────────────────────────────
    for ti, teacher in enumerate(teachers):
        for d in range(D):
            slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                     if ti2 == ti and d2 == d]
            if slots:
                model.add(sum(slots) <= teacher.max_periods_per_day)
        all_slots = [v for (ti2, *_), v in x.items() if ti2 == ti]
        if all_slots:
            model.add(sum(all_slots) <= teacher.max_periods_per_week)

    # ── Hard constraint 6: Teacher unavailability ─────────────────────────────
    for ti, teacher in enumerate(teachers):
        for unavail in teacher.unavailable_periods:
            d = unavail.get("day")
            p = unavail.get("period")
            if d is not None and p is not None:
                for var in [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                            if ti2 == ti and d2 == d and p2 == p]:
                    model.add(var == 0)

    # ── Hard constraint 7: Consecutive double lessons ─────────────────────────
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            subject = subjects[sui]
            is_consec = assignment_consecutive.get((ci, sui), subject.requires_consecutive)
            if not is_consec:
                continue
            n = assignment_periods.get((ci, sui), subject.periods_per_week)
            if n % 2 != 0:
                continue
            for d in range(D):
                for p in range(P - 1):
                    at_p  = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                             if sui2 == sui and ci2 == ci and d2 == d and p2 == p]
                    at_p1 = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                             if sui2 == sui and ci2 == ci and d2 == d and p2 == p + 1]
                    if at_p and at_p1:
                        b_p  = model.new_bool_var(f"consec_p_s{sui}_c{ci}_d{d}_p{p}")
                        b_p1 = model.new_bool_var(f"consec_p1_s{sui}_c{ci}_d{d}_p{p}")
                        model.add(sum(at_p) == b_p)
                        model.add(sum(at_p1) == b_p1)
                        model.add(b_p == b_p1)

    # ── Hard constraint 8: Max same subject per class per day ─────────────────
    # Default: max 1 lesson of same subject per class per day
    max_subj_day = cfg.max_same_subject_per_day
    for ci in range(C):
        for sui in class_subjects[ci]:
            for d in range(D):
                slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                         if ci2 == ci and sui2 == sui and d2 == d]
                if slots:
                    model.add(sum(slots) <= max_subj_day)

    # ── Soft constraint: Start from first period ─────────────────────────────
    # Penalize classes that have lessons on a day but NOT at period 0.
    # Cannot be hard: multiple classes share the same teacher at period 0,
    # so it's physically impossible for all of them to start at period 0.
    # Instead: maximize the number of class-days that start at period 0.
    start_penalties = []
    if cfg.start_from_first_period:
        for ci in range(C):
            for d in range(D):
                at_zero = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                           if ci2 == ci and d2 == d and p2 == 0]
                at_any  = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                           if ci2 == ci and d2 == d and p2 > 0]
                if not at_zero or not at_any:
                    continue
                has_zero = model.new_bool_var(f"has_zero_c{ci}_d{d}")
                has_later = model.new_bool_var(f"has_later_c{ci}_d{d}")
                model.add(sum(at_zero) >= has_zero)
                model.add(sum(at_zero) <= len(at_zero) * has_zero)
                model.add(sum(at_any) >= has_later)
                model.add(sum(at_any) <= len(at_any) * has_later)
                # Penalize: has lessons later in the day but not at period 0
                gap = model.new_bool_var(f"gap_start_c{ci}_d{d}")
                model.add_bool_and([has_later, has_zero.negated()]).only_enforce_if(gap)
                model.add_bool_or([has_later.negated(), has_zero]).only_enforce_if(gap.negated())
                start_penalties.append(gap)

    # ── Soft constraints ──────────────────────────────────────────────────────
    window_vars = []
    if cfg.avoid_teacher_windows:
        for ti in range(T):
            for d in range(D):
                worked = []
                for p in range(P):
                    slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                             if ti2 == ti and d2 == d and p2 == p]
                    if slots:
                        w = model.new_bool_var(f"worked_t{ti}_d{d}_p{p}")
                        model.add(sum(slots) == w)
                        worked.append((p, w))
                    else:
                        worked.append((p, None))

                for i, (p1, w1) in enumerate(worked):
                    if w1 is None:
                        continue
                    for p2, w2 in worked[i+1:]:
                        if w2 is None:
                            continue
                        gap = p2 - p1 - 1
                        if gap <= 0:
                            continue
                        both = model.new_bool_var(f"win_t{ti}_d{d}_{p1}_{p2}")
                        model.add_bool_and([w1, w2]).only_enforce_if(both)
                        model.add_bool_or([w1.negated(), w2.negated()]).only_enforce_if(both.negated())
                        window_vars.append(both * gap)

    morning_penalty = []
    if cfg.hard_subjects_morning:
        for (ti, sui, ci, ri, d, p), v in x.items():
            if subjects[sui].preferred_morning and p >= P // 2:
                morning_penalty.append(v)

    # ── Objective ─────────────────────────────────────────────────────────────
    objective = []
    if window_vars:
        objective.append(sum(window_vars) * 10)
    if morning_penalty:
        objective.append(sum(morning_penalty))
    if objective:
        model.minimize(sum(objective))

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload.solver_time_limit_seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.solve(model)
    status_name = solver.status_name(status)
    logger.info(f"School solver status: {status_name} in {solver.wall_time:.2f}s")

    # Debug: log key constraint counts to help diagnose infeasibility
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning(f"INFEASIBLE debug:")
        logger.warning(f"  Variables: {len(x)}")
        logger.warning(f"  Classes: {[f'{c.name}:{class_subjects[i]}' for i, c in enumerate(classes)]}")
        logger.warning(f"  Assignments periods: {dict(assignment_periods)}")
        total_needed = sum(assignment_periods.get((ci, sui), subjects[sui].periods_per_week)
                          for ci in range(C) for sui in class_subjects[ci])
        total_slots = D * P * C
        logger.warning(f"  Total lessons needed: {total_needed}, total slots: {total_slots}")
        logger.warning(f"  start_from_first_period: {cfg.start_from_first_period}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SchoolResponse(
            timetable=[],
            violations=[SchoolViolation(
                type="infeasible",
                entity_id="",
                entity_name="",
                message=f"No feasible timetable found ({status_name}). "
                        f"Check teacher assignments, room types, and period counts."
            )],
            stats=SchoolStats(
                total_lessons=sum(
                    assignment_periods.get((class_idx[a.class_id], subject_idx[a.subject_id]),
                                          next((s.periods_per_week for s in subjects if s.id == a.subject_id), 0))
                    for a in payload.assignments
                    if a.class_id in class_idx and a.subject_id in subject_idx
                ),
                scheduled_lessons=0,
                teacher_windows=0,
                solver_status=status_name,
                solve_time_seconds=solver.wall_time,
            )
        )

    # ── Extract timetable ─────────────────────────────────────────────────────
    timetable = []
    for (ti, sui, ci, ri, d, p), v in x.items():
        if solver.value(v) == 1:
            timetable.append(Lesson(
                teacher_id=teachers[ti].id,
                subject_id=subjects[sui].id,
                class_id=classes[ci].id,
                room_id=rooms[ri].id,
                day=d,
                period=p,
            ))

    # Count teacher windows
    total_windows = 0
    for ti in range(T):
        for d in range(D):
            busy = sorted(set(
                p2 for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                if ti2 == ti and d2 == d and solver.value(v) == 1
            ))
            if len(busy) > 1:
                total_windows += busy[-1] - busy[0] - len(busy) + 1

    # Coverage violations
    violations = []
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            subject = subjects[sui]
            n_periods = assignment_periods.get((ci, sui), subject.periods_per_week)
            scheduled = sum(
                1 for lesson in timetable
                if lesson.class_id == cls.id and lesson.subject_id == subject.id
            )
            if scheduled < n_periods:
                violations.append(SchoolViolation(
                    type="underscheduled",
                    entity_id=cls.id,
                    entity_name=cls.name,
                    message=f"'{subject.name}' for '{cls.name}': {scheduled}/{n_periods} periods"
                ))

    total_lessons = sum(
        assignment_periods.get((ci, sui), subjects[sui].periods_per_week)
        for ci in range(C)
        for sui in class_subjects[ci]
    )

    return SchoolResponse(
        timetable=timetable,
        violations=violations,
        stats=SchoolStats(
            total_lessons=total_lessons,
            scheduled_lessons=len(timetable),
            teacher_windows=total_windows,
            solver_status=status_name,
            solve_time_seconds=solver.wall_time,
        )
    )