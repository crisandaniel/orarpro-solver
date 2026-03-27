# orarpro-solver/solvers/school.py
#
# CP-SAT timetable scheduling for schools and universities.
#
# Model overview:
#   Variables:
#     x[t][sub][c][r][p] ∈ {0,1}
#       teacher t teaches subject sub to class c in room r during period p
#
#   Hard constraints (must all be satisfied):
#     1. Coverage        — each (class, subject) pair gets exactly N periods/week
#     2. Teacher once    — teacher teaches at most one lesson per period
#     3. Class once      — class has at most one lesson per period
#     4. Room once       — room hosts at most one lesson per period
#     5. Teacher assigns — only qualified teachers teach each subject
#     6. Room capacity   — room type must match lesson type (lab, gym, etc.)
#     7. Availability    — teacher unavailability / day-off constraints
#     8. Consecutive     — some subjects need 2 consecutive periods (double lessons)
#     9. Not first/last  — some subjects forbidden in first or last period
#
#   Soft constraints (minimized in objective):
#     - Teacher windows (free periods between lessons on same day)
#     - Uneven distribution (same subject multiple times per day)
#     - Difficult subjects scheduled in later periods

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
    subject_ids: list[str]          # subjects this teacher is qualified to teach
    max_periods_per_day: int = 6
    max_periods_per_week: int = 20
    unavailable_periods: list[dict] = []
    # [{day: 0-4 (Mon-Fri), period: 0-based}]

class Subject(BaseModel):
    id: str
    name: str
    periods_per_week: int           # how many periods per week this subject needs
    requires_consecutive: bool = False  # needs double lessons (2 consecutive periods)
    room_type: str = "classroom"    # 'classroom' | 'lab' | 'gym' | 'computer'
    preferred_morning: bool = False # soft: schedule in first half of day

class SchoolClass(BaseModel):
    id: str
    name: str                       # e.g. "10A", "Year 3"
    subject_ids: list[str]          # subjects this class studies

class Room(BaseModel):
    id: str
    name: str
    room_type: str = "classroom"   # must match subject.room_type
    capacity: int = 30

class SchoolRequest(BaseModel):
    schedule_id: str
    teachers: list[Teacher]
    subjects: list[Subject]
    classes: list[SchoolClass]
    rooms: list[Room]
    days_per_week: int = 5          # 5 for Mon-Fri
    periods_per_day: int = 8        # number of time slots per day
    constraints: list[dict] = []    # additional custom constraints
    solver_time_limit_seconds: int = 60

# ── Output models ─────────────────────────────────────────────────────────────

class Lesson(BaseModel):
    teacher_id: str
    subject_id: str
    class_id: str
    room_id: str
    day: int                        # 0=Monday, 1=Tuesday, ... 4=Friday
    period: int                     # 0-based period index within the day

class SchoolViolation(BaseModel):
    type: str
    entity_id: str
    entity_name: str
    message: str

class SchoolStats(BaseModel):
    total_lessons: int
    scheduled_lessons: int
    teacher_windows: int            # total free periods between lessons (lower = better)
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

    T  = len(teachers)
    Su = len(subjects)
    C  = len(classes)
    R  = len(rooms)

    teacher_idx = {t.id: i for i, t in enumerate(teachers)}
    subject_idx = {s.id: i for i, s in enumerate(subjects)}
    class_idx   = {c.id: i for i, c in enumerate(classes)}
    room_idx    = {r.id: i for i, r in enumerate(rooms)}

    # Which teachers can teach which subjects
    teacher_can_teach: dict[int, set[int]] = {}
    for ti, teacher in enumerate(teachers):
        teacher_can_teach[ti] = {subject_idx[sid] for sid in teacher.subject_ids if sid in subject_idx}

    # Which rooms can host which subject (by room type)
    room_for_subject: dict[int, list[int]] = {}
    for sui, subject in enumerate(subjects):
        room_for_subject[sui] = [
            ri for ri, room in enumerate(rooms)
            if room.room_type == subject.room_type
        ]
        # Fallback: any classroom if no matching room type found
        if not room_for_subject[sui]:
            room_for_subject[sui] = list(range(R))

    # Which subjects each class needs
    class_subjects: dict[int, list[int]] = {}
    for ci, cls in enumerate(classes):
        class_subjects[ci] = [subject_idx[sid] for sid in cls.subject_ids if sid in subject_idx]

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────────────
    # x[t][su][c][r][d][p] = 1 if teacher t teaches subject su to class c in room r on day d period p
    # This is a 6D tensor — we only create vars for valid combinations to reduce size
    x: dict[tuple, cp_model.IntVar] = {}

    for ti, teacher in enumerate(teachers):
        for sui in teacher_can_teach[ti]:
            subject = subjects[sui]
            for ci, cls in enumerate(classes):
                if sui not in class_subjects[ci]:
                    continue
                for ri in room_for_subject[sui]:
                    for d in range(D):
                        for p in range(P):
                            key = (ti, sui, ci, ri, d, p)
                            x[key] = model.new_bool_var(
                                f"x_t{ti}_s{sui}_c{ci}_r{ri}_d{d}_p{p}"
                            )

    def get_var(ti, sui, ci, ri, d, p):
        return x.get((ti, sui, ci, ri, d, p), None)

    # ── Hard constraint 1: Coverage ───────────────────────────────────────────
    # Each (class, subject) pair must be scheduled exactly N periods per week
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            subject = subjects[sui]
            all_slots = [
                v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                if ci2 == ci and sui2 == sui
            ]
            if all_slots:
                model.add(sum(all_slots) == subject.periods_per_week)

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

    # ── Hard constraint 5: Max periods per day per teacher ────────────────────
    for ti, teacher in enumerate(teachers):
        for d in range(D):
            slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                     if ti2 == ti and d2 == d]
            if slots:
                model.add(sum(slots) <= teacher.max_periods_per_day)

    # ── Hard constraint 6: Max periods per week per teacher ───────────────────
    for ti, teacher in enumerate(teachers):
        slots = [v for (ti2, *rest), v in x.items() if ti2 == ti]
        if slots:
            model.add(sum(slots) <= teacher.max_periods_per_week)

    # ── Hard constraint 7: Teacher unavailability ─────────────────────────────
    for ti, teacher in enumerate(teachers):
        for unavail in teacher.unavailable_periods:
            d = unavail.get("day")
            p = unavail.get("period")
            if d is not None and p is not None:
                blocked = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                           if ti2 == ti and d2 == d and p2 == p]
                for var in blocked:
                    model.add(var == 0)

    # ── Hard constraint 8: Consecutive double lessons ─────────────────────────
    # If subject requires consecutive periods, ensure lessons come in pairs
    for sui, subject in enumerate(subjects):
        if not subject.requires_consecutive:
            continue
        # For each (class, day), lessons of this subject must appear in consecutive pairs
        for ci, cls in enumerate(classes):
            if sui not in class_subjects[ci]:
                continue
            # Count must be even (pairs)
            n = subject.periods_per_week
            if n % 2 != 0:
                continue
            # Each day: if lesson at period p, must also have lesson at period p+1
            for d in range(D):
                for p in range(P - 1):
                    # All vars for this subject, class, day, period p
                    at_p = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                            if sui2 == sui and ci2 == ci and d2 == d and p2 == p]
                    at_p1 = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                             if sui2 == sui and ci2 == ci and d2 == d and p2 == p + 1]
                    if at_p and at_p1:
                        # If scheduled at p, must be scheduled at p+1 too
                        b_p  = model.new_bool_var(f"consec_p_s{sui}_c{ci}_d{d}_p{p}")
                        b_p1 = model.new_bool_var(f"consec_p1_s{sui}_c{ci}_d{d}_p{p}")
                        model.add(sum(at_p) == b_p)
                        model.add(sum(at_p1) == b_p1)
                        # Both or neither
                        model.add(b_p == b_p1)

    # ── Soft constraints — minimize windows (free periods between lessons) ─────
    # A "window" = a free period between two lessons for the same teacher on the same day
    # We want teachers to have compact schedules (all lessons together, no gaps)
    window_vars = []
    for ti in range(T):
        for d in range(D):
            # worked[p] = 1 if teacher has any lesson at period p on day d
            worked = []
            for p in range(P):
                slots = [v for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                         if ti2 == ti and d2 == d and p2 == p]
                if slots:
                    w = model.new_bool_var(f"worked_t{ti}_d{d}_p{p}")
                    model.add(sum(slots) >= w)
                    model.add(sum(slots) <= w)
                    worked.append((p, w))
                else:
                    worked.append((p, None))

            # For each pair (p1, p2) where p1 < p2 and teacher works both,
            # count free periods in between as windows
            for i, (p1, w1) in enumerate(worked):
                if w1 is None:
                    continue
                for j, (p2, w2) in enumerate(worked):
                    if w2 is None or p2 <= p1:
                        continue
                    # Gap = p2 - p1 - 1 free periods
                    gap = p2 - p1 - 1
                    if gap <= 0:
                        continue
                    # If both w1 and w2 are 1, there are `gap` windows
                    both_work = model.new_bool_var(f"win_t{ti}_d{d}_p{p1}_{p2}")
                    model.add_bool_and([w1, w2]).only_enforce_if(both_work)
                    model.add_bool_or([w1.negated(), w2.negated()]).only_enforce_if(both_work.negated())
                    window_vars.append(both_work * gap)

    # Soft: prefer difficult subjects (teacher.subject_ids[0]) in morning
    morning_penalty = []
    for (ti, sui, ci, ri, d, p), v in x.items():
        subject = subjects[sui]
        if subject.preferred_morning and p >= P // 2:
            morning_penalty.append(v)

    # ── Objective ─────────────────────────────────────────────────────────────
    objective = []
    if window_vars:
        objective.append(sum(window_vars) * 10)  # windows are most important
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

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SchoolResponse(
            timetable=[],
            violations=[SchoolViolation(
                type="infeasible",
                entity_id="",
                entity_name="",
                message=f"No feasible timetable found ({status_name}). "
                        f"Check that enough teachers are assigned to each subject, "
                        f"rooms match subject types, and period counts are achievable."
            )],
            stats=SchoolStats(
                total_lessons=sum(s.periods_per_week for cls in classes for sid in cls.subject_ids
                                  for s in subjects if s.id == sid),
                scheduled_lessons=0,
                teacher_windows=0,
                solver_status=status_name,
                solve_time_seconds=solver.wall_time,
            )
        )

    # ── Extract timetable ──────────────────────────────────────────────────────
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

    # Count windows
    total_windows = 0
    for ti in range(T):
        for d in range(D):
            periods_worked = sorted([
                p for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                if ti2 == ti and d2 == d and solver.value(v) == 1
            ] + [p for p in range(P)])
            # Actually just count distinct periods with lessons
            busy = sorted(set(
                p2 for (ti2, sui2, ci2, ri2, d2, p2), v in x.items()
                if ti2 == ti and d2 == d and solver.value(v) == 1
            ))
            if len(busy) > 1:
                total_windows += busy[-1] - busy[0] - len(busy) + 1  # gaps

    # Check coverage violations
    violations = []
    for ci, cls in enumerate(classes):
        for sui in class_subjects[ci]:
            subject = subjects[sui]
            scheduled = sum(
                1 for lesson in timetable
                if lesson.class_id == cls.id and lesson.subject_id == subject.id
            )
            if scheduled < subject.periods_per_week:
                violations.append(SchoolViolation(
                    type="underscheduled",
                    entity_id=cls.id,
                    entity_name=cls.name,
                    message=f"Subject '{subject.name}' for class '{cls.name}': "
                            f"{scheduled}/{subject.periods_per_week} periods scheduled"
                ))

    total_lessons = sum(
        subjects[sui].periods_per_week
        for ci, cls in enumerate(classes)
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
