# orarpro-solver/solvers/school.py  (v4)
#
# Input model (v4 — Lesson-based, allowedSlots pre-calculated):
#   Lessons:  atomic units with allowed_slots, duration, teacher/class/subject
#   Teachers: limits (max/min per day/week) + preferred slots (soft)
#   Classes:  max_lessons_per_day, stage
#   Rooms:    type (for preferred_room matching)
#   SoftRules: weights for objective
#
# Variables:
#   x[lesson_id][slot] ∈ {0,1}  — is lesson placed in this slot?
#   For duration=2: x[lesson_id][slot] means it occupies slot AND slot+1
#
# HARD constraints:
#   1. Each lesson placed exactly once
#   2. Lesson only in allowed_slots
#   3. Teacher once per slot
#   4. Class once per slot
#   5. Room once per slot (if room assigned)
#   6. Duration=2: consecutive slots in same day
#   7. Teacher max/min per day/week
#   8. Class max per day
#
# SOFT (objective, weighted):
#   - Teacher gaps (windows between lessons)
#   - Last slot for young classes (primary/middle)
#   - Same subject twice per day per class
#   - Hard subjects in morning
#   - Start from first slot (no gaps at day start for class)

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from ortools.sat.python import cp_model
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


# ── Input models ──────────────────────────────────────────────────────────────

class Lesson(BaseModel):
    id: str
    class_id: str
    subject_id: str
    teacher_id: str
    duration: int = 1                   # 1 or 2 (double period)
    allowed_slots: list[str]            # "day-period" strings
    preferred_room_id: Optional[str] = None

class TeacherConfig(BaseModel):
    id: str
    name: str = ''
    max_lessons_per_day:  Optional[int] = None
    max_lessons_per_week: Optional[int] = None
    min_lessons_per_week: Optional[int] = None
    preferred_slots: list[str] = []

class ClassConfig(BaseModel):
    id: str
    name: str = ''
    name: str
    stage: str = 'high'               # primary/middle/high/university
    max_lessons_per_day: int = 8

class RoomConfig(BaseModel):
    id: str
    name: str
    type: str = 'generic'

class SoftRules(BaseModel):
    avoidGapsForTeachers:        bool = True
    avoidLastHourForStages:      list[str] = ['primary', 'middle']
    avoidSameSubjectTwicePerDay: bool = True
    hardSubjectsMorning:         bool = False  # needs subject difficulty info
    startFromFirstSlot:          bool = True
    weights: dict = {
        'teacherGaps':  80,
        'lastHour':     60,
        'sameSubject':  70,
        'hardMorning':  50,
        'startFirst':   90,
    }

class SchoolRequest(BaseModel):
    schedule_id: str
    lessons: list[Lesson]
    teachers: list[TeacherConfig] = []
    classes: list[ClassConfig] = []
    rooms: list[RoomConfig] = []
    days_per_week: int = 5
    slots_per_day: int = 8
    soft_rules: SoftRules = SoftRules()
    solver_time_limit_seconds: int = 50


# ── Output models ─────────────────────────────────────────────────────────────

class PlacedLesson(BaseModel):
    lesson_id: str
    class_id: str
    subject_id: str
    teacher_id: str
    room_id: Optional[str] = None
    day: int
    period: int
    duration: int = 1

class SchoolViolation(BaseModel):
    type: str
    message: str

class SchoolStats(BaseModel):
    total_lessons: int
    scheduled_lessons: int
    solver_status: str
    solve_time_seconds: float
    objective_value: Optional[float] = None

class SchoolResponse(BaseModel):
    timetable: list[PlacedLesson]
    violations: list[SchoolViolation]
    stats: SchoolStats
    debug_log: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_slot(slot: str) -> tuple[int, int]:
    """Parse "day-period" string to (day, period) ints."""
    parts = slot.split('-')
    return int(parts[0]), int(parts[1])


# ── Solver ────────────────────────────────────────────────────────────────────

def solve_school(payload: SchoolRequest) -> SchoolResponse:
    D   = payload.days_per_week
    P   = payload.slots_per_day
    cfg = payload.soft_rules
    debug_log: list[dict] = []

    logger.info(f"=== School solver v4 ===")
    logger.info(f"  {len(payload.lessons)} lessons, {D}d × {P}p = {D*P} slots")
    logger.info(f"  teachers: {len(payload.teachers)}, classes: {len(payload.classes)}")
    # Log class configs
    for c in payload.classes:
        logger.info(f"  CLASS {c.name}: max_per_day={c.max_lessons_per_day} stage={c.stage}")
    # Log teacher configs
    for t in payload.teachers:
        logger.info(f"  TEACHER {t.name}: max_pd={t.max_lessons_per_day} max_pw={t.max_lessons_per_week}")

    # Index configs
    teacher_cfg = {t.id: t for t in payload.teachers}
    class_cfg   = {c.id: c for c in payload.classes}

    # Log assignments
    for l in payload.lessons:
        logger.info(f"  Lesson {l.id}: class={l.class_id[:8]} subj={l.subject_id[:8]} "
                    f"teacher={l.teacher_id[:8]} dur={l.duration} allowed={len(l.allowed_slots)} slots")
        debug_log.append({
            "type":       "assignment",
            "lesson_id":  l.id,
            "class_id":   l.class_id,
            "subject_id": l.subject_id,
            "teacher_id": l.teacher_id,
            "duration":   l.duration,
            "allowed":    len(l.allowed_slots),
            "weekly_hours": 1,  # each lesson = 1 unit
        })

    model = cp_model.CpModel()

    # ── Variables ─────────────────────────────────────────────────────────────
    # x[lesson_id][slot_str] = BoolVar
    # Only create vars for allowed slots
    x: dict[str, dict[str, cp_model.IntVar]] = {}
    for lesson in payload.lessons:
        x[lesson.id] = {}
        valid_slots = lesson.allowed_slots
        # For duration=2, filter out slots where next slot (same day) doesn't exist
        if lesson.duration == 2:
            valid_slots = [
                s for s in valid_slots
                if (d := parse_slot(s))[1] < P - 1  # not last slot of day
                and f"{d[0]}-{d[1]+1}" in lesson.allowed_slots  # next slot also allowed
            ]
        for slot in valid_slots:
            x[lesson.id][slot] = model.new_bool_var(f"x_{lesson.id[:8]}_{slot}")

    total_vars = sum(len(slots) for slots in x.values())
    logger.info(f"  Variables: {total_vars}")
    debug_log.append({"type": "variables", "count": total_vars})

    # ── HARD 1: Each lesson placed exactly once ───────────────────────────────
    for lesson in payload.lessons:
        slot_vars = list(x[lesson.id].values())
        if not slot_vars:
            logger.warning(f"  Lesson {lesson.id}: NO valid slots!")
        model.add(sum(slot_vars) == 1)

    # ── HARD 2: Teacher once per slot ─────────────────────────────────────────
    # Group lessons by teacher
    teacher_lessons: dict[str, list[Lesson]] = defaultdict(list)
    for lesson in payload.lessons:
        teacher_lessons[lesson.teacher_id].append(lesson)

    for teacher_id, lessons in teacher_lessons.items():
        for d in range(D):
            for p in range(P):
                vars_at_slot = []
                for l in lessons:
                    slot = f"{d}-{p}"
                    # Lesson starts at this slot
                    if slot in x[l.id]:
                        vars_at_slot.append(x[l.id][slot])
                    # Duration=2: lesson starting at p-1 also occupies slot p
                    if l.duration == 2 and p > 0:
                        prev = f"{d}-{p-1}"
                        if prev in x[l.id]:
                            vars_at_slot.append(x[l.id][prev])
                if len(vars_at_slot) > 1:
                    model.add(sum(vars_at_slot) <= 1)

    # ── HARD 3: Class once per slot ───────────────────────────────────────────
    class_lessons: dict[str, list[Lesson]] = defaultdict(list)
    for lesson in payload.lessons:
        class_lessons[lesson.class_id].append(lesson)

    for class_id, lessons in class_lessons.items():
        for d in range(D):
            for p in range(P):
                vars_at_slot = []
                for l in lessons:
                    slot = f"{d}-{p}"
                    if slot in x[l.id]:
                        vars_at_slot.append(x[l.id][slot])
                    if l.duration == 2 and p > 0:
                        prev = f"{d}-{p-1}"
                        if prev in x[l.id]:
                            vars_at_slot.append(x[l.id][prev])
                if len(vars_at_slot) > 1:
                    model.add(sum(vars_at_slot) <= 1)

    # ── HARD 4: Teacher max per day/week ──────────────────────────────────────
    for teacher_id, lessons in teacher_lessons.items():
        tcfg = teacher_cfg.get(teacher_id)
        if not tcfg:
            continue
        # Per day
        if tcfg.max_lessons_per_day:
            for d in range(D):
                day_vars = [
                    v for l in lessons
                    for slot, v in x[l.id].items()
                    if parse_slot(slot)[0] == d
                ]
                if day_vars:
                    model.add(sum(day_vars) <= tcfg.max_lessons_per_day)
        # Per week
        if tcfg.max_lessons_per_week:
            all_vars = [v for l in lessons for v in x[l.id].values()]
            if all_vars:
                model.add(sum(all_vars) <= tcfg.max_lessons_per_week)
        # Min per week (normă)
        if tcfg.min_lessons_per_week:
            all_vars = [v for l in lessons for v in x[l.id].values()]
            if all_vars:
                model.add(sum(all_vars) >= tcfg.min_lessons_per_week)

    # ── HARD 5: Class max per day ─────────────────────────────────────────────
    for class_id, lessons in class_lessons.items():
        ccfg = class_cfg.get(class_id)
        max_pd = ccfg.max_lessons_per_day if ccfg else 8
        for d in range(D):
            day_vars = [
                v for l in lessons
                for slot, v in x[l.id].items()
                if parse_slot(slot)[0] == d
            ]
            if day_vars:
                model.add(sum(day_vars) <= max_pd)

    # ── SOFT objective — temporarily disabled for debugging ─────────────────
    # Toate soft constraints dezactivate pentru a confirma feasibility
    objective = []
    logger.info("  Soft constraints: DISABLED for debug")

    # ── Model stats ──────────────────────────────────────────────────────────
    proto = model.proto
    logger.info(f"  Model: {len(proto.variables)} vars, {len(proto.constraints)} constraints")

    # ── Pre-solve sanity check ───────────────────────────────────────────────
    # Log exact constraints per teacher to understand infeasibility
    for teacher_id, lessons in teacher_lessons.items():
        tname = next((t.name for t in payload.teachers if t.id == teacher_id), teacher_id[:8])
        tcfg  = next((t for t in payload.teachers if t.id == teacher_id), None)
        classes_for_teacher = {}
        for l in lessons:
            cname = next((c.name for c in payload.classes if c.id == l.class_id), l.class_id[:8])
            classes_for_teacher[cname] = classes_for_teacher.get(cname, 0) + 1
        total = len(lessons)
        avail = len(set(s for l in lessons for s in l.allowed_slots))
        max_pd = tcfg.max_lessons_per_day if tcfg and tcfg.max_lessons_per_day else P
        max_pw = tcfg.max_lessons_per_week if tcfg and tcfg.max_lessons_per_week else "∞"
        logger.info(
            f"  [CHECK] {tname}: {total} lecții, {avail} sloturi avail, "
            f"max {max_pd}/zi × {D}z = {max_pd*D} capacitate. "
            f"Clase: {classes_for_teacher}. "
            f"{'✓ OK' if total <= avail and total <= max_pd * D else '✗ IMPOSIBIL'}"
        )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = payload.solver_time_limit_seconds
    solver.parameters.num_search_workers  = 4
    solver.parameters.log_search_progress = False

    status      = solver.solve(model)
    status_name = solver.status_name(status)
    obj_val     = solver.objective_value if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None

    logger.info(f"  Status: {status_name} in {solver.wall_time:.2f}s obj={obj_val}")

    # ── Identify which constraint causes INFEASIBLE ───────────────────────────
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("  Trying assumption-based diagnosis...")
        # Test 1: remove soft objective — does it become feasible?
        test_model = cp_model.CpModel()
        # Rebuild only hard constraints (simplified)
        tx = {}
        for lesson in payload.lessons:
            tx[lesson.id] = {}
            for slot in lesson.allowed_slots:
                tx[lesson.id][slot] = test_model.new_bool_var(f"t_{lesson.id[:6]}_{slot}")
        # HARD 1
        for lesson in payload.lessons:
            test_model.add(sum(tx[lesson.id].values()) == 1)
        # HARD 2: teacher
        for teacher_id, lessons in teacher_lessons.items():
            for d in range(D):
                for p in range(P):
                    vars_at = []
                    for l in lessons:
                        if f"{d}-{p}" in tx[l.id]: vars_at.append(tx[l.id][f"{d}-{p}"])
                        if l.duration == 2 and p > 0 and f"{d}-{p-1}" in tx[l.id]:
                            vars_at.append(tx[l.id][f"{d}-{p-1}"])
                    if len(vars_at) > 1: test_model.add(sum(vars_at) <= 1)
        # HARD 3: class
        for class_id, lessons in class_lessons.items():
            for d in range(D):
                for p in range(P):
                    vars_at = []
                    for l in lessons:
                        if f"{d}-{p}" in tx[l.id]: vars_at.append(tx[l.id][f"{d}-{p}"])
                        if l.duration == 2 and p > 0 and f"{d}-{p-1}" in tx[l.id]:
                            vars_at.append(tx[l.id][f"{d}-{p-1}"])
                    if len(vars_at) > 1: test_model.add(sum(vars_at) <= 1)
        # HARD 4: teacher max/day/week
        for teacher_id, lessons in teacher_lessons.items():
            tcfg = teacher_cfg.get(teacher_id)
            if tcfg and tcfg.max_lessons_per_day:
                for d in range(D):
                    dv = [tx[l.id][s] for l in lessons for s in tx[l.id] if int(s.split('-')[0]) == d]
                    if dv: test_model.add(sum(dv) <= tcfg.max_lessons_per_day)
            if tcfg and tcfg.max_lessons_per_week:
                aw = [v for l in lessons for v in tx[l.id].values()]
                if aw: test_model.add(sum(aw) <= tcfg.max_lessons_per_week)
        # HARD 5: class max/day
        for class_id, lessons in class_lessons.items():
            ccfg = class_cfg.get(class_id)
            max_pd = ccfg.max_lessons_per_day if ccfg else P
            for d in range(D):
                dv = [tx[l.id][s] for l in lessons for s in tx[l.id] if int(s.split('-')[0]) == d]
                if dv: test_model.add(sum(dv) <= max_pd)

        test_solver = cp_model.CpSolver()
        test_solver.parameters.max_time_in_seconds = 10
        test_status = test_solver.solve(test_model)
        if test_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.warning("  DIAGNOSIS: Hard constraints ONLY → FEASIBLE. Problema e in soft constraints sau interactiunea lor.")
        else:
            logger.warning(f"  DIAGNOSIS: Hard constraints alone → {test_solver.status_name(test_status)}. Problema e in hard constraints.")
            # Find which HARD constraint fails
            # Test without HARD 4 (teacher max/day)
            test2 = cp_model.CpModel()
            tx2 = {}
            for lesson in payload.lessons:
                tx2[lesson.id] = {}
                for slot in lesson.allowed_slots:
                    tx2[lesson.id][slot] = test2.new_bool_var(f"t2_{lesson.id[:6]}_{slot}")
            for lesson in payload.lessons:
                test2.add(sum(tx2[lesson.id].values()) == 1)
            for teacher_id, lessons in teacher_lessons.items():
                for d in range(D):
                    for p in range(P):
                        vars_at = [tx2[l.id][f"{d}-{p}"] for l in lessons if f"{d}-{p}" in tx2[l.id]]
                        if len(vars_at) > 1: test2.add(sum(vars_at) <= 1)
            for class_id, lessons in class_lessons.items():
                for d in range(D):
                    for p in range(P):
                        vars_at = [tx2[l.id][f"{d}-{p}"] for l in lessons if f"{d}-{p}" in tx2[l.id]]
                        if len(vars_at) > 1: test2.add(sum(vars_at) <= 1)
            t2s = test_solver.solve(test2)
            if t2s in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                logger.warning("  DIAGNOSIS: Fara HARD4+5 → FEASIBLE. Problema e in max_lessons_per_day/week.")
            else:
                logger.warning("  DIAGNOSIS: Problema e in HARD1+2+3 (placement/teacher/class overlap).")
    debug_log.append({
        "type":         "solver_status",
        "status":       status_name,
        "time_seconds": round(solver.wall_time, 2),
        "feasible":     status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        "objective":    obj_val,
    })

    total = len(payload.lessons)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        reasons = _analyze_infeasibility(payload, teacher_lessons, class_lessons, D, P)
        for r in reasons:
            logger.warning(f"  INFEASIBLE: {r}")
        return SchoolResponse(
            timetable=[], debug_log=debug_log,
            violations=[SchoolViolation(type="infeasible", message=r) for r in reasons],
            stats=SchoolStats(total_lessons=total, scheduled_lessons=0,
                              solver_status=status_name, solve_time_seconds=solver.wall_time)
        )

    # ── Extract ───────────────────────────────────────────────────────────────
    timetable: list[PlacedLesson] = []
    for lesson in payload.lessons:
        for slot, var in x[lesson.id].items():
            if solver.value(var) == 1:
                d, p = parse_slot(slot)
                timetable.append(PlacedLesson(
                    lesson_id=  lesson.id,
                    class_id=   lesson.class_id,
                    subject_id= lesson.subject_id,
                    teacher_id= lesson.teacher_id,
                    room_id=    lesson.preferred_room_id,
                    day=d, period=p,
                    duration=   lesson.duration,
                ))

    # Post-solve validation
    from collections import Counter
    for teacher_id in teacher_lessons:
        t_lessons = [l for l in timetable if l.teacher_id == teacher_id]
        slot_counts = Counter((l.day, l.period) for l in t_lessons)
        dups = {k: v for k, v in slot_counts.items() if v > 1}
        if dups:
            logger.error(f"  TEACHER CONFLICT {teacher_id[:8]}: {dups}")
        logger.info(f"  teacher {teacher_id[:8]}: {len(t_lessons)} lessons → {sorted(slot_counts.keys())}")
        debug_log.append({
            "type":       "teacher_result",
            "teacher_id": teacher_id,
            "lessons":    len(t_lessons),
            "slots":      sorted(slot_counts.keys()),
            "conflict":   bool(dups),
        })

    violations = []
    placed_count = Counter(l.lesson_id for l in timetable)
    for lesson in payload.lessons:
        if placed_count.get(lesson.id, 0) == 0:
            violations.append(SchoolViolation(
                type="unscheduled",
                message=f"Lecție neplanificată: class={lesson.class_id[:8]} subj={lesson.subject_id[:8]}"
            ))

    debug_log.append({
        "type":       "summary",
        "total":      total,
        "scheduled":  len(timetable),
        "violations": len(violations),
    })
    logger.info(f"=== Done: {len(timetable)}/{total} lessons, {len(violations)} violations ===")

    return SchoolResponse(
        timetable=timetable, violations=violations, debug_log=debug_log,
        stats=SchoolStats(
            total_lessons=total, scheduled_lessons=len(timetable),
            solver_status=status_name, solve_time_seconds=solver.wall_time,
            objective_value=obj_val,
        )
    )


def _analyze_infeasibility(payload, teacher_lessons, class_lessons, D, P):
    """
    Analiză detaliată a motivelor de infeasibility.
    Folosește nume reale, verifică scenariile comune.
    """
    from collections import defaultdict
    reasons = []
    total_slots = D * P

    # Index nume
    teacher_names = {t.id: t.name for t in payload.teachers}
    class_names   = {c.id: c.name for c in payload.classes}

    # Acumulează ore per profesor per clasă
    teacher_hours:       dict[str, int]             = defaultdict(int)
    teacher_class_hours: dict[str, dict[str, int]]  = defaultdict(lambda: defaultdict(int))
    class_hours:         dict[str, int]             = defaultdict(int)

    for lesson in payload.lessons:
        cname = class_names.get(lesson.class_id, lesson.class_id[:8])
        tname = teacher_names.get(lesson.teacher_id, lesson.teacher_id[:8])
        teacher_hours[tname]             += 1
        teacher_class_hours[tname][cname]+= 1
        class_hours[cname]              += 1

    # ── 1. Lecții fără sloturi valide ────────────────────────────────────────
    for lesson in payload.lessons:
        if not x_has_slots(lesson, D, P):
            tname = teacher_names.get(lesson.teacher_id, lesson.teacher_id[:8])
            cname = class_names.get(lesson.class_id, lesson.class_id[:8])
            reasons.append(f"{tname} → {cname}: niciun slot valid disponibil")

    # ── 2. Verificări per profesor ────────────────────────────────────────────
    for teacher_id, lessons in teacher_lessons.items():
        tname  = teacher_names.get(teacher_id, teacher_id[:8])
        total  = len(lessons)
        tcfg   = next((t for t in payload.teachers if t.id == teacher_id), None)
        max_pd = tcfg.max_lessons_per_day  if tcfg and tcfg.max_lessons_per_day  else P
        max_pw = tcfg.max_lessons_per_week if tcfg and tcfg.max_lessons_per_week else None

        # a) Depășire normă săptămânală
        if max_pw and total > max_pw:
            reasons.append(
                f"{tname}: {total} ore/săpt depășește limita de {max_pw} ore/săpt"
            )

        # b) Ore/zi medii depășesc max/zi
        avg_per_day = total / D
        if avg_per_day > max_pd:
            reasons.append(
                f"{tname}: {total} ore în {D} zile = {avg_per_day:.1f} ore/zi medie "
                f"> limita {max_pd}/zi → imposibil de distribuit"
            )

        # c) Numărul de clase depășește max ore/zi
        #    Profesorul trebuie să fie în fiecare clasă cel puțin 1 oră/zi
        #    Deci are nevoie de minim 1 slot per clasă per zi → N clase = N ore/zi minim
        n_classes = len(teacher_class_hours[tname])
        if n_classes > max_pd:
            reasons.append(
                f"{tname}: predă la {n_classes} clase dar max {max_pd} ore/zi — "
                f"nu poate fi în toate clasele zilnic. "
                f"Soluție: mărește max ore/zi la ≥{n_classes} sau distribuie clasele la mai mulți profesori."
            )

        # d) Sloturi disponibile insuficiente față de lecții
        all_allowed: set[str] = set()
        for l in lessons:
            all_allowed.update(l.allowed_slots)
        if len(all_allowed) < total:
            reasons.append(
                f"{tname}: {total} lecții dar doar {len(all_allowed)} sloturi disponibile "
                f"după indisponibilități — imposibil să plaseze toate lecțiile"
            )

    # ── 3. Verificări per clasă ───────────────────────────────────────────────
    for class_id, lessons in class_lessons.items():
        cname  = class_names.get(class_id, class_id[:8])
        total  = len(lessons)
        ccfg   = next((c for c in payload.classes if c.id == class_id), None)
        max_pd = ccfg.max_lessons_per_day if ccfg else P

        if total > D * max_pd:
            reasons.append(
                f"Clasa {cname}: {total} ore/săpt > {D}z × {max_pd} ore/zi = {D*max_pd} sloturi maxime"
            )

    # ── 4. Soft constraints prea stricte ────────────────────────────────────
    # Dacă nu am găsit cauze hard, soft constraints cu weight mare pot fi cauza
    w = payload.soft_rules.weights if payload.soft_rules else {}
    soft_issues = []
    if w.get('sameSubject', 0) > 50:
        # avoidSameSubjectTwicePerDay limitează la 1 lecție/zi per clasă per materie
        # Dacă o materie are >D ore/săpt la o clasă → imposibil
        for class_id, lessons in class_lessons.items():
            cname = class_names.get(class_id, class_id[:8])
            subj_counts: dict[str, int] = defaultdict(int)
            for l in lessons:
                subj_counts[l.subject_id] += 1
            for subj_id, count in subj_counts.items():
                if count > D:
                    soft_issues.append(
                        f"Clasa {cname}: {count} ore dintr-o materie > {D} zile — "
                        f"'avoidSameSubjectTwicePerDay' (weight={w.get('sameSubject')}) "
                        f"face imposibilă plasarea. Reduce weight-ul sau numărul de ore."
                    )
    if w.get('startFirst', 0) > 70 and len(teacher_lessons) >= 3:
        soft_issues.append(
            f"'startFromFirstSlot' (weight={w.get('startFirst')}) cu {len(teacher_lessons)} profesori "
            f"poate crea conflicte — toți vor primul slot. Încearcă să reduci weight-ul sub 50."
        )
    for issue in soft_issues:
        reasons.append(f"⚠ Soft constraint: {issue}")

    # ── 5. Sumar intrări (mereu afișat) ──────────────────────────────────────
    total_lessons = len(payload.lessons)
    reasons.append(
        f"[SUMAR] {total_lessons} lecții · {len(class_hours)} clase · "
        f"{len(teacher_hours)} profesori · {D}z × {P}sl = {total_slots} sloturi/profesor"
    )

    for tname, total in sorted(teacher_hours.items(), key=lambda x: -x[1]):
        tcfg  = next((t for t in payload.teachers if teacher_names.get(t.id) == tname), None)
        max_w = tcfg.max_lessons_per_week if tcfg and tcfg.max_lessons_per_week else "∞"
        max_d = tcfg.max_lessons_per_day  if tcfg and tcfg.max_lessons_per_day  else "∞"
        avg   = total / D
        clase_str = ", ".join(f"{c}:{h}h" for c, h in sorted(teacher_class_hours[tname].items()))
        warn  = " ⚠" if (isinstance(max_w, int) and total > max_w) or avg > (max_d if isinstance(max_d, int) else 999) else ""
        reasons.append(
            f"[PROF] {tname}: {total}h/săpt, {avg:.1f}h/zi medie "
            f"(max {max_w}/săpt, {max_d}/zi){warn} → {clase_str}"
        )

    for cname, total in sorted(class_hours.items()):
        ccfg   = next((c for c in payload.classes if class_names.get(c.id) == cname), None)
        max_pd = ccfg.max_lessons_per_day if ccfg else P
        warn   = " ⚠" if total > D * max_pd else ""
        reasons.append(
            f"[CLASĂ] {cname}: {total}h/săpt (max {max_pd}/zi × {D}z = {D*max_pd}){warn}"
        )

    return reasons


def x_has_slots(lesson, D, P):
    slots = lesson.allowed_slots
    if lesson.duration == 2:
        slots = [s for s in slots
                 if (dp := (int(s.split('-')[0]), int(s.split('-')[1])))[1] < P - 1
                 and f"{dp[0]}-{dp[1]+1}" in lesson.allowed_slots]
    return len(slots) > 0


def x_get_slots(lesson, D, P):
    return lesson.allowed_slots