# OrarPro Solver — FastAPI + OR-Tools CP-SAT

Microservice for schedule generation. Two endpoints:

- `POST /solve/shifts` — shift scheduling (HoReCa, factories, clinics)
- `POST /solve/school` — timetable scheduling (schools, universities)

## Local Development

```bash
cd orarpro-solver
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# API docs available at:
open http://localhost:8000/docs
```

## Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init          # name it: orarpro-solver
railway up

# Get your URL
railway domain        # e.g. https://orarpro-solver-production.up.railway.app
```

Then add to Vercel environment variables:
```
SOLVER_URL=https://orarpro-solver-production.up.railway.app
```

## Environment Variables (Railway)

None required — OR-Tools is self-contained.

Optional:
- `PORT` — set automatically by Railway (default 8000)

## API Examples

### Shifts solver

```bash
curl -X POST https://your-solver.up.railway.app/solve/shifts \
  -H "Content-Type: application/json" \
  -d '{
    "schedule_id": "abc123",
    "employees": [
      {"id": "e1", "name": "Ion", "experience_level": "senior"},
      {"id": "e2", "name": "Maria", "experience_level": "mid"}
    ],
    "shift_definitions": [
      {"id": "s1", "name": "Tura 1", "shift_type": "morning",
       "start_time": "06:00", "end_time": "14:00",
       "crosses_midnight": false, "duration_hours": 8}
    ],
    "working_dates": ["2026-04-01", "2026-04-02"],
    "slots_per_shift": {"s1": 1},
    "config": {
      "min_employees_per_shift": 1,
      "max_consecutive_days": 6,
      "min_rest_hours_between_shifts": 11,
      "max_weekly_hours": 48,
      "max_night_shifts_per_week": 3,
      "enforce_legal_limits": true,
      "balance_shift_distribution": true,
      "shift_consistency": 2
    }
  }'
```

### School solver

```bash
curl -X POST https://your-solver.up.railway.app/solve/school \
  -H "Content-Type: application/json" \
  -d '{
    "schedule_id": "abc123",
    "teachers": [
      {"id": "t1", "name": "Prof. Ionescu", "subject_ids": ["math"],
       "max_periods_per_day": 6, "max_periods_per_week": 20}
    ],
    "subjects": [
      {"id": "math", "name": "Matematică", "periods_per_week": 4,
       "room_type": "classroom", "preferred_morning": true}
    ],
    "classes": [
      {"id": "c1", "name": "10A", "subject_ids": ["math"]}
    ],
    "rooms": [
      {"id": "r1", "name": "Sala 101", "room_type": "classroom", "capacity": 30}
    ],
    "days_per_week": 5,
    "periods_per_day": 8
  }'
```

## Architecture

```
Next.js (Vercel)
  └── POST /api/schedules/[id]/generate
        └── fetch → SOLVER_URL/solve/shifts
              └── OR-Tools CP-SAT
              └── returns assignments[]
        └── save to Supabase
```

## Fallback

If the solver is unreachable (Railway sleeping), Next.js falls back
to the built-in greedy algorithm automatically.
