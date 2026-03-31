// POST /api/schedules/[id]/generate — pipeline generare orar business:
//   1. Încarcă angajați, ture, constrângeri, concedii, sărbători
//   2. Apelează CP-SAT solver (/solve/business) — fallback la greedy dacă indisponibil
//   3. Salvează asignările în shift_assignments
//   4. Analiză AI opțională
//   5. Update status + AI suggestions
//   6. Audit log
// Used by: ScheduleActions 'Generează' + ConstraintsPanel 'Salvează și regenerează'.

import { NextResponse } from 'next/server'
import { createServerSupabaseClient, createAdminClient } from '@/lib/supabase/server'
import { getOrgContext } from '@/lib/dal/org'
import { generateSchedule } from '@/lib/algorithms/generate'
import { analyzeScheduleWithAI } from '@/lib/ai/analyze'
import { getHolidaysInRange } from '@/lib/holidays'
import { addDays, format, parseISO } from 'date-fns'

const SOLVER_URL = process.env.SOLVER_URL
console.log('[generate] SOLVER_URL:', SOLVER_URL ?? '(not set — will use greedy)')

function buildWorkingDates(schedule: any, holidays: any[]): string[] {
  const dates: string[] = []
  const holidaySet = new Set(holidays.map((h: any) => h.date))
  let current = parseISO(schedule.start_date)
  const end = parseISO(schedule.end_date)
  while (current <= end) {
    const dateStr = format(current, 'yyyy-MM-dd')
    const dow = current.getDay() === 0 ? 7 : current.getDay()
    const isWorkingDay = schedule.working_days?.includes(dow)
    const skipHoliday = holidaySet.has(dateStr) && !schedule.include_holidays
    if (isWorkingDay && !skipHoliday) dates.push(dateStr)
    current = addDays(current, 1)
  }
  return dates
}

function shiftDurationHours(shift: any): number {
  const [sh, sm] = shift.start_time.split(':').map(Number)
  const [eh, em] = shift.end_time.split(':').map(Number)
  let mins = (eh * 60 + em) - (sh * 60 + sm)
  if (mins < 0) mins += 24 * 60
  return mins / 60
}

async function callCpSatSolver(payload: any): Promise<any | null> {
  if (!SOLVER_URL) return null
  try {
    const res = await fetch(`${SOLVER_URL}/solve/business`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(45_000),
    })
    if (!res.ok) {
      console.error(`CP-SAT /solve/business returned ${res.status}:`, await res.text().catch(() => ''))
      return null
    }
    return await res.json()
  } catch (err: any) {
    console.warn('CP-SAT solver unavailable — falling back to greedy:', err.message)
    return null
  }
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const supabase = await createServerSupabaseClient()
  const admin = createAdminClient()

  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const ctx = await getOrgContext(user.id)
  if (!ctx) return NextResponse.json({ error: 'No organization' }, { status: 400 })
  const orgId = ctx.org.id

  const empIds = (await admin.from('employees').select('id').eq('organization_id', orgId))
    .data?.map((e: any) => e.id) ?? []

  const [scheduleRes, employeesRes, shiftDefsRes, scheduleShiftsRes, constraintsRes, leavesRes, unavailRes] =
    await Promise.all([
      admin.from('schedules').select('*').eq('id', id).eq('organization_id', orgId).single(),
      admin.from('employees').select('*').eq('organization_id', orgId).eq('is_active', true),
      admin.from('shift_definitions').select('*').eq('organization_id', orgId),
      admin.from('schedule_shifts').select('*').eq('schedule_id', id),
      admin.from('constraints').select('*').eq('schedule_id', id).eq('is_active', true),
      admin.from('employee_leaves').select('*').in('employee_id', empIds),
      admin.from('employee_unavailability').select('*').in('employee_id', empIds),
    ])

  const schedule = scheduleRes.data
  if (!schedule) return NextResponse.json({ error: 'Schedule not found' }, { status: 404 })

  await admin.from('schedules').update({ status: 'generating' }).eq('id', id)

  const holidays = await getHolidaysInRange(schedule.country_code, schedule.start_date, schedule.end_date)

  const slotsPerShift: Record<string, number> = {}
  for (const ss of scheduleShiftsRes.data ?? []) slotsPerShift[ss.shift_definition_id] = ss.slots_per_day

  const linkedShiftIds = new Set((scheduleShiftsRes.data ?? []).map((ss: any) => ss.shift_definition_id))
  const shiftDefs = (shiftDefsRes.data ?? []).filter((s: any) => linkedShiftIds.has(s.id))
  const gc = (schedule.generation_config as any) ?? {}
  const workingDates = buildWorkingDates(schedule, holidays)

  // ── Try CP-SAT solver (/solve/business) ──────────────────────────────────
  let assignments: any[] = []
  let violations: any[] = []
  let stats: any = {}
  let usedSolver = 'greedy'

  if (SOLVER_URL) {
    const solverResult = await callCpSatSolver({
      schedule_id: id,
      employees: (employeesRes.data ?? []).map((e: any) => ({
        id: e.id, name: e.name, experience_level: e.experience_level, color: e.color,
      })),
      shift_definitions: shiftDefs.map((s: any) => ({
        id: s.id, name: s.name,
        shift_type: s.shift_type,
        start_time: s.start_time,
        end_time: s.end_time,
        crosses_midnight: s.crosses_midnight ?? false,
        duration_hours: shiftDurationHours(s),
      })),
      working_dates: workingDates,
      slots_per_shift: slotsPerShift,
      constraints: (constraintsRes.data ?? []).map((c: any) => ({
        type: c.type, employee_id: c.employee_id,
        target_employee_id: c.target_employee_id,
        shift_definition_id: c.shift_definition_id,
        value: c.value, note: c.note,
        is_active: c.is_active,
      })),
      leaves: (leavesRes.data ?? []).map((l: any) => ({
        employee_id: l.employee_id, start_date: l.start_date, end_date: l.end_date,
      })),
      unavailability: (unavailRes.data ?? []).map((u: any) => ({
        employee_id: u.employee_id, date: u.date, day_of_week: u.day_of_week,
      })),
      config: {
        min_employees_per_shift:       gc.min_employees_per_shift ?? 1,
        max_consecutive_days:          gc.max_consecutive_days ?? 6,
        min_rest_hours_between_shifts: gc.min_rest_hours_between_shifts ?? 11,
        max_weekly_hours:              gc.max_weekly_hours ?? 48,
        max_night_shifts_per_week:     gc.max_night_shifts_per_week ?? 3,
        enforce_legal_limits:          gc.enforce_legal_limits ?? true,
        balance_shift_distribution:    gc.balance_shift_distribution ?? true,
        shift_consistency:             gc.shift_consistency ?? 2,
      },
      solver_time_limit_seconds: Math.min(
        30,
        Math.max(10, Math.ceil((employeesRes.data?.length ?? 1) * workingDates.length / 20))
      ),
    })

    if (solverResult?.assignments?.length > 0) {
      usedSolver = `cp-sat (${solverResult.stats?.solver_status ?? 'FEASIBLE'})`
      assignments = solverResult.assignments.map((a: any) => ({
        schedule_id: id,
        employee_id: a.employee_id,
        shift_definition_id: a.shift_definition_id,
        date: a.date,
        is_manual_override: false,
        role_in_shift: null,
        note: null,
      }))
      violations = (solverResult.violations ?? []).map((v: any) => ({
        type: v.type, employeeId: v.employee_id, employeeName: v.employee_name,
        date: v.date, message: v.message,
      }))
      stats = {
        totalSlots: solverResult.stats?.total_slots ?? 0,
        filledSlots: assignments.length,
        hoursPerEmployee: solverResult.stats?.hours_per_employee ?? {},
        solver: usedSolver,
      }
    }
  }

  // ── Fallback greedy ───────────────────────────────────────────────────────
  if (assignments.length === 0) {
    const result = generateSchedule({
      schedule: schedule as any,
      employees: employeesRes.data ?? [],
      shiftDefinitions: shiftDefs as any,
      constraints: constraintsRes.data as any ?? [],
      leaves: leavesRes.data ?? [],
      unavailability: unavailRes.data ?? [],
      holidays,
      slotsPerShift,
    })
    assignments = result.assignments as any[]
    violations  = result.violations
    stats       = { ...result.stats, solver: 'greedy' }
    usedSolver  = 'greedy'
  }

  // ── Save assignments ──────────────────────────────────────────────────────
  await admin.from('shift_assignments').delete().eq('schedule_id', id)
  const BATCH = 100
  for (let i = 0; i < assignments.length; i += BATCH) {
    await admin.from('shift_assignments').insert(assignments.slice(i, i + BATCH))
  }

  // ── AI analysis (opțional) ────────────────────────────────────────────────
  const locale = request.headers.get('accept-language')?.startsWith('ro') ? 'ro' : 'en'
  let aiSuggestions = null
  try {
    aiSuggestions = await analyzeScheduleWithAI({
      schedule: schedule as any,
      assignments: assignments as any,
      employees: employeesRes.data ?? [],
      shiftDefinitions: shiftDefs as any,
      constraints: constraintsRes.data as any ?? [],
      violations,
      locale,
    })
  } catch (err: any) {
    console.warn('AI analysis skipped:', err.message)
  }

  await admin.from('schedules').update({
    status: 'generated',
    ai_suggestions: aiSuggestions,
    ai_analyzed_at: new Date().toISOString(),
  }).eq('id', id)

  await admin.from('audit_log').insert({
    user_id: user.id,
    action: 'schedule_generate',
    resource: 'schedule',
    resource_id: id,
    metadata: { filled_slots: assignments.length, violations: violations.length, solver: usedSolver },
  })

  console.log(`[generate] Done — solver: ${usedSolver}, assignments: ${assignments.length}`)
  return NextResponse.json({ success: true, stats, violations, aiSuggestions, solver: usedSolver })
}