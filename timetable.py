
import json, os, sys
from ortools.sat.python import cp_model
from collections import defaultdict


def find_file(name):
    candidates = [
        name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"'{name}' not found. Put rooms.json and subjects.json "
        f"in the same folder as this script.\nSearched: {candidates}"
    )

print("=" * 65)
print("  TIMETABLE GENERATOR")
print("  Universite de Yaounde I  -  Departement Informatique")
print("=" * 65)

# ── rooms.json ───────────────────────────────────────────────────
with open(find_file("rooms.json"), encoding="utf-8") as f:
    rooms_raw = json.load(f)

rooms_list = rooms_raw.get("Informatique", [])
if not rooms_list:
    sys.exit("[ERROR] No rooms found under 'Informatique' in rooms.json")

rooms = [r["num"] for r in rooms_list]
print(f"\n[rooms.json]  {len(rooms)} rooms loaded")
for r in rooms_list:
    print(f"   {r['num']:<8}  cap={r['capacite']:<5}  {r['batiment']}")

# ── subjects.json ────────────────────────────────────────────────
with open(find_file("subjects.json"), encoding="utf-8") as f:
    subjects_raw = json.load(f)

niveaux = subjects_raw.get("niveau", {})
if not niveaux:
    sys.exit("[ERROR] Key 'niveau' not found in subjects.json")

_unknown_ctr = 0

def make_teacher(lecturers):
    """Build a teacher label; return a unique placeholder when unknown."""
    global _unknown_ctr
    label = "_".join(p.strip() for p in (lecturers or []) if p.strip())
    if not label:
        _unknown_ctr += 1
        return f"__TBD_{_unknown_ctr}__"  
    return label

all_classes = {}   

for lvl, semesters in niveaux.items():
    for sem_key, sem_data in semesters.items():
        seen_codes = set()      
        valid = []
        for subj in sem_data.get("subjects", []):
            code = subj.get("code", "").strip()
            if not code:
                continue

            if code in seen_codes:
                print(f"  [WARN] Duplicate code '{code}' in L{lvl}_{sem_key.upper()} "
                      f"– keeping first occurrence only")
                continue
            seen_codes.add(code)

            name = subj.get("name", "")
            if isinstance(name, list):
                name = " ".join(name).strip()
            name = name.strip() or code

            teacher = make_teacher(subj.get("Course Lecturer"))

            valid.append({"code": code, "name": name, "teacher": teacher})

        if valid:
            sem_num  = "1" if sem_key.lower() == "s1" else "2"
            class_id = f"L{lvl}_S{sem_num}"
            all_classes[class_id] = valid

if not all_classes:
    sys.exit("[ERROR] No valid classes found in subjects.json")

print(f"\n[subjects.json]  {len(all_classes)} classes loaded")
for cid in sorted(all_classes.keys()):
    print(f"   {cid:<10}  {len(all_classes[cid])} courses")


DAYS    = list(range(6))
PERIODS = list(range(5))
PERIOD_NAMES = {
    0: "07:00-09:55", 1: "10:05-12:55", 2: "13:05-15:55",
    3: "16:05-18:55", 4: "19:05-21:55",
}
DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
WEIGHTS   = {0:5, 1:4, 2:3, 3:2, 4:1}


classes     = sorted(all_classes.keys())
subjects_of = {c: [s["code"] for s in all_classes[c]] for c in classes}
teacher_of  = {(c, s["code"]): s["teacher"]
               for c in classes for s in all_classes[c]}

def semester(class_id):
    return class_id.split("_")[1]   # "L1_S1" → "S1"


print("\n" + "-"*65)
print("  Building CP-SAT model …")

model  = cp_model.CpModel()

# Decision variables  x[(c,s,r,d,p)] ∈ {0,1}
x = {
    (c, s, r, d, p): model.NewBoolVar(f"x|{c}|{s}|{r}|{d}|{p}")
    for c in classes
    for s in subjects_of[c]
    for r in rooms
    for d in DAYS
    for p in PERIODS
}
print(f"  Variables : {len(x):,}")

# C1 – class has at most 1 session per slot
for c in classes:
    for d in DAYS:
        for p in PERIODS:
            model.Add(
                sum(x[c,s,r,d,p]
                    for s in subjects_of[c]
                    for r in rooms) <= 1
            )

# C2 – each course scheduled exactly once
for c in classes:
    for s in subjects_of[c]:
        model.Add(
            sum(x[c,s,r,d,p]
                for r in rooms
                for d in DAYS
                for p in PERIODS) == 1
        )

# C4 – room used by at most 1 class per slot
for r in rooms:
    for d in DAYS:
        for p in PERIODS:
            model.Add(
                sum(x[c,s,r,d,p]
                    for c in classes
                    for s in subjects_of[c]) <= 1
            )

teacher_sem_groups = defaultdict(list)   # (teacher, semester) → [(c,s), …]
for (c, s), t in teacher_of.items():
    if not t.startswith("__TBD_"):
        teacher_sem_groups[(t, semester(c))].append((c, s))

c5_count = 0
for (t, _sem), pairs in teacher_sem_groups.items():
    if len(pairs) < 2:
        continue
    for d in DAYS:
        for p in PERIODS:
            model.Add(
                sum(x[c,s,r,d,p]
                    for (c,s) in pairs
                    for r in rooms) <= 1
            )
            c5_count += 1

print(f"  Constraints : C1 + C2 + C3(implicit) + C4 + C5({c5_count})")

# Objective
model.Maximize(
    sum(WEIGHTS[p] * x[c,s,r,d,p]
        for c in classes
        for s in subjects_of[c]
        for r in rooms
        for d in DAYS
        for p in PERIODS)
)
print("  Objective   : Maximise Σ w[p]·x  (morning sessions prioritised)")


solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds  = 120.0
solver.parameters.log_search_progress  = False

print("\n  Solving … (timeout 120 s)\n")
status = solver.Solve(model)

STATUS = {
    cp_model.OPTIMAL:    "OPTIMAL",
    cp_model.FEASIBLE:   "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN:    "UNKNOWN",
}
print(f"  Status  : {STATUS.get(status, status)}")


if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"  Objective : {solver.ObjectiveValue():.0f}")
    print(f"  Wall time : {solver.WallTime():.2f} s")

    timetable = defaultdict(list)
    for c in classes:
        for s in subjects_of[c]:
            for r in rooms:
                for d in DAYS:
                    for p in PERIODS:
                        if solver.Value(x[c,s,r,d,p]):
                            t = teacher_of[(c,s)]
                            timetable[c].append({
                                "code":        s,
                                "room":        r,
                                "day":         DAY_NAMES[d],  "di": d,
                                "period":      PERIOD_NAMES[p], "pi": p,
                                "teacher":     "TBD" if t.startswith("__TBD_") else t,
                                "before_noon": p < 2,
                            })

    W  = "-" * 82
    W2 = "=" * 82
    for c in sorted(timetable.keys()):
        entries = sorted(timetable[c], key=lambda e: (e["di"], e["pi"]))
        print(f"\n{W2}\n  {c}  ({len(entries)} sessions)\n{W2}")
        print(f"  {'DAY':<12} {'PERIOD':<18} {'COURSE':<12} {'ROOM':<8} TEACHER")
        print(W)
        for e in entries:
            tag = "  <- before noon" if e["before_noon"] else ""
            print(f"  {e['day']:<12} {e['period']:<18} {e['code']:<12} {e['room']:<8} {e['teacher']}{tag}")
        print(W)

    total  = sum(len(v) for v in timetable.values())
    bn     = sum(1 for v in timetable.values() for e in v if e["before_noon"])
    pct    = 100*bn//total if total else 0
    print(f"\n{W2}\n  STATISTICS\n{W2}")
    print(f"  Total sessions   : {total}")
    print(f"  Before noon      : {bn}/{total}  ({pct}%)")
    print(f"  Classes covered  : {len(timetable)}\n{W2}\n")

else:
    total  = sum(len(v) for v in subjects_of.values())
    slots  = len(rooms) * len(DAYS) * len(PERIODS)
    print(f"\n  [!] No feasible solution found.")
    print(f"  Courses to schedule : {total}")
    print(f"  Available slots     : {slots}  "
          f"({len(rooms)} rooms × {len(DAYS)} days × {len(PERIODS)} periods)")
