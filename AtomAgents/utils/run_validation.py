"""
Post-run validation gates for LAMMPS and FEniCSx results.

Every check exists because a real session shipped garbage as PASS:
- a 100k-step "production" run with no time integrator (Temp frozen, atoms
  never moved) was reported "Run succeeded";
- stress.dat harvested from a blown-up run ended at 3.4e11 bar (~34 TPa);
- a stress-strain file interleaved two run phases (duplicate strain values);
- a singular-BC FEM solve (745 iterations, PETSc reason=3) had its figures
  published; reported iteration counts contradicted solver.log in 3 of 4
  sessions;
- gmsh full-plate meshes silently lost every boundary tag except the hole;
- one params.json wasn't even valid JSON.

Verdicts are machine-readable; agents must quote them verbatim instead of
narrating their own success. Thresholds are env-tunable (ATOMAGENTS_VAL_*).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _check(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _finish(checks: List[Dict[str, str]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overall = PASS
    for c in checks:
        if _SEVERITY[c["status"]] > _SEVERITY[overall]:
            overall = c["status"]
    verdict: Dict[str, Any] = {"overall": overall, "checks": checks}
    if extra:
        verdict.update(extra)
    return verdict


def format_verdict_markdown(verdict: Dict[str, Any]) -> str:
    lines = [f"**Validation verdict: {verdict.get('overall', '?')}**", ""]
    for c in verdict.get("checks", []):
        lines.append(f"- [{c['status']}] {c['name']}: {c['detail']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LAMMPS
# ---------------------------------------------------------------------------

# Fix styles that perform time integration (a run without one does not move
# atoms — the box may still deform, producing a fake stress-strain curve).
_INTEGRATING_FIX = re.compile(
    r"^\s*fix\s+\S+\s+\S+\s+(nve|nvt|npt|nph|langevin/(?:drude)?|brownian|rigid\S*|move|gld|gle|nvt/sllod|nve/limit|nve/noforce)\b",
    re.MULTILINE,
)
_FIX_DEFORM = re.compile(r"^\s*fix\s+\S+\s+\S+\s+deform\b", re.MULTILINE)
_UNFIX = re.compile(r"^\s*unfix\s+(\S+)", re.MULTILINE)
_RUN_CMD = re.compile(r"^\s*run\s+\d+", re.MULTILINE)


def _parse_thermo_segments(log_text: str) -> List[Dict[str, List[float]]]:
    """Parse a LAMMPS log into thermo segments (one per run/reset_timestep
    block). Each segment maps column name -> list of float values."""
    segments: List[Dict[str, List[float]]] = []
    current: Optional[Dict[str, List[float]]] = None
    headers: List[str] = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Step") and len(stripped.split()) > 1:
            headers = stripped.split()
            current = {h: [] for h in headers}
            segments.append(current)
            continue
        if current is not None:
            parts = stripped.split()
            if len(parts) == len(headers):
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    current = None  # e.g. "Loop time of ..." or banner lines
                    continue
                for h, v in zip(headers, values):
                    current[h].append(v)
            elif stripped.startswith(("Loop time", "WARNING", "ERROR")):
                current = None
    return [s for s in segments if s.get("Step")]


def validate_lammps_run(
    script_text: str,
    log_text: str,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a completed LAMMPS run from its input script and log."""
    checks: List[Dict[str, str]] = []
    max_pressure_warn = _env_float("ATOMAGENTS_VAL_MAX_PRESSURE_BAR", 1e7)  # 1 TPa
    max_pressure_fail = _env_float("ATOMAGENTS_VAL_MAX_PRESSURE_BAR_FAIL", 1e9)
    energy_growth_warn = _env_float("ATOMAGENTS_VAL_ENERGY_GROWTH_WARN", 1e2)
    energy_growth_fail = _env_float("ATOMAGENTS_VAL_ENERGY_GROWTH_FAIL", 1e4)

    # --- log-level fatals ---------------------------------------------------
    errors = re.findall(r"^ERROR.*$", log_text, re.MULTILINE)
    if errors:
        checks.append(_check("lammps_errors", FAIL, errors[-1][:300]))
    if "No fixes with time integration" in log_text:
        checks.append(
            _check(
                "time_integrator",
                FAIL,
                "LAMMPS warned 'No fixes with time integration' — atoms never "
                "moved; any deformation output is a fake affine remap.",
            )
        )
    elif script_text and _FIX_DEFORM.search(script_text) and _RUN_CMD.search(script_text):
        if not _INTEGRATING_FIX.search(script_text):
            checks.append(
                _check(
                    "time_integrator",
                    FAIL,
                    "Script uses fix deform + run but defines no time-integrating "
                    "fix (nve/nvt/npt/...).",
                )
            )
        else:
            checks.append(_check("time_integrator", PASS, "integrating fix present"))
    if "Inconsistent image flags" in log_text:
        checks.append(
            _check(
                "image_flags",
                WARN,
                "'Inconsistent image flags' warning — per-atom unwrapped "
                "coordinates and derived strain may be wrong.",
            )
        )
    m = re.search(r"Dangerous builds = (\d+)", log_text)
    if m and int(m.group(1)) > 0:
        checks.append(_check("dangerous_builds", WARN, f"{m.group(1)} dangerous neighbor builds"))

    # --- thermo checks ------------------------------------------------------
    segments = _parse_thermo_segments(log_text)
    saw_nan = False
    frozen = False
    blowup: Optional[str] = None
    over_pressure: Optional[str] = None
    for seg in segments:
        for col, values in seg.items():
            if any(math.isnan(v) or math.isinf(v) for v in values):
                saw_nan = True
        temp = seg.get("Temp", [])
        steps = seg.get("Step", [])
        if len(temp) >= 10 and steps and steps[0] != steps[-1]:
            if max(temp) - min(temp) == 0.0 and temp[0] != 0.0:
                frozen = True
        for ecol in ("TotEng", "PotEng"):
            e = seg.get(ecol, [])
            if len(e) >= 5 and abs(e[0]) > 1e-6:
                growth = max(abs(v) for v in e) / abs(e[0])
                if growth > energy_growth_fail:
                    blowup = f"{ecol} grew {growth:.3g}x within one run segment"
                elif growth > energy_growth_warn and blowup is None:
                    blowup = f"WARN:{ecol} grew {growth:.3g}x within one run segment"
        for pcol in ("Press", "Pxx", "Pyy", "Pzz"):
            p = seg.get(pcol, [])
            if p:
                pmax = max(abs(v) for v in p)
                if pmax > max_pressure_fail:
                    over_pressure = f"{pcol} reached {pmax:.3g} bar"
                elif pmax > max_pressure_warn and over_pressure is None:
                    over_pressure = f"WARN:{pcol} reached {pmax:.3g} bar"
    if saw_nan:
        checks.append(_check("nan_inf", FAIL, "NaN/inf in thermo output"))
    if frozen:
        checks.append(
            _check(
                "frozen_dynamics",
                FAIL,
                "Temp constant over an entire run segment — no dynamics occurred.",
            )
        )
    if blowup:
        if blowup.startswith("WARN:"):
            checks.append(_check("energy_growth", WARN, blowup[5:]))
        else:
            checks.append(_check("energy_growth", FAIL, blowup))
    if over_pressure:
        if over_pressure.startswith("WARN:"):
            checks.append(_check("pressure_magnitude", WARN, over_pressure[5:]))
        else:
            checks.append(
                _check(
                    "pressure_magnitude",
                    FAIL,
                    over_pressure + " — physically implausible; the run blew up.",
                )
            )

    # --- referenced files ---------------------------------------------------
    if script_text and work_dir:
        wd = Path(work_dir)
        missing = []
        for cmd in ("read_data", "read_restart", "include"):
            for m2 in re.finditer(rf"^\s*{cmd}\s+(\S+)", script_text, re.MULTILINE):
                ref = m2.group(1)
                p = Path(ref) if ref.startswith("/") else wd / ref
                if not p.exists():
                    missing.append(ref)
        for m2 in re.finditer(r"^\s*pair_coeff\s+.*?(\S+\.(?:eam\S*|sw|tersoff|airebo|meam\S*|table))\b",
                              script_text, re.MULTILINE):
            ref = m2.group(1)
            p = Path(ref) if ref.startswith("/") else wd / ref
            if not p.exists():
                missing.append(ref)
        if missing:
            checks.append(_check("referenced_files", FAIL, f"missing: {missing}"))
        else:
            checks.append(_check("referenced_files", PASS, "all referenced files exist"))

    if not segments and not errors:
        checks.append(_check("thermo_output", WARN, "no thermo table found in log"))
    if not checks:
        checks.append(_check("lammps_run", PASS, "no issues detected"))
    return _finish(checks, {"n_thermo_segments": len(segments)})


def validate_stress_strain(
    strain: Sequence[float],
    stress: Sequence[float],
) -> Dict[str, Any]:
    """Validate a stress-strain dataset before it is plotted/reported."""
    checks: List[Dict[str, str]] = []
    if len(strain) != len(stress) or not strain:
        return _finish([_check("shape", FAIL, "empty or mismatched strain/stress arrays")])

    n_dup = sum(1 for a, b in zip(strain, strain[1:]) if a == b)
    dup_frac = n_dup / max(1, len(strain) - 1)
    dup_frac_fail = _env_float("ATOMAGENTS_VAL_DUP_STRAIN_FRAC", 0.1)
    if dup_frac > dup_frac_fail or n_dup >= 10:
        checks.append(
            _check(
                "duplicate_strains",
                FAIL,
                f"{n_dup} duplicated consecutive strain values ({dup_frac:.0%}) — "
                "the file likely interleaves two run phases (relaxation + "
                "loading); split by reset_timestep segment before extracting.",
            )
        )
    elif n_dup >= 3:
        checks.append(
            _check(
                "duplicate_strains",
                WARN,
                f"{n_dup} duplicated consecutive strain values ({dup_frac:.0%}).",
            )
        )
    if any(b < a for a, b in zip(strain, strain[1:])):
        checks.append(_check("monotonic_strain", WARN, "strain is not monotonically increasing"))

    smax = max(abs(v) for v in stress)
    # Plausibility bound: no simulated material sustains TPa-scale stress
    # (a real session shipped 3.4e7 GPa harvested from a blown-up run).
    max_gpa = _env_float("ATOMAGENTS_VAL_MAX_STRESS_GPA", 1000.0)
    if smax > max_gpa:
        checks.append(
            _check(
                "stress_magnitude",
                FAIL,
                f"max |stress| = {smax:.4g} (assumed GPa) exceeds the "
                f"plausibility bound {max_gpa:.4g} — the source run blew up.",
            )
        )
    s0 = abs(stress[0])
    if smax > 0 and strain[0] == 0.0:
        frac = s0 / smax
        if frac > 0.5:
            checks.append(
                _check(
                    "residual_stress",
                    FAIL,
                    f"|stress|={s0:.4g} at zero strain ({frac:.0%} of max) — the "
                    "system was not relaxed to zero pressure before loading.",
                )
            )
        elif frac > 0.2:
            checks.append(
                _check(
                    "residual_stress",
                    WARN,
                    f"|stress|={s0:.4g} at zero strain ({frac:.0%} of max).",
                )
            )
    if not checks:
        checks.append(_check("stress_strain", PASS, f"{len(strain)} points, max |stress|={smax:.4g}"))
    return _finish(checks)


# ---------------------------------------------------------------------------
# FEM
# ---------------------------------------------------------------------------

_PETSC_LINE = re.compile(
    r"PETSc\s+reason=(-?\d+)\s+its=(\d+)\s+rfinal=([\d.eE+-]+)"
    r"(?:\s+elapsed=([\d.eE+-]+)s)?(?:\s+ndofs=(\d+))?"
)


def parse_fem_solver_log(log_text: str) -> Optional[Dict[str, Any]]:
    """Extract the LAST PETSc summary line — the numbers reports must use
    (LLM-authored metrics contradicted solver.log in 3 of 4 sessions)."""
    matches = list(_PETSC_LINE.finditer(log_text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "reason": int(m.group(1)),
        "iterations": int(m.group(2)),
        "final_residual": float(m.group(3)),
        "elapsed_s": float(m.group(4)) if m.group(4) else None,
        "ndofs": int(m.group(5)) if m.group(5) else None,
    }


def validate_fem_solve(
    log_text: str,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate a FEniCSx/PETSc solve from its solver log."""
    max_iterations = max_iterations or int(_env_float("ATOMAGENTS_VAL_MAX_KSP_ITERS", 200))
    checks: List[Dict[str, str]] = []
    metrics = parse_fem_solver_log(log_text)
    if metrics is None:
        return _finish([_check("solver_log", FAIL, "no PETSc summary line found in log")])
    reason = metrics["reason"]
    its = metrics["iterations"]
    if reason < 0:
        checks.append(_check("convergence", FAIL, f"PETSc diverged (reason={reason})"))
    elif reason == 2:
        checks.append(_check("convergence", PASS, f"converged on rtol (reason=2, its={its})"))
    elif reason == 3:
        checks.append(
            _check(
                "convergence",
                WARN,
                f"converged on ABSOLUTE tolerance only (reason=3, its={its}) — "
                "often a symptom of a singular/ill-conditioned system (e.g. an "
                "unconstrained rigid-body mode).",
            )
        )
    else:
        checks.append(_check("convergence", PASS, f"converged (reason={reason}, its={its})"))
    if its > 3 * max_iterations:
        checks.append(
            _check(
                "iteration_count",
                FAIL,
                f"{its} iterations (>3x budget {max_iterations}) — check BCs for "
                "unconstrained rigid-body modes.",
            )
        )
    elif its > max_iterations:
        checks.append(_check("iteration_count", WARN, f"{its} iterations (budget {max_iterations})"))
    for e in re.findall(r"\[(?:IMPORT_)?ERROR\].*", log_text):
        checks.append(_check("solver_errors", FAIL, e[:300]))
    return _finish(checks, {"metrics": metrics})


def validate_facet_tags(
    tag_report: Any,
    expected_facet_tags: Sequence[int],
) -> Dict[str, Any]:
    """Assert that every expected boundary tag survived meshing — the gmsh
    full-plate meshes silently kept only the hole tag, the single blocker
    behind the FEM track's 0-for-6 record."""
    if isinstance(tag_report, (str, Path)):
        try:
            tag_report = json.loads(Path(tag_report).read_text())
        except Exception as e:
            return _finish([_check("tag_report", FAIL, f"unreadable tag report: {e}")])
    present = set()
    facet = tag_report.get("facet", {}) if isinstance(tag_report, dict) else {}
    for k in facet.keys():
        try:
            present.add(int(k))
        except (TypeError, ValueError):
            continue
    missing = [t for t in expected_facet_tags if t not in present]
    if missing:
        return _finish(
            [
                _check(
                    "facet_tags",
                    FAIL,
                    f"missing facet tags {missing}; present: {sorted(present)}. "
                    "Boundary conditions on missing tags will silently not apply.",
                )
            ]
        )
    return _finish([_check("facet_tags", PASS, f"all expected tags present: {sorted(present)}")])


def validate_hole_geometry(
    domain: Tuple[float, float, float, float],
    holes: Sequence[Tuple[float, float, float]],
) -> Dict[str, Any]:
    """Preflight for plate-with-holes geometry: (xmin, ymin, xmax, ymax) and
    (cx, cy, r) per hole. A hole entirely outside the domain was silently
    'cut' (i.e. not cut at all) in a real session and reported as success."""
    checks: List[Dict[str, str]] = []
    xmin, ymin, xmax, ymax = domain
    for i, (cx, cy, r) in enumerate(holes, 1):
        if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
            checks.append(
                _check(f"hole_{i}_position", FAIL, f"centre ({cx}, {cy}) lies OUTSIDE the domain")
            )
        elif (cx - r < xmin or cx + r > xmax or cy - r < ymin or cy + r > ymax):
            checks.append(
                _check(
                    f"hole_{i}_position",
                    WARN,
                    f"hole ({cx}, {cy}, r={r}) crosses the domain boundary — valid "
                    "only if this is an intended symmetry cut.",
                )
            )
        else:
            checks.append(_check(f"hole_{i}_position", PASS, "inside domain"))
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            (x1, y1, r1), (x2, y2, r2) = holes[i], holes[j]
            d = math.hypot(x2 - x1, y2 - y1)
            if d < r1 + r2:
                checks.append(
                    _check(
                        f"holes_{i+1}_{j+1}_overlap",
                        FAIL,
                        f"centre distance {d:.4g} < r1+r2 = {r1 + r2:.4g} — holes overlap",
                    )
                )
    if not checks:
        checks.append(_check("hole_geometry", PASS, "no holes specified"))
    return _finish(checks)


def validate_params_json(path: str) -> Dict[str, Any]:
    """params.json must at least be valid JSON with quoted keys (one real
    session shipped `{ E: 210e9, plane: stress }`, which json.load rejects)."""
    p = Path(path)
    if not p.exists():
        return _finish([_check("params_json", FAIL, f"missing: {path}")])
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        return _finish([_check("params_json", FAIL, f"invalid JSON: {e}")])
    if not isinstance(data, dict) or not data:
        return _finish([_check("params_json", FAIL, "not a non-empty JSON object")])
    return _finish(
        [_check("params_json", PASS, f"valid JSON with keys: {sorted(data.keys())[:12]}")]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate simulation outputs.")
    ap.add_argument("--lammps-log", help="LAMMPS log file")
    ap.add_argument("--script", help="LAMMPS input script (with --lammps-log)")
    ap.add_argument("--work-dir", help="run directory (file-existence checks)")
    ap.add_argument("--fem-log", help="FEniCSx solver log")
    ap.add_argument("--stress-dat", help="two-column strain/stress data file")
    ap.add_argument("--tag-report", help="mesh tag report JSON")
    ap.add_argument("--expected-tags", help="comma-separated expected facet tags")
    args = ap.parse_args(argv)

    rc = 0
    verdicts = []
    if args.lammps_log:
        script = Path(args.script).read_text() if args.script else ""
        verdicts.append(
            validate_lammps_run(script, Path(args.lammps_log).read_text(errors="replace"), args.work_dir)
        )
    if args.fem_log:
        verdicts.append(validate_fem_solve(Path(args.fem_log).read_text(errors="replace")))
    if args.stress_dat:
        strain, stress = [], []
        for line in Path(args.stress_dat).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.lstrip().startswith("#"):
                try:
                    strain.append(float(parts[0]))
                    stress.append(float(parts[-1]))
                except ValueError:
                    continue
        verdicts.append(validate_stress_strain(strain, stress))
    if args.tag_report:
        expected = [int(x) for x in (args.expected_tags or "").split(",") if x.strip()]
        verdicts.append(validate_facet_tags(args.tag_report, expected))
    for v in verdicts:
        print(format_verdict_markdown(v))
        print()
        if v["overall"] == FAIL:
            rc = 1
    if not verdicts:
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
