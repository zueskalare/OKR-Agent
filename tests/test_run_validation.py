"""Validation gates, replayed against REAL failure artifacts from the job
archive (checked in as fixtures)."""

from pathlib import Path

from AtomAgents.utils.run_validation import (
    validate_fem_solve,
    validate_facet_tags,
    validate_hole_geometry,
    validate_lammps_run,
    validate_stress_strain,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _statuses(verdict):
    return {c["name"]: c["status"] for c in verdict["checks"]}


def test_no_integrator_run_fails():
    log = (FIXTURES / "run_prod_no_integrator.log").read_text()
    v = validate_lammps_run("", log)
    s = _statuses(v)
    assert v["overall"] == "FAIL"
    assert s.get("time_integrator") == "FAIL"
    assert s.get("frozen_dynamics") == "FAIL"


def test_static_integrator_check():
    script = "fix pull all deform 1 z erate 0.001\nrun 1000\n"
    v = validate_lammps_run(script, "")
    assert _statuses(v).get("time_integrator") == "FAIL"

    script_ok = "fix mynvt all nvt temp 300 300 0.1\nfix pull all deform 1 z erate 0.001\nrun 1000\n"
    v2 = validate_lammps_run(script_ok, "")
    assert _statuses(v2).get("time_integrator") == "PASS"


def test_healthy_run_passes():
    log = (
        "   Step          Temp          PotEng         KinEng         TotEng         Press          Volume\n"
        "     0   300.1   -8623.1   175.2   -8447.9   -12.3   32400000\n"
        "  1000   301.2   -8620.5   175.9   -8444.6   -14.1   32400000\n"
        "  2000   299.4   -8619.8   174.8   -8445.0   -13.3   32400000\n"
        "  3000   300.7   -8618.2   175.5   -8442.7   -12.9   32400000\n"
        "  4000   298.9   -8617.9   174.5   -8443.4   -13.7   32400000\n"
        "Loop time of 12.3\n"
    )
    v = validate_lammps_run("fix a all nvt temp 300 300 0.1\nrun 4000\n", log)
    assert v["overall"] == "PASS", v


def test_interleaved_stress_strain_fails():
    strain, stress = [], []
    for line in (FIXTURES / "stress_interleaved.dat").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.lstrip().startswith("#"):
            try:
                strain.append(float(parts[0]))
                stress.append(float(parts[-1]))
            except ValueError:
                continue
    v = validate_stress_strain(strain, stress)
    assert _statuses(v).get("duplicate_strains") == "FAIL"


def test_singular_solver_log_fails_and_metrics_come_from_log():
    v = validate_fem_solve((FIXTURES / "solver_singular.log").read_text())
    assert v["overall"] == "FAIL"
    assert v["metrics"]["iterations"] == 745
    assert v["metrics"]["reason"] == 3
    assert v["metrics"]["ndofs"] == 30810


def test_healthy_solve_passes():
    v = validate_fem_solve("[INFO] PETSc reason=2 its=17 rfinal=7.4e-12 elapsed=0.25s ndofs=3170")
    assert v["overall"] == "PASS"


def test_missing_facet_tags_fail():
    v = validate_facet_tags(FIXTURES / "tag_report_single.json", [2, 3, 4, 5])
    assert v["overall"] == "FAIL"
    assert "missing facet tags [2, 3, 4]" in v["checks"][0]["detail"]


def test_hole_geometry_preflight():
    # The real 3holes spec: domain x in [0, 0.15], hole centre at x=0.225
    v = validate_hole_geometry(
        (0, 0, 0.15, 0.10),
        [(0.075, 0.10, 0.05), (0.15, 0.10, 0.05), (0.225, 0.10, 0.05)],
    )
    assert v["overall"] == "FAIL"
    # A symmetry-cut hole at the corner is WARN, not FAIL
    v2 = validate_hole_geometry((0, 0, 1.0, 1.0), [(0.0, 0.0, 0.1)])
    assert v2["overall"] == "WARN"
