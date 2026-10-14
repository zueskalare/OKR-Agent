"""Generate the rectangular-cross-section twisted CNT-bundle CG model.

Companion to ../hex_cross_section, whose data file
(runs/jobs/session-atomagent-demo1/input/model1.data) was reverse-engineered to
fix every parameter below; the only intended difference is the bundle's
cross-sectional shape.

Reference bundle (hex): 92 chains on a hexagonal lattice, nearest-neighbour
centre spacing 12.0 A, chains twisted 2 full laps over L, 407 beads/chain at
dz = 2.46 A, box (-90,90)x(-90,90)x(0,1000).

This bundle (rect): the same 92 chains, same bead model, same twist, same box,
placed instead on a SQUARE lattice of the same 12.0 A nearest-neighbour spacing,
arranged as a 12x8 grid with the 4 corner sites removed -> exactly 92 chains in
a rectangular (aspect 1.5) outline.

Why 12x8-minus-corners: chain count is the control that matters here. The
tension script normalises the gage stress by the *box* volume, which is held
identical, so the measured axial stress scales with the number of load-bearing
chains; matching 92 exactly is what isolates the cross-section-shape effect.
A square lattice admits no 4-fold-symmetric 92-site patch (100-4=96, 100-12=88),
but a 2-fold-symmetric 12x8 rectangle minus its 4 corners hits 92 on the nose,
keeps every chain at >=2 in-plane neighbours, and is genuinely rectangular.
"""

import math
from pathlib import Path

A_LAT = 12.0  # square-lattice nearest-neighbour centre spacing [A]
NX, NY = 12, 8  # grid before corner removal
DZ = 2.46  # bead spacing along the tube axis [A]
N_BEAD = 407  # beads per chain
LZ = 1000.0  # box length along z [A]
TWIST_LAPS = 2.0  # full turns of the bundle over LZ
MASS = 240.22
XLO, XHI = -90.0, 90.0
YLO, YHI = -90.0, 90.0

OUT = Path(__file__).with_name("model_rect.data")


def lattice_sites():
    """12x8 square-lattice grid centred on the axis, 4 corner sites removed."""
    corners = {(0, 0), (NX - 1, 0), (0, NY - 1), (NX - 1, NY - 1)}
    return [
        (A_LAT * (i - (NX - 1) / 2), A_LAT * (j - (NY - 1) / 2))
        for j in range(NY)
        for i in range(NX)
        if (i, j) not in corners
    ]


def build():
    sites = lattice_sites()
    omega = math.radians(360.0 * TWIST_LAPS / LZ)  # twist rate [rad/A]

    atoms, bonds = [], []
    for mol, (x0, y0) in enumerate(sites, start=1):
        r = math.hypot(x0, y0)
        th0 = math.atan2(y0, x0)
        first = len(atoms) + 1
        for k in range(N_BEAD):
            z = k * DZ
            th = th0 + omega * z
            atoms.append((mol, r * math.cos(th), r * math.sin(th), z))
            if k:
                bonds.append((first + k - 1, first + k))
    return sites, atoms, bonds


def write(sites, atoms, bonds):
    out = [
        "LAMMPS data file: Rectangular CNT bundle CG (generated)",
        "",
        f"{len(atoms)} atoms",
        f"{len(bonds)} bonds",
        "1 atom types",
        "1 bond types",
        "",
        f"{XLO:.6f} {XHI:.6f} xlo xhi",
        f"{YLO:.6f} {YHI:.6f} ylo yhi",
        f"{0.0:.6f} {LZ:.6f} zlo zhi",
        "",
        "Masses",
        "",
        f"1 {MASS:.6f}",
        "",
        "Atoms # bond",
        "",
    ]
    out += [
        f"{i} {mol} 1 {x:.6f} {y:.6f} {z:.6f}"
        for i, (mol, x, y, z) in enumerate(atoms, start=1)
    ]
    out += ["", "Bonds", ""]
    out += [f"{i} 1 {a} {b}" for i, (a, b) in enumerate(bonds, start=1)]
    out.append("")
    OUT.write_text("\n".join(out))


if __name__ == "__main__":
    sites, atoms, bonds = build()
    write(sites, atoms, bonds)
    rmax = max(math.hypot(x, y) for x, y in sites)
    print(f"{OUT}: {len(sites)} chains, {len(atoms)} atoms, {len(bonds)} bonds")
    print(f"  grid {NX}x{NY} minus 4 corners, a = {A_LAT} A")
    print(f"  x extent +/-{max(abs(x) for x, _ in sites):.1f} A, "
          f"y extent +/-{max(abs(y) for _, y in sites):.1f} A")
    print(f"  max centre radius {rmax:.2f} A, box half-width 90.0 A "
          f"-> margin {90.0 - rmax:.2f} A")
