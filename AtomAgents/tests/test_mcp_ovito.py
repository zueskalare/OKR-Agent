import base64
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("ovito", reason="OVITO not installed")

from mcp_service.tools import mcp_render_md_model  # noqa: E402


def test_ovito_render_glucose(tmp_path):
    xyz_path = os.path.join(os.path.dirname(__file__), "glucose.xyz")
    save_path = tmp_path / "glucose_ovito.png"
    ok, payload, n_particles = mcp_render_md_model(
        xyz_path,
        orientation=(0, 0, -1),
        cam_pos=(0, 0, 10),
        save_dir=str(save_path),
    )
    assert ok, payload
    assert n_particles > 0
    img_bytes = base64.b64decode(payload)
    assert img_bytes.startswith(b"\x89PNG")
    assert save_path.exists() and save_path.stat().st_size > 1000
