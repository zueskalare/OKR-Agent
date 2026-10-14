"""
AtomAgents MCP Service using FastMCP (Model Context Protocol Python SDK).
"""

# import sys
# from pathlib import Path

# Include local MCP Python SDK (resources/python-sdk/src)
# sys.path.insert(0, str(Path(__file__).parent.parent / "resources/python-sdk/src"))

from mcp.server.fastmcp import FastMCP

try:
    # Package-style import (uvicorn AtomAgents.mcp_service.mcp_service:app)
    from .tools import (
        _guard_workspace_path,
        _sandbox_status,
        _tool_root,
        mcp_file_reader,
        mcp_code_runner,
        mcp_web_search,
        mcp_render_md_model,
        mcp_lammps_runner,
        mcp_fenics_mesh_visualize,
        mcp_fenics_poisson_visualize,
        mcp_fenics_elasticity_visualize,
        mcp_bash_runner,
        mcp_write_file,
        mcp_download_file,
    )
except ImportError:
    # Standalone deployment with cwd = mcp_service/
    from tools import (
        _guard_workspace_path,
        _sandbox_status,
        _tool_root,
        mcp_file_reader,
        mcp_code_runner,
        mcp_web_search,
        mcp_render_md_model,
        mcp_lammps_runner,
        mcp_fenics_mesh_visualize,
        mcp_fenics_poisson_visualize,
        mcp_fenics_elasticity_visualize,
        mcp_bash_runner,
        mcp_write_file,
        mcp_download_file,
    )
import os
from json import dumps
from pathlib import Path

# --- Startup gate (S2): never serve tools without a jail -------------------
# An import-time raise aborts uvicorn before it binds and covers every launch
# form (module/package/`python -m` invocations, the Dockerfile CMD, compose).
# It lives here and not in tools.py because tests import tools.py at module
# scope; nothing in the suite imports this module.
_ROOT = _tool_root()
if _ROOT is None and os.environ.get(
    "ATOMAGENTS_ALLOW_UNJAILED", ""
).strip().lower() not in ("1", "true", "yes", "on"):
    raise RuntimeError(
        "ATOMAGENTS_TOOL_ROOT is not set to an existing directory — refusing "
        "to start the MCP tool service.\n"
        "It is the write/exec jail root for every agent-issued file, script "
        "and shell command; without it code_runner / bash_run / lammps_run "
        "would operate on the whole filesystem.\n"
        "  make mcp                      # from the repo root (recommended)\n"
        '  ATOMAGENTS_TOOL_ROOT="$PWD/runs/jobs" python -m uvicorn '
        "AtomAgents.mcp_service.mcp_service:app --host 127.0.0.1 --port 9999\n"
        "Set ATOMAGENTS_ALLOW_UNJAILED=1 to start without a jail (never with "
        "a real model attached). See README 'Security (tool sandbox)'."
    )
# flush=True: stdout is block-buffered when redirected (nohup, docker logs,
# systemd), and an operator must see the posture line at startup, not when
# the buffer eventually fills.
print(
    f"[MCP] tool jail: {_ROOT or 'DISABLED (ATOMAGENTS_ALLOW_UNJAILED=1)'} | "
    f"child sandbox: {_sandbox_status()}",
    flush=True,
)

# Initialize FastMCP server
mcp = FastMCP("AtomAgents MCP Service")


def _fmt(v) -> str:
    """Format a numeric metric; non-numeric/absent values render as 'n/a'."""
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return "n/a"

# --------- Auto-build documentation RAG index on service start ---------
# The Fenicsx documentation PDF is expected under:
#   AtomAgents/mcp_service/faiss_rag/resources
# We build the index into:
#   AtomAgents/mcp_service/faiss_rag/index

### NOTE: The index build can be slow (minutes) and requires FAISS and other packages.
### ! This is deprecated for now, we will use travily to prebuild the index and check it in.
# DOC_RESOURCES_DIR = str(Path(__file__).parent / "faiss_rag" / "resources" / "fenicsx")
# DOC_INDEX_DIR = str(
#     Path(__file__).parent / "faiss_rag" / "resources" / "fenicsx" / "index"
# )

# # Build (or rebuild) the index when the service module is imported
# try:
#     print(
#         f"[MCP] Building documentation RAG index from {DOC_RESOURCES_DIR} -> {DOC_INDEX_DIR}"
#     )
#     # Prefer direct module to avoid JSON encode/decode
#     try:
#         from faiss_rag import build_index as _build_index  # type: ignore

#         _ = _build_index(
#             DOC_RESOURCES_DIR, DOC_INDEX_DIR, model="auto", chunk_size=1000, overlap=200
#         )
#     except Exception:
#         print("[MCP] Documentation index build failed.")
#     print("[MCP] Documentation index ready.")
# except Exception as e:
#     print(f"[MCP] Documentation index build failed: {e}")


@mcp.tool()
def file_reader(file: str, offset: int = 0, limit_bytes: int = 0) -> str:
    """
    Read a file's content. Any text file works (content-sniffed, so .data/.in/
    .xdmf/.lammpstrj are fine); PNG/JPG/PDF return multimodal parts. Large text
    is windowed head+tail — page with offset/limit_bytes. Directories return a
    listing.
    Args:
        file (str): file path
        offset (int): byte offset into text files (default 0)
        limit_bytes (int): max bytes of text to return (default: tool cap)
    Returns:
        str: JSON content parts, or {"error": ...} on failure.
    """
    print(f"file_reader called with file: {file}")
    success, res = mcp_file_reader(file, offset=offset, limit_bytes=limit_bytes)
    # res is already a JSON string; re-encoding it broke image/PDF unwrapping.
    return res


@mcp.tool()
def code_runner(code: str, cwd: str = "", timeout_sec: int = 300) -> str:  # dumped string
    """Run a Python script in a sandboxed subprocess. cwd is REQUIRED and must
    be an absolute path inside the job workspace; the script runs with a
    scrubbed environment (network access remains). No script arguments — it
    must be ready to run as-is.
    Args:
        code (str): Python code to execute.
        cwd (str): REQUIRED absolute working directory (the job workspace).
        timeout_sec (int): max seconds before the run is killed (default 300).
    Returns:
        str: JSON string with 'stdout', 'stderr', 'returncode' fields."""
    print(f"code_runner called with cwd={cwd!r} code: {code[:500]}")
    return mcp_code_runner(code, cwd=cwd or None, timeout_sec=timeout_sec)


@mcp.tool()
def write_file(path: str, content: str, mode: str = "w") -> str:
    """Write content to a file verbatim (UTF-8). Use this to author scripts,
    input decks, and configs — it avoids every shell-quoting/heredoc pitfall.
    Args:
        path (str): absolute destination path inside the job workspace.
        content (str): exact file content.
        mode (str): "w" overwrite (default) or "a" append.
    Returns:
        str: JSON with 'path' and 'bytes_written', or {"error": ...}."""
    print(f"write_file called with path: {path} ({len(content)} chars, mode={mode})")
    return mcp_write_file(path, content, mode)


@mcp.tool()
def download_file(url: str, dest: str, expect: str = "auto") -> str:
    """Download a URL to a file with payload validation — refuses to save an
    HTML landing/error page as a data file and reports size + sha256.
    Args:
        url (str): http(s) URL.
        dest (str): absolute destination path.
        expect (str): "text", "binary", or "auto" (default).
    Returns:
        str: JSON with 'path', 'bytes', 'sha256', 'content_type', or {"error": ...}."""
    print(f"download_file called with url: {url} -> {dest}")
    return mcp_download_file(url, dest, expect)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Use DuckDuckGo search to fetch relevant web results for a query.
    Args:
        query (str): Search keywords.
        max_results (int): Max results to return (default 5).
    Returns:
        str: JSON string with a list of search results, each containing 'title', 'href', and optional 'body'.
    Usage: For agents/tools needing current info, web links, or knowledge discovery tasks.
    """
    print(f"web_search called with query: {query}, max_results: {max_results}")
    return mcp_web_search(query, max_results)


@mcp.tool()
def bash_run(cmd: str, cwd: str = "", timeout_sec: int = 60) -> str:
    """Run a Bash command inside the job workspace. cwd is REQUIRED (absolute
    path inside the workspace); the command runs with a scrubbed environment
    and git/dangerous commands are denied. Shell state (env vars, cd) does
    NOT persist between calls — use absolute paths. For writing files use the
    write_file tool, not heredocs.
    Args:
        cmd: Command string interpreted by bash.
        cwd: REQUIRED absolute working directory (the job workspace).
        timeout_sec: Max seconds before timing out (default 60).
    Returns:
        JSON string with 'stdout', 'stderr', 'returncode', and 'cwd'.
    """
    print(f"bash_run called with cmd: {cmd}, cwd: {cwd}, timeout_sec: {timeout_sec}")
    return mcp_bash_runner(cmd, cwd or None, timeout_sec)


@mcp.tool()
def md_visualize(
    file: str,
    save_file: str,
    orientation: str = "-1,-1,-1",
    cam_pos: str = "1000,1000,1000",
) -> str:
    """Visualize structural/MD file and return base64 PNG image.
    When determine the orientation and camera position, consider the cell size and the center of the cell. to avoid the camera position is too close to the cell, to avoid the blank or blocked image.
    Args:
        file (str): Path to the structural/MD file.
        save_file (str): Path to save the visualization image. png format.
        orientation (str): Camera orientation as "x,y,z" vector (example "-1,-1,-1").
        cam_pos (str): Camera position as "x,y,z" vector (example "1000,1000,1000").
    Returns:
        str: JSON string with a single image part containing the base64 PNG data.
    Usage: For agents/tools needing to visualize MD structures or simulations.
    """
    print(
        f"md_visualize called with file: {file}, orientation: {orientation}, cam_pos: {cam_pos}"
    )
    try:
        ori = tuple(float(x) for x in orientation.split(","))
        cam = tuple(float(x) for x in cam_pos.split(","))
    except ValueError as e:
        return dumps({"Error": f"orientation/cam_pos must be comma-separated numbers: {e}"})
    # get the bytes of the file
    if not Path(file).exists():
        return dumps({"Error": "File does not exist."})
    guard = _guard_workspace_path(save_file)
    if guard:
        return dumps({"Error": guard})
    if not Path(save_file).parent.exists():
        # create the directory
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
    success, res, num = mcp_render_md_model(
        file, orientation=ori, cam_pos=cam, save_dir=save_file
    )
    if success:
        parts = [
            {"type": "text", "text": f"Visualized {num} particles."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{res}",
                },
            },
        ]
        # print("md_visualize parts:", parts)
        return dumps(parts)
    else:
        return dumps({"Error": res})


@mcp.tool()
def lammps_run(infile_string: str, work_dir: str, infile: str) -> str:
    """Run a LAMMPS simulation given the INPUT SCRIPT CONTENT (not a path) and
    return the parsed result. The run executes with cwd=work_dir, so relative
    paths in the script resolve inside it. The result includes the ERROR/WARNING
    lines and the log tail — inspect them after every run; there is no live
    monitoring during execution (the call blocks until LAMMPS exits).
    Do not use LAMMPS variables (var/${...}) in the script.
    Args:
        infile_string (str): LAMMPS input script as a string.
        work_dir (str): Absolute directory to run the simulation in.
        infile (str): Name of the input file to create in the work directory.
    Returns:
        str: JSON with 'returncode', 'errors', 'warnings', 'output' (log tail),
        'log_file', 'stderr', 'error' (summary or null).
    Usage: For agents/tools needing to run LAMMPS simulations and analyze results.
    """
    print(
        f"lammps_run called with work_dir: {work_dir}, infile: {infile}, "
        f"script: {infile_string[:300]}..."
    )
    return mcp_lammps_runner(infile_string, work_dir, infile)


# # --- FEniCS FEM tools ---
@mcp.tool()
def fem_mesh_visualize(
    save_file: str,
    domain: str = "unit_square",
    nx: int = 32,
    ny: int = 32,
    nz: int | None = None,
) -> str:
    """Create a FEniCS mesh and return a base64 PNG.
    Args:
        save_file: Path to save the PNG.
        domain: 'unit_square' or 'unit_cube'.
        nx, ny, nz: Mesh resolution (nz only for unit_cube).
    Returns:
        JSON string with an image part or error.
    """
    print(
        f"fem_mesh_visualize(save_file={save_file}, domain={domain}, nx={nx}, ny={ny}, nz={nz})"
    )
    guard = _guard_workspace_path(save_file)
    if guard:
        return dumps({"Error": guard})
    ok, res = mcp_fenics_mesh_visualize(save_file, domain, nx, ny, nz)
    if ok:
        parts = [
            {
                "type": "text",
                "text": f"Mesh {domain} generated with ({nx},{ny}{',' + str(nz) if nz else ''}).",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{res}"},
            },
        ]
        return dumps(parts)
    else:
        return dumps({"Error": res})


@mcp.tool()
def fem_poisson_solve_visualize(
    save_file: str,
    domain: str = "unit_square",
    nx: int = 32,
    ny: int = 32,
    f_value: float = 1.0,
    dirichlet_value: float = 0.0,
    nz: int | None = None,
) -> str:
    """Solve -Δu=f with Dirichlet BC using FEniCS and return a base64 PNG of the solution.
    Args:
        save_file: Path to save the PNG.
        domain: 'unit_square' or 'unit_cube'.
        nx, ny, nz: Mesh resolution (nz only for unit_cube).
        f_value: Constant RHS value.
        dirichlet_value: Constant boundary value.
    Returns:
        JSON string of parts with image and text summary, or error.
    """
    print(
        "fem_poisson_solve_visualize(",
        f"save_file={save_file}, domain={domain}, nx={nx}, ny={ny}, nz={nz}, f={f_value}, uD={dirichlet_value}",
        ")",
    )
    guard = _guard_workspace_path(save_file)
    if guard:
        return dumps({"Error": guard})
    ok, img_or_err, metrics = mcp_fenics_poisson_visualize(
        save_file, domain, nx, ny, f_value, dirichlet_value, nz
    )
    if ok:
        summary = (
            f"Solved Poisson. L2={_fmt(metrics.get('l2_norm'))}, "
            f"Energy={_fmt(metrics.get('grad_energy'))}"
        )
        parts = [
            {"type": "text", "text": summary},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_or_err}"},
            },
        ]
        return dumps(parts)
    else:
        return dumps({"Error": img_or_err})


@mcp.tool()
def fem_elasticity_solve_visualize(
    save_file: str,
    domain: str = "unit_square",
    nx: int = 32,
    ny: int = 32,
    nz: int | None = None,
    E: float = 210e9,
    nu: float = 0.3,
    traction: str = "0.0,0.0,0.0",
    plane_stress: bool = False,
    field: str = "u_mag",
) -> str:
    """Solve linear elasticity with Dirichlet u=0 on left and traction on right. Visualize displacement magnitude or von Mises stress.traction: comma-separated components, length matches domain dimension.
    Args:
        save_file: Path to save the PNG.
        domain: 'unit_square' or 'unit_cube'.
        nx, ny, nz: Mesh resolution (nz only for unit_cube).
        E: Young's modulus.
        nu: Poisson ratio.
        traction: Comma-separated traction vector components.
        plane_stress: Whether to use plane stress (2D only).
        field: 'u_mag' or 'von_mises' to visualize.
    Returns:
        JSON string of parts with image and text summary, or error.
    """
    print(
        "fem_elasticity_solve_visualize(",
        f"save_file={save_file}, domain={domain}, nx={nx}, ny={ny}, nz={nz}, E={E}, nu={nu}, traction={traction}, plane_stress={plane_stress}, field={field}",
        ")",
    )
    try:
        tr = tuple(float(x) for x in traction.split(","))
    except ValueError as e:
        return dumps({"Error": f"traction must be comma-separated numbers: {e}"})
    guard = _guard_workspace_path(save_file)
    if guard:
        return dumps({"Error": guard})
    ok, img_or_err, metrics = mcp_fenics_elasticity_visualize(
        save_file,
        domain,
        nx,
        ny,
        nz,
        E,
        nu,
        tr,
        plane_stress,
        field,
    )
    if ok:
        summary = (
            f"Elasticity {field}. uL2={_fmt(metrics.get('u_mag_L2'))}, "
            f"vmL2={_fmt(metrics.get('von_mises_L2'))}"
        )
        parts = [
            {"type": "text", "text": summary},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_or_err}"},
            },
        ]
        return dumps(parts)
    else:
        return dumps({"Error": img_or_err})


## ! This is deprecated for now, we will use travily to prebuild the index and check it in.
# # ------- Documentation RAG endpoint (prebuilt index) --------
# @mcp.tool()
# def mcp_fenicsx_doc_query(query: str, top_k: int = 5) -> str:
#     """Query the prebuilt Fenicsx documentation index and return top_k results for reference of code.
#     Args:
#         query (str): Query string.
#         top_k (int): Number of top results to return (default 5).
#     Returns:
#         str: JSON string with 'hits' list containing 'score', 'source', 'chunk', and 'text' for each hit.
#     Usage: For agents/tools needing to reference the Fenicsx documentation.
#     """
#     print(f"mcp_doc_query(top_k={top_k}) :: {query[:80]}...")
#     print(f"Documentation index dir: {DOC_INDEX_DIR}")
#     # Prefer direct module to avoid extra JSON encoding/decoding
#     from faiss_rag import query as _query  # type: ignore

#     res = _query(DOC_INDEX_DIR, query, top_k=top_k)
#     res = res["hits"]
#     res = [f"chunk {i}: {h['text']}" for i, h in enumerate(res)]
#     res = "\n\n".join(res)
#     return res


# # Documentation RAG with Tavily for fenicsx
# Note: You need to set the environment variable TAVILY_API_KEY with your Tavily API
# NB: names must not start with "mcp_" — Anthropic's API classifies requests
# carrying an mcp_-prefixed tool as a third-party app and rejects them with an
# "extra usage" 400 when routed through a subscription-auth proxy.
@mcp.tool()
def fenicsx_doc_query(query: str) -> str:
    """Query the prebuilt Fenicsx documentation index and return top_k results for reference of code.
    Args:
        query (str): Instructions for searching the documentation about fenicsx coding.
    Returns:
        str: JSON dump with 'results' from documentation search.
    Usage: For agents/tools needing to reference the Fenicsx documentation.
    """

    # url for fenicsx documentation https://docs.fenicsproject.org/dolfinx/v0.9.0/python/
    from tavily import TavilyClient

    client = TavilyClient()
    response = client.crawl(
        "https://docs.fenicsproject.org/dolfinx/v0.9.0/python/",
        instructions=query,
        timeout=600,
        limit=10,
    )
    return dumps(response)


# Documentation RAG with Tavily for lammps
@mcp.tool()
def lammps_doc_query(query: str) -> str:
    """Query the prebuilt LAMMPS documentation index and return top_k results for reference of code.
    Args:
        query (str): Instructions for searching the documentation about lammps coding.
    Returns:
        str: JSON dump with 'results' from documentation search.
    Usage: For agents/tools needing to reference the LAMMPS documentation.
    """

    # url for lammps documentation https://docs.lammps.org/
    from tavily import TavilyClient

    client = TavilyClient()
    response = client.crawl(
        "https://docs.lammps.org/", instructions=query, timeout=600, limit=10
    )
    return dumps(response)


# Expose ASGI app for Uvicorn
app = mcp.streamable_http_app()
