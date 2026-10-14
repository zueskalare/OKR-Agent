# Contributing

## Layout

- `AtomAgents/` — framework: agents, memory, MCP tool service, templates, utils.
- `backend/` — OpenAI-compatible FastAPI server (`openai_backend.py`).
- `benchmarks/` — campaign harness, baseline arms, hash-locked briefs and tool lock.
- `tests/` + `AtomAgents/tests/` — hermetic test suite (subprocess/`--help`
  checks, fake LLMs, real-failure fixtures under `tests/fixtures/`).
- `examples/`, `analysis/` — samples and post-hoc tooling; nothing in the
  framework imports them.
- `runs/` — runtime data, gitignored.

## Development

```bash
conda env create -f environment.yml -p ./venv && ./venv/bin/pip install -e .[dev]
make test            # pytest over tests/ + AtomAgents/tests/ — no services, no keys
make lint            # ruff
make briefs-check    # README demo blocks == benchmarks/briefs (hash-locked)
```

Python 3.11, 4-space indentation, line length ~100, type hints and explicit
returns; modules/functions `snake_case`, classes `PascalCase`. Comments explain
*why* (most of the harness comments record the failure that motivated a line —
keep that habit).

## Tests

- Keep them hermetic: no network, no API keys, no running services. Tests that
  need a real model must be opt-in behind an environment variable and skip by
  default.
- `tests/test_release_hygiene.py` fails on credential-shaped strings, personal
  absolute paths and notebooks with outputs. Strip notebooks before committing
  (`python -c "import nbformat,sys; ..."` or `nbstripout`).
- Name new files `tests/test_<feature>.py`; prefer quick (<5 s) tests.

## Things that must not change casually

These are inputs of the published benchmark; their hashes are recorded in
every run manifest and the dataset. Changing them makes new runs incomparable:

- `AtomAgents/templates/prompts/*.yaml`, `AtomAgents/templates/fenicsx/*.yaml`
- `benchmarks/briefs/*.txt`, `benchmarks/briefs/briefs.lock.json`, and the three
  `### … demo` blocks in `README.md` they are extracted from
- `benchmarks/tools.lock.json`, and the names / docstrings / schemas of the
  tools in `AtomAgents/mcp_service/mcp_service.py` it fingerprints
- `benchmarks/prompts/baseline_{md,fem}.md`

If you do mean to change the experiment, regenerate the lock files
(`python -m benchmarks.extract_briefs`, `python -m benchmarks.toolbind --write`)
and say so in the commit message.

## Commits and pull requests

Short, imperative, scoped subjects ("Fix runner --prefix default"). A PR needs
a description, rationale, how it was tested (commands, expected output), and
a note on anything that touches the hash-locked inputs above. Never commit
`.env`, `runs/`, session databases or notebook outputs.
