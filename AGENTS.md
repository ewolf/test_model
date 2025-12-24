# Repository Guidelines

## Project Structure & Module Organization
- Keep core code under `src/` with clear domain-driven submodules (e.g., `src/data`, `src/models`, `src/training`). Place reusable utilities in `src/utils/`.
- Store experiments and exploratory work in `notebooks/`; keep large artifacts out of git.
- Commit raw datasets to `data/` only when they are small and license-safe; otherwise reference download scripts in `data/README.md`.
- Put tests beside the code in `tests/` mirroring package layout (e.g., `src/models/train.py` ↔ `tests/models/test_train.py`).
- Version intermediate model weights and metrics in `models/` with lightweight checkpoints; prefer pushing heavyweight artifacts to external storage.

## Build, Test, and Development Commands
- Use Python 3.11+ and a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies once a `requirements.txt` or `pyproject.toml` is present: `pip install -r requirements.txt`.
- Run the suite locally: `python -m pytest` from the repo root; add `-k <pattern>` to target modules.
- Format and lint before committing: `python -m black src tests` and `python -m ruff check src tests` if those tools are configured.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, 88–100 char lines, meaningful docstrings, and type hints on public functions.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `CONSTANT_CASE` for immutable module-level values.
- Keep modules focused; avoid broad "helpers" names—prefer descriptive files like `data_loader.py` or `losses.py`.

## Testing Guidelines
- Write unit tests for every new behavior; place shared fixtures in `tests/conftest.py`.
- Name tests `test_<subject>_<expectation>` and include edge cases (shape/NaN handling, device switching, seeding).
- Track coverage with `pytest --cov=src --cov-report=term-missing`; keep coverage steady or improving.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., `add dataset loader for mnist`) and group related changes together.
- Reference issues in PR descriptions, summarize behavior changes, and note any data/model version impacts.
- Include screenshots or logs for notable training runs or metrics regressions; call out breaking interface changes explicitly.

## Security & Configuration Tips
- Keep secrets out of the repo; use `.env` (ignored by `.gitignore`) for local credentials and document required keys in `.env.example`.
- Validate any external datasets or checkpoints for licenses and integrity; store hashes when possible.
