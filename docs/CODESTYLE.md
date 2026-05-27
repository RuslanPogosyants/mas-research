# Python Code Style

Checked via `ruff` + `mypy --strict` + `pre-commit`. Comply when writing, do not fix post-hoc.

## Language
- **English only** in comments, docstrings, identifiers. No Cyrillic in `*.py`.
- **No `# noqa`, no `# type: ignore`** suppressions — fix correctly, do not silence the linter.

## Types
- `T | None` / `T | S` — **not** `Optional`, **not** `Union`.
- `Final[TYPE]` for constants.
- Type hints required on every function and method (including `-> None`).

## Imports
- **Absolute** only, no relative (`from .foo`, `from ..bar`).
- No `from module import *`.
- One module per line.
- Order: stdlib `import` → third-party `from` → local `from` → `if TYPE_CHECKING:`.

## Size limits
- Line: **120** chars.
- Nesting: **5** levels.
- Function: **5** arguments, **20** lines.

## Control flow
- `is None` / `is not None` — **not** `== None`.
- `zip(..., strict=True)` — always.
- Concrete exceptions, not bare `Exception`.
- No `lambda` (except `key=` in `sorted`/`min`/`max`).
- No `eval`, `exec`.
- No `print` for debugging — use `logger`.

## Naming
- `snake_case` for variables, functions, modules.
- `PascalCase` for classes.
- `_name` (single underscore) for protected.
- Boolean: prefix `is_`, `has_`, `are_`, `can_`, `should_`.
- Descriptive names — not `x`, `tmp`, `data`, `obj`.

## Docstrings and formatting
- Google-style docstrings.
- `ruff format` before commit (120 cols).

## Async
- **Async-first** for I/O: DB, HTTP, files, Redis, RabbitMQ.
- Do not mix sync and async in one function.

---

# Soft Preferences (new code)

Preferences, not blockers. Apply when writing **new** code/blocks. For legacy edits minimum-change wins. Violation is a discussion topic, not a bug.

**Do not over-do it.**

## Structure
- **Function > class** if there is no stateful mutation. Do not class up a "bag of pure functions with a config dict".
- **Early return** instead of nested `if/else` — keeps nesting ≤ 5.

## Data containers — choose by boundary
- **Pydantic `BaseModel`** — at the service boundary: FastAPI request/response, external payload validation, serialization.
- **`@dataclass(slots=True)`** — internal value object without validation, without complex inheritance. `frozen=True` only if immutability is principled.
- **`TypedDict`** — typing an already-existing dict (raw JSON shape) without runtime overhead.
- NOT Pydantic for every internal struct — validation overhead and extra dependency in pure modules.

## Polymorphism
- **`typing.Protocol`** — duck-typing, test doubles, structural typing.
- **`abc.ABC`** — shared concrete behaviour (`super().__init__()`, mixins), nominal inheritance check.
- Different jobs — choose by semantics, not "Protocol is always better".

## Modern syntax
- `match/case` — for **structural** payload decomposition (dict shape, Enum + value, tuple). NOT for counting `isinstance` branches — that is `singledispatch`.
- `typing.Literal[...]` — finite set of strings/numbers in signatures.
- `typing.TypeGuard` for custom narrowing functions; `assert isinstance(x, T)` for inline narrowing.

## Async
- `asyncio.TaskGroup` — preferred grouping (3.11+).
- `asyncio.timeout()` — context-aware timeout (3.11+).
- `asyncio.gather(*tasks, return_exceptions=True)` — collect all exceptions when needed.
- `asyncio.to_thread(sync_fn, ...)` — CPU-bound or sync-only library inside async.

## Stdlib reach (with caveats)
- `functools.cache` — **only** for pure free functions. NOT on instance methods (cache holds `self` alive → leak). NOT on async functions (does not work; use `async-lru`).
- `functools.singledispatch` — type-based dispatch instead of `isinstance` chains.
- `itertools.pairwise` — window of 2. `itertools.groupby` — requires **sorted** input, otherwise silent bug.
- `contextlib.ExitStack` / `closing` — dynamic resource lifecycles.

## Exceptions
- **No catch-log-reraise.** `except E: log.error(e); raise` is trash, the traceback is already there. Catch only when there is something to **do**: fallback, recovery, add context to the message.

## Docstrings
- **Public API** (FastAPI endpoints, library exports, public service methods) — required, Google-style.
- **Internal helpers** — only if there is a non-trivial "why" / invariant / edge case. Do NOT restate the signature: `"""Returns the user."""` over `def get_user() -> User` is noise.

## Anti-cleverness
- Readability > brevity. Do not use `match` / `Protocol` / walrus if a simple `if` / class / inline is shorter and clearer.
- If applying an idiom needs a comment "why I did this" — wrong idiom chosen.
