# Contributing

## Code style

See [`docs/CODESTYLE.md`](docs/CODESTYLE.md). Enforced by `ruff` + `mypy --strict` + `pre-commit`.

Install hooks once:

```powershell
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Commit message format

Enforced by `commit-msg` hook (`scripts/check_commit_msg.py`).

```
<type>: <description>
```

- `<type>` ∈ {`feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`}
- `<description>` — 6 to 9 English words, imperative mood, no trailing period
- No scope in parentheses
- One line subject. Body optional, separated by blank line.

### Examples

✅ `feat: add pydantic message model for fipa-acl protocol`
✅ `test: add contract tests for message shape validation`
✅ `fix: dispatch loop deadline check off by one`
✅ `refactor: extract retry policy into separate module`
✅ `docs: add readme section for quickstart and architecture`
✅ `chore: bump faster-whisper to 1.0.3 for cuda fix`
✅ `ci: add coverage upload step to test workflow`

❌ `feat(core): add pydantic message model` — scope, too few words
❌ `chore: setup` — too few words
❌ `Add transcriber` — no type prefix
❌ Too many words: keep subject within 6 to 9 words.

## Author

All commits use a single author: `Ruslan Pogosyants <ruslanpogosyants9594@gmail.com>`. No co-authored trailers.
