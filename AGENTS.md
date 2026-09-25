# AGENTS.md

**Parax** is the parameter and constraint layer for JAX: array-like variables with
metadata (`Constrained`, `Random`, `Derived`, ...), bijective constraints and
distributions built on [distreqx](https://github.com/lockwo/distreqx), and tree tools
for unwrapping parameterised PyTrees.

## Breaking changes are fine

Pre-1.0. No backwards-compatibility shims, deprecation aliases or legacy code paths
unless asked. Prefer the clean design, within the issue's seam.

## Seams

Each issue names one seam: a file, a set of files, or a folder. If the work needs a
change outside it, stop and ask.

`CONTEXT.md` and ADRs (architecture decision records, `docs/adr/`) are edited only by
the reviewer, in their own commit. If you need a domain term that isn't in the issue,
an ADR or `CONTEXT.md`, that's a design decision: stop and ask.

Start every PR body with `## Cross-seam`, listing each file changed outside the issue's
seam with the ADR or issue line that allows it, or `DECISION NEEDED`. Write `none` if
there are none.

## Commands

```bash
.venv/bin/python -m pytest tests/test_x.py        # the tests covering a change, while working
.venv/bin/python -m pytest tests/                 # full suite, once before committing
.venv/bin/python -m pytest --codeblocks docs/     # the code blocks in the docs, which CI also runs
```

No linter or formatter is configured. Match the style of surrounding code.

## Commits

Do not add yourself as an author. No `Co-Authored-By` trailer, no session link, no
tool attribution.

## Source layout

Real code lives in `parax/`. `parax/_vendor/` holds copies of distreqx classes that
`parax.bijectors` and `parax.distributions` fall back to when the installed distreqx
lacks them; import through those two modules.

## Documentation

Google-style docstrings (`Args:`, `Returns:`), rendered by mkdocs. The site's pages are
listed in `mkdocs.yml`'s `nav`, and every code block under `docs/` runs in CI.

## Issues and docs

### Issue tracker

Issues live as GitHub issues on `gvcallen/parax`, managed with the `gh` CLI. See
`docs/agents/issue-tracker.md`.

### Triage labels

See `docs/agents/triage-labels.md`.

### Domain docs

One `CONTEXT.md` and one `docs/adr/` at the repo root; either may not exist yet. See
`docs/agents/domain.md`.
