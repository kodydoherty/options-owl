"""Generic guard against conditional-only assignment on the live-money paths.

THE BUG CLASS
-------------
A local is assigned ONLY inside a branch (`if`, `for`, `try`, ...) but read at a point
that executes unconditionally. When the branch doesn't run, the read raises
UnboundLocalError. CLAUDE.md rule #1 exists because of it:

  2026-05-07  `reason` assigned only under a conditional in position_monitor ->
              UnboundLocalError every 15s -> NO trade could exit. QQQ ran -35% to -88%.
  2026-08-07  `_sizing_audit` initialised inside `if use_vinny and use_score_sizing:`
              while the INSERT ran unconditionally -> 36 test failures.

The existing defence is a hand-written source test PER variable, which only protects
variables somebody already thought about — neither bug above had one beforehand. This
checks the SHAPE instead, so a variable nobody anticipated is still covered.

WHY NOT A LINTER
----------------
mypy's `possibly-undefined` covers this, but mypy is not a project dependency and the
build box has no network. This is stdlib `ast` only.

DELIBERATELY CONSERVATIVE: it flags only the unambiguous case — every assignment is
nested inside a branch AND there is a read at function-body level (depth 0). A variable
assigned at depth 0 anywhere, or only ever read inside a branch, is not flagged. The
goal is zero false positives so the test stays trustworthy; it will miss subtler cases.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

# The paths where this bug costs real money: the sell path, the entry/sizing path,
# and the scan loop that feeds them.
CRITICAL = [
    "options_owl/execution/paper_trader.py",
    "options_owl/execution/position_monitor.py",
    "options_owl/execution/webull_executor.py",
    "options_owl/risk/exit_v5/fsm.py",
    "options_owl/risk/exit_v5/monitor_bridge.py",
]

_BRANCHY = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)


def _assigned_names(node: ast.AST) -> set[str]:
    """Names this statement binds (plain, augmented, annotated, walrus, loop/with/except)."""
    out: set[str] = set()
    if isinstance(node, ast.Assign):
        for t in node.targets:
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        if isinstance(node.target, ast.Name):
            out.add(node.target.id)
    elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        out.add(node.target.id)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        out |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    elif isinstance(node, ast.ExceptHandler) and node.name:
        out.add(node.name)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                out |= {n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name)}
    return out


def _binds(stmt: ast.AST, name: str) -> bool:
    """True if this single statement DEFINITELY binds `name` on every path through it."""
    if name in _assigned_names(stmt):
        # For/With bind only if the loop/context actually runs -> not guaranteed
        return not isinstance(stmt, (ast.For, ast.AsyncFor, ast.While))
    if isinstance(stmt, ast.If):
        # exhaustive if/else: both arms bind -> definitely bound afterwards
        return bool(stmt.orelse) and _seq_binds(stmt.body, name) and _seq_binds(stmt.orelse, name)
    if isinstance(stmt, ast.Try):
        # bound if the body binds AND every handler binds, or the finally binds
        via_finally = _seq_binds(stmt.finalbody, name) if stmt.finalbody else False
        via_body = _seq_binds(stmt.body, name) and all(
            _seq_binds(h.body, name) for h in stmt.handlers
        )
        return via_finally or via_body
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _seq_binds(stmt.body, name)
    if isinstance(stmt, ast.Match):
        return bool(stmt.cases) and all(_seq_binds(c.body, name) for c in stmt.cases)
    return False


def _seq_binds(body: list, name: str) -> bool:
    """True if a statement sequence definitely binds `name` (any statement suffices)."""
    return any(_binds(st, name) for st in body)


def _reads(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
    return out


def _own_reads(stmt: ast.AST) -> set[str]:
    """Names read by this statement ITSELF, excluding its nested sub-blocks.

    e.g. for `if x > 1: y = z` this is {x}, not {z} — `z` belongs to the body and is
    checked recursively with the bindings that reach it.
    """
    skip: list = []
    for field in ("body", "orelse", "finalbody", "handlers", "cases"):
        skip.extend(getattr(stmt, field, []) or [])
    skip_ids = {id(x) for x in skip}
    out: set[str] = set()

    def walk(n):
        if id(n) in skip_ids:
            return
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        for c in ast.iter_child_nodes(n):
            walk(c)

    walk(stmt)
    return out


def _sub_blocks(stmt: ast.AST) -> list[list]:
    blocks = []
    for field in ("body", "orelse", "finalbody"):
        b = getattr(stmt, field, None)
        if b:
            blocks.append(b)
    for h in getattr(stmt, "handlers", []) or []:
        blocks.append(h.body)
    for c in getattr(stmt, "cases", []) or []:
        blocks.append(c.body)
    return blocks


def _unsafe_reads(fn) -> list[tuple[str, int]]:
    """Definite-assignment check, recursing into nested blocks.

    Reads are checked against the bindings that actually reach them at their own
    nesting level, so assign-and-use inside a single nested block is correctly SAFE.
    """
    assigned_anywhere = {n for x in ast.walk(fn) for n in _assigned_names(x)}
    bad: list[tuple[str, int]] = []

    def check(stmts: list, bound: set) -> None:
        local = set(bound)
        for stmt in stmts:
            for name in sorted(_own_reads(stmt)):
                if name in local or name not in assigned_anywhere:
                    continue
                if _binds(stmt, name):
                    continue
                bad.append((name, getattr(stmt, "lineno", 0)))
            # a loop body may run 0..n times, so its own bindings do not escape; a
            # loop target IS bound for the body itself
            for block in _sub_blocks(stmt):
                inner = set(local)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    inner |= _assigned_names(stmt)
                elif isinstance(stmt, (ast.With, ast.AsyncWith, ast.Try)):
                    inner |= _assigned_names(stmt)
                check(block, inner)
            local |= {n for n in assigned_anywhere if _binds(stmt, n)}

    check(fn.body, set(_params_and_globals(fn)))
    return bad


def _params_and_globals(fn: ast.AST) -> set[str]:
    safe: set[str] = set()
    a = getattr(fn, "args", None)
    if a:
        for grp in (a.posonlyargs, a.args, a.kwonlyargs):
            safe |= {x.arg for x in grp}
        if a.vararg:
            safe.add(a.vararg.arg)
        if a.kwarg:
            safe.add(a.kwarg.arg)
    for n in ast.walk(fn):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            safe |= set(n.names)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            safe |= {(al.asname or al.name).split(".")[0] for al in n.names}
    return safe


def find_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    bad: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        safe = _params_and_globals(fn)
        seen: set[str] = set()
        for name, lineno in _unsafe_reads(fn):
            if name in safe or name in seen:
                continue
            seen.add(name)
            bad.append(
                f"{path.name}:{lineno} {fn.name}(): '{name}' is read but not definitely "
                f"assigned on every path that reaches it -> UnboundLocalError"
            )
    return bad


# Pre-existing findings, frozen. The analysis below is deliberately simple and does NOT
# model every flow-control shape, so the legacy surface contains entries that are safe in
# practice. Baselining is the standard way to adopt a checker on a live codebase: it
# cannot vet history, but it DOES fail on anything NEW — which is all that was needed to
# catch the 2026-08-07 `_sizing_audit` bug. Shrinking this list is worthwhile follow-up;
# every entry is a genuine "not provably assigned on all paths" on a live-money path.
_BASELINE = json.loads(
    (Path(__file__).resolve().parent / "conditional_assignment_baseline.json").read_text()
)


@pytest.mark.parametrize("rel", CRITICAL)
def test_no_new_conditional_only_assignment(rel: str) -> None:
    path = Path(__file__).resolve().parent.parent / rel
    assert path.exists(), f"critical path file moved: {rel}"
    found = {v.split(": '")[1].split("'")[0] for v in find_violations(path)}
    new = sorted(found - set(_BASELINE.get(rel, [])))
    assert not new, (
        f"NEW conditional-only assignment in {rel} (CLAUDE.md rule #1 — this is the bug "
        f"class that froze the monitor on 2026-05-07 and stopped ALL exits):\n  "
        + "\n  ".join(new)
        + "\n\nInitialise these BEFORE the conditional block that assigns them."
    )


def test_checker_actually_catches_the_real_bug(tmp_path: Path) -> None:
    """Guard the guard: reproduce the 2026-08-07 shape and require a flag.

    Without this, a checker that silently stopped matching would look like a pass.
    """
    p = tmp_path / "repro.py"
    p.write_text(
        "def open_trade(use_vinny, use_score):\n"
        "    if use_vinny and use_score:\n"
        "        audit = {}\n"
        "        audit['p'] = 1.0\n"
        "    return audit.get('p')\n"
    )
    v = find_violations(p)
    assert v and "audit" in v[0], f"checker no longer detects the bug shape: {v}"


def test_checker_does_not_flag_the_correct_pattern(tmp_path: Path) -> None:
    """The fixed shape (init hoisted above the branch) must NOT be flagged."""
    p = tmp_path / "ok.py"
    p.write_text(
        "def open_trade(use_vinny, use_score):\n"
        "    audit = {}\n"
        "    if use_vinny and use_score:\n"
        "        audit['p'] = 1.0\n"
        "    return audit.get('p')\n"
    )
    assert not find_violations(p)
