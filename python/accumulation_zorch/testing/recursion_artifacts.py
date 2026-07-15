"""Locating the off-tree recursion fixtures — the on-demand recursion gates' input.

The recursion fixtures (`recursion_*_fixtures.json`, 17-147 MB) are emitted by the
`--features recursion` Rust dump tests and gitignored, so they exist only on a machine
that deliberately ran the dumps. Bazel cannot declare them as `data`, and both rules
below follow from that.

**The directory must be named explicitly, via `$ACCUMULATION_ZORCH_ARTIFACTS`.** For these
three tests there is deliberately no `<repo>/artifacts` default: a default would have to be
reached by resolving `__file__` back out of the runfiles tree into the source checkout, so
the test would read a file Bazel has never heard of — and, not being a declared input, one
Bazel cannot invalidate its cached result on. Requiring the variable keeps the
non-hermetic read visible at the invocation rather than buried in a default. (The
`external` tag on these targets covers the other half: Bazel re-runs them instead of
trusting that cache.)

This is the rule for the `manual` test targets here, **not** a repo-wide one — the
`export/export_*.py` lowering binaries resolve the same variable with an `artifacts/`
default, and the Rust side (`fixture_json::artifacts_dir`) defaults to
`<manifest>/artifacts`. Three resolutions for one variable is one too many; draining them
into a single helper is tracked separately. What forces the difference *here* is Bazel's
caching of test results, which neither of the others is subject to.

**A missing fixture is an error, not a skip.** These targets are `manual`, so the only way
one runs is a deliberate request — and then a skip is never what the caller asked for. It
would also be invisible: a skipped absltest case exits 0, which Bazel reports as `PASSED`,
indistinguishable from a real byte-match.
"""

import os
from pathlib import Path

_ENV = "ACCUMULATION_ZORCH_ARTIFACTS"


def fixture(name: str, dump: str) -> Path:
    """The path to off-tree fixture `name`, else raise saying how to produce it.

    `dump` is the cargo command that generates `name`; it is quoted back in the error, so
    a caller who has never run the dumps can act on the failure without leaving it.
    """
    artifacts = os.environ.get(_ENV)
    if not artifacts:
        raise RuntimeError(
            f"${_ENV} is unset; it must hold the absolute path of the directory "
            f"containing {name}:\n\n"
            f"    {_ENV}=$PWD/artifacts bazel test <target>\n")
    if not Path(artifacts).is_absolute():
        # Caught separately from the missing-fixture case below, which a relative
        # path would otherwise masquerade as: bazel runs the test from its runfiles
        # tree, so `artifacts/x.json` resolves there and reports "no fixture" even
        # though the file exists — and regenerating it puts it right back where it
        # was, for the same error.
        raise RuntimeError(
            f"${_ENV} must be absolute, got {artifacts!r}. Bazel runs the test from "
            f"its runfiles tree, so a relative path resolves against that, not your "
            f"checkout. Use $PWD/{artifacts}.\n")
    path = Path(artifacts) / name
    if not path.exists():
        raise RuntimeError(
            f"no fixture at {path} — generate it with:\n\n"
            f"    {_ENV}={artifacts} {dump}\n")
    return path
