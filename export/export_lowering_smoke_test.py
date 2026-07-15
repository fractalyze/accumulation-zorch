"""CPU lowering smoke test for the //export lowerers.

Each exporter lowers a frx prove/decide core to StableHLO bytecode that the Rust
GPU byte-match / bench consumes. A dangling reference inside an exporter (e.g. a
helper removed from a leaf module -- what broke ``export_ipa``'s ``curve.msm``)
is a runtime attribute access: invisible to ``bazel build``, ``py_compile``, and
pyflakes, tripped only when the core is actually lowered. The GPU byte-match that
would otherwise catch it is hardware-gated (``#[ignore]``), so this CPU-only test
guards the lower path of every exporter whose fixtures are committed.

Out of scope: the nark / fold_zk exporters lower inside ``main()`` from
cargo-generated fixtures (``ART/recursion_*.json``), not committed testdata.
"""
import os
import tempfile
import unittest
from pathlib import Path

from absl.testing import absltest

from accumulation_zorch import curve
import export_as_decide
import export_ipa
import export_ipa_fold
import export_prove

_CURVES = (curve.PALLAS, curve.VESTA)
_EXPORTERS = (export_prove, export_ipa, export_ipa_fold, export_as_decide)


def setUpModule():
    # The exporters write to their module-level ART dir (default: the repo
    # `artifacts/`, read-only under the test sandbox). Redirect each to a writable
    # temp dir (reclaimed at module teardown); the export functions read ART as a
    # global at call time.
    tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR"))
    unittest.addModuleCleanup(tmp.cleanup)
    art = Path(tmp.name)
    for mod in _EXPORTERS:
        mod.ART = art


class ExportLoweringSmokeTest(absltest.TestCase):
    """Every committed-fixture exporter must lower its core to non-empty bytecode
    without raising."""

    def _assert_lowered(self, out: Path):
        self.assertTrue(out.is_file(), f"{out} was not created")
        self.assertGreater(out.stat().st_size, 0, f"{out.name} lowered to 0 bytes")

    def test_nark_prove(self):
        self._assert_lowered(export_prove.export_zk_general())
        self._assert_lowered(export_prove.export_no_zk_general())

    def test_ipa_decider(self):
        for cv in _CURVES:
            self._assert_lowered(export_ipa.export_decider(cv))

    def test_ipa_fold(self):
        for cv in _CURVES:
            self._assert_lowered(export_ipa_fold.export_fold(cv))
            self._assert_lowered(export_ipa_fold.export_fold_zk(cv))

    def test_as_decider(self):
        for cv in _CURVES:
            self._assert_lowered(export_as_decide.export_decider(cv))


if __name__ == "__main__":
    absltest.main()
