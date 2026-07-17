"""The other half of the test obligation: a chain nobody replayed.

Every other gate here is (fixed input -> golden bytes) — arkworks is the oracle and
the fixtures carry the circuit, the committer key, the sponge constants and every
sampled randomizer. That pins the port to arkworks but says nothing about whether
the scheme *closes* on inputs arkworks never saw: a prove whose accumulator no
fold consumes, and a fold whose output no decider settles, could both be wrong in
ways a byte-match cannot see.

This runs the chain the fixtures cannot: accumulate a statement of our own, fold
it forward, and make the decider settle the result — no fixture data anywhere, on
freshly sampled randomness. The oracle is the scheme's own closure (the six
deferred IOUs re-derive from the witness), which is exactly what a golden file
cannot check.
"""

import numpy as np
from absl.testing import absltest

from accumulation_zorch import curve, r1cs_nark_as, sponge

_N = 8          # R1CS rows = chain steps
_W = 2          # first witness wire: z = [1, out] ‖ [w_0 ..]
_SEED = 3


def _circuit() -> tuple[list, list, list]:
    """`(w_i + 5) * w_i = w_{i+1}`, the last row landing on the public output.

    A and B must stay different linear combinations: if both selected `w_i` the
    prove would emit `comm_a == comm_b` and the fold would combine two identical
    commitments, so a fold bug that swapped them could not fail here."""
    a = [[(1, _W + i), (5, 0)] for i in range(_N)]
    b = [[(1, _W + i)] for i in range(_N)]
    c = [[(1, _W + i + 1)] for i in range(_N - 1)] + [[(1, 1)]]
    return a, b, c


def _assignment(cv: curve.Curve) -> tuple[list[int], list[int]]:
    w = [cv.fr(_SEED)]
    for _ in range(_N - 1):
        w.append((w[-1] + cv.fr(5)) * w[-1])
    out = (w[-1] + cv.fr(5)) * w[-1]
    return [1, int(out)], [int(x) for x in w]


class AsChainTest(absltest.TestCase):
    def _decides(self, cv: curve.Curve, a: list, b: list, c: list, generators: list,
                 hiding: np.ndarray, acc: r1cs_nark_as.FoldedAccumulator) -> bool:
        """The decider settles `acc`: recompute the six commitments from the witness
        and compare against the ones the accumulator carries. Both sides are
        canonical `g1_affine`, so array equality is exact point equality."""
        got = r1cs_nark_as.decide(cv, a, b, c, generators, hiding, acc.to_decide())
        return np.array_equal(np.array(got), np.array(acc.comms))

    def test_accumulate_then_fold_decides_with_no_fixture(self) -> None:
        for cv in (curve.PALLAS, curve.VESTA):
            rng = np.random.default_rng(0)
            gens = list(np.arange(1, _N + 2, dtype=cv.g1))
            generators, hiding = gens[:_N], gens[_N]
            params = sponge.default_params(cv)
            a, b, c = _circuit()
            r1cs_input, witness = _assignment(cv)

            def step(acc: r1cs_nark_as.FoldedAccumulator | None = None
                     ) -> r1cs_nark_as.FoldedAccumulator:
                rnd = r1cs_nark_as.sample_randomness(cv, rng, len(witness))
                if acc is None:
                    return r1cs_nark_as.accumulate(
                        cv, a, b, c, r1cs_input, witness, generators, hiding, params, _N, rnd)[0]
                return r1cs_nark_as.fold(
                    cv, a, b, c, r1cs_input, witness, generators, hiding, params, _N, acc, rnd)[0]

            acc = step()
            self.assertTrue(self._decides(cv, a, b, c, generators, hiding, acc),
                            f"[{cv.name}] decider rejected a freshly accumulated statement")
            for i in range(3):
                folded = step(acc)
                self.assertTrue(self._decides(cv, a, b, c, generators, hiding, folded),
                                f"[{cv.name}] decider rejected the accumulator after fold {i + 1}")
                self.assertEqual(len(folded.blinded_witness), len(acc.blinded_witness),
                                 f"[{cv.name}] fold grew the accumulator — it must stay fixed-size")
                acc = folded

    def test_decider_rejects_a_tampered_folded_witness(self) -> None:
        """Without this the chain test would pass on a decider that accepts anything."""
        cv = curve.PALLAS
        rng = np.random.default_rng(0)
        gens = list(np.arange(1, _N + 2, dtype=cv.g1))
        generators, hiding = gens[:_N], gens[_N]
        params = sponge.default_params(cv)
        a, b, c = _circuit()
        r1cs_input, witness = _assignment(cv)
        acc_rnd = r1cs_nark_as.sample_randomness(cv, rng, len(witness))
        fold_rnd = r1cs_nark_as.sample_randomness(cv, rng, len(witness))
        acc = r1cs_nark_as.accumulate(cv, a, b, c, r1cs_input, witness, generators, hiding,
                                      params, _N, acc_rnd)[0]
        acc = r1cs_nark_as.fold(cv, a, b, c, r1cs_input, witness, generators, hiding,
                                params, _N, acc, fold_rnd)[0]

        bw = [int(x) for x in acc.blinded_witness]
        tampered = acc._replace(
            blinded_witness=np.asarray([bw[0] + 1] + bw[1:], dtype=cv.fr))
        self.assertFalse(self._decides(cv, a, b, c, generators, hiding, tampered),
                         f"[{cv.name}] decider accepted a tampered folded witness")


if __name__ == "__main__":
    absltest.main()
