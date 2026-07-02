"""Byte-match spike: zorch's IPA fold + `ArkIpaChallenger` reproduce
accumulation-zorch's exact IPA opening proof, over BOTH Pasta cycle curves.

An **equivalence test between two folds on identical inputs**, gating the
zorch-integration wiring:

* Oracle — `ipa_pc.open_no_zk`, the local arkworks-faithful NumPy fold
  (byte-matched to real arkworks via `ipa_as_test.py`'s Slice-2b). Returns
  `IpaProof(l_vec, r_vec, final_comm_key, c)` from the deferred even/odd
  (BCLMS) generator fold.
* Candidate — zorch's `zorch.pcs.ipa.prover._open_one`, driven by the
  accumulation consumer's `ipa_challenger.ArkIpaChallenger` (the same
  arkworks-faithful Fiat-Shamir the oracle uses). Its `IpaProof(l, r, a)` comes
  from a uniform per-round basis fold and a `lax.scan`; the folded generator is
  not carried in the proof, so the candidate's final commitment key is recovered
  the way zorch's own verifier does (`verifier.settle`): re-derive the round
  challenges through the challenger and pay the one size-`n` MSM
  `G_final = ⟨challenge_vector(u), G⟩`.

The assertion is byte-for-byte over each round's `L_j`/`R_j`, the collapsed
coefficient (`a == c`), and the final commitment key. The two folds organize the
basis differently (uniform per-round vs deferred two-level), so an exact
byte-match is a real statement: the deferred fold is a pure reorganization that
leaves every proof element identical.

The inputs come from `ipa_as_fixtures.json` (which carries `generators`, `h`, and
the `decider_coeffs` — the raw `ipa_fixtures.json` has NO generators and cannot
drive a fold). The IpaKey maps arkworks onto zorch as `basis = generators`,
`u = h` (arkworks `h` ↔ zorch's inner-product generator U; the candidate scales it
to `h' = U·ξ₀` itself), `s = None` (no-zk). The commitment is the honest Pedersen
commitment `⟨coeffs, generators⟩`; it only seeds ξ₀, but must be identical on both
sides. Both folds derive `b = (1, x, …, x^{n-1})` internally, so only `x` is passed.

Run EAGERLY (no `@jax.jit` over the open — `ArkIpaChallenger.seed` binds the
opening point/value host-side; the internal `lax.scan` runs eagerly fine):

    JAX_PLATFORMS=cpu PYTHONPATH=python \
      python python/accumulation_zorch/testing/ipa_zorch_spike_test.py
"""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax

from accumulation_zorch import curve, ipa_challenger, ipa_pc, sponge
from zorch.pcs.ipa.math import challenge_vector
from zorch.pcs.ipa.prover import _open_one
from zorch.pcs.ipa.setup import IpaKey

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (curve, IPA-AS fixture, sponge fixture for the ARK constants) per cycle curve —
# one generic prover, two curves. The `ipa_as_*` fixtures carry `generators`/`h`
# (the raw `ipa_fixtures.json` does not), so they are the ones that drive a fold.
_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_as_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_as_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params(cv: curve.Curve, sponge_fixture: Path) -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(sponge_fixture.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _pt_hex(cv: curve.Curve, point: Any) -> str:
    """arkworks compressed serialization (33 bytes) of an affine point — the byte
    form the comparison is made on. `dtype=cv.g1` normalizes a jacobian
    `lax.msm`/fold result to affine first."""
    return curve.point_to_bytes(cv, np.asarray(point, dtype=cv.g1)).hex()


def _fr_hex(cv: curve.Curve, value: Any) -> str:
    """32-byte LE canonical serialization of a scalar."""
    return np.asarray(value, dtype=cv.fr).tobytes().hex()


class _Inputs:
    """The shared fold inputs, parsed once per curve: the honest commitment, the
    opening point, the check-polynomial coefficients, the basis, and the
    inner-product generator `h`."""

    def __init__(self, cv: curve.Curve, fixture: Path) -> None:
        d = json.loads(fixture.read_text())
        # `decider_coeffs` is a real arkworks-golden power-of-two `fr` vector
        # (length 8 here); `generators` matches it in length. The opening point is a
        # genuine `fr` value (the accumulator's new opening point).
        self.coeffs_int = [_fr(h) for h in d["decider_coeffs"]]
        n = len(self.coeffs_int)
        self.generators = [_point(cv, g) for g in d["generators"]][:n]
        self.svk_h = _point(cv, d["h"])
        self.point_int = _fr(d["accumulator"]["point"])
        # The commitment only seeds ξ₀, but must be identical on both sides; the
        # honest Pedersen commitment of these coeffs is what `open_no_zk` expects.
        self.commitment = curve.pedersen_commit(cv, self.generators, self.coeffs_int)


def _candidate_final_key(
    cv: curve.Curve, params: Any, key: IpaKey, commitment: Any, x: Any, value: Any, proof: Any
) -> Any:
    """The candidate's fully-folded generator, recovered the way zorch's verifier
    settles a claim (`verifier.settle`): re-derive the round challenges by driving
    the `ArkIpaChallenger` over the proof's own `L_j`/`R_j`, then pay the one
    size-`n` MSM `G_final = ⟨challenge_vector(u), G⟩`. `_open_one` folds the basis
    inside its `lax.scan` and does not carry it in `IpaProof`, so this is the
    zorch-native way to name the folded generator."""
    ch = ipa_challenger.ark_challenger(cv, params)
    ch, _xi0 = ch.seed(commitment, x, value)
    us = []
    for j in range(proof.l.shape[0]):
        ch, uj = ch.challenge(proof.l[j], proof.r[j])
        us.append(uj)
    s = challenge_vector(jnp.stack(us))
    return lax.msm(s, key.basis[: s.shape[0]])


class IpaZorchSpikeTest(absltest.TestCase):
    def test_zorch_fold_byte_matches_ipa_pc_oracle(self) -> None:
        for cv, fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            inp = _Inputs(cv, fixture)

            # Oracle: accumulation-zorch's arkworks-faithful NumPy fold.
            oracle = ipa_pc.open_no_zk(
                cv, params, inp.svk_h, inp.commitment, inp.point_int, inp.coeffs_int, inp.generators)

            # Candidate: zorch's `_open_one`, driven by the ArkIpaChallenger.
            basis = jnp.asarray(np.stack([np.asarray(g, dtype=cv.g1) for g in inp.generators]))
            key = IpaKey(basis=basis, u=jnp.asarray(np.asarray(inp.svk_h, dtype=cv.g1)), s=None)
            coeffs = jnp.asarray(np.array(inp.coeffs_int, dtype=cv.fr))
            x = jnp.asarray(np.array([inp.point_int], dtype=cv.fr))[0]
            commitment = jnp.asarray(inp.commitment)
            fs = ipa_challenger.ark_challenger(cv, params)
            _fs, value, proof = _open_one(key, commitment, coeffs, x, fs)

            rounds = len(oracle.l_vec)
            self.assertEqual(proof.l.shape[0], rounds, f"[{cv.name}] round count")

            for j in range(rounds):
                got, want = _pt_hex(cv, proof.l[j]), _pt_hex(cv, oracle.l_vec[j])
                self.assertEqual(got, want, f"[{cv.name}] L[{j}]: {got} != {want}")
                got, want = _pt_hex(cv, proof.r[j]), _pt_hex(cv, oracle.r_vec[j])
                self.assertEqual(got, want, f"[{cv.name}] R[{j}]: {got} != {want}")

            got_a, want_c = _fr_hex(cv, proof.a), cv.fr(oracle.c).tobytes().hex()
            self.assertEqual(got_a, want_c, f"[{cv.name}] collapsed coeff a != c: {got_a} != {want_c}")

            final_key = _candidate_final_key(cv, params, key, commitment, x, value, proof)
            got_fk, want_fk = _pt_hex(cv, final_key), _pt_hex(cv, oracle.final_comm_key)
            self.assertEqual(got_fk, want_fk, f"[{cv.name}] final_comm_key: {got_fk} != {want_fk}")

            # The opened value the fold binds must be the honest evaluation
            # `⟨coeffs, (1, x, …)⟩` — a guard that the candidate opened this poly.
            z = [pow(inp.point_int, i, cv.fr_modulus) for i in range(len(inp.coeffs_int))]
            want_v = ipa_pc._inner_product(cv, inp.coeffs_int, z)
            self.assertEqual(_fr_hex(cv, value), cv.fr(want_v).tobytes().hex(), f"[{cv.name}] value")

            print(f"  [{cv.name}] zorch _open_one + ArkIpaChallenger byte-matches "
                  f"ipa_pc.open_no_zk ({rounds} fold rounds: L/R + collapsed c + final_comm_key)")

    def test_mismatched_coeffs_diverge(self) -> None:
        """A negative control: perturbing one coefficient must change the proof —
        guards the equivalence above against a vacuous (identity) pass."""
        cv, fixture, sponge_fixture = _CURVES[0]
        params = _params(cv, sponge_fixture)
        inp = _Inputs(cv, fixture)

        basis = jnp.asarray(np.stack([np.asarray(g, dtype=cv.g1) for g in inp.generators]))
        key = IpaKey(basis=basis, u=jnp.asarray(np.asarray(inp.svk_h, dtype=cv.g1)), s=None)
        x = jnp.asarray(np.array([inp.point_int], dtype=cv.fr))[0]
        commitment = jnp.asarray(inp.commitment)

        coeffs = jnp.asarray(np.array(inp.coeffs_int, dtype=cv.fr))
        _fs, _v, proof = _open_one(key, commitment, coeffs, x, ipa_challenger.ark_challenger(cv, params))

        bad_int = list(inp.coeffs_int)
        bad_int[0] = (bad_int[0] + 1) % cv.fr_modulus
        bad = jnp.asarray(np.array(bad_int, dtype=cv.fr))
        _fs2, _v2, proof2 = _open_one(key, commitment, bad, x, ipa_challenger.ark_challenger(cv, params))

        self.assertNotEqual(_pt_hex(cv, proof.l[0]), _pt_hex(cv, proof2.l[0]),
                            "a perturbed coefficient must change the proof's L[0]")
        print("  mutation check: a perturbed coefficient diverges the fold's L[0]")


if __name__ == "__main__":
    absltest.main()
