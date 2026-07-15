"""Slice-4 byte-match: the multi-addend AS **instance** fold (num_addends=3).

The single-input AS prove (`r1cs_nark_as._build_zk_core`) folds one input + its
proof randomness (num_addends=2, `beta=[1,c]`). The full IVC step folds one input
INTO one prior accumulator (num_addends=3, `beta=[1,c₁,c₂]`), with the addend order
`[old_acc, input, proof_randomness]` — the `compute_*_components` chain order.

This isolates and byte-matches the **AS-level instance combine** of that fold —
the flagged multi-addend `beta` sponge (now absorbing an `AccumulatorInstance`
before the input) plus the `r1cs_input` / `comm_a,b,c` combine — against the golden
(`examples/dump_as_fold_zk.rs`). It reuses the validated NARK port
(`nark.prove_zk_core`) to re-derive input₂'s commitments and the same AS `comm_r` /
blinded-commitment logic `_build_zk_core` uses; the witness combine and the HP-level
old-accumulator fold are separate slices, so this checks the instance only (not the
folded `hp_instance`).

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:as_fold_zk_test
"""

import json
from pathlib import Path
from typing import Any

import frx
import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from frx import lax

from accumulation_zorch import absorbable, curve, field, nark, r1cs_nark_as, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_FIXTURE = _TESTDATA / "as_fold_zk_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fr_bytes(values: list[int]) -> bytes:
    return b"".join(cv.fr(v).tobytes() for v in values)


def _combined_instance(d: Any, s: Any, params: Any) -> tuple:
    """Reproduce the AS-level instance fold for one seed: `(combined_input,
    combined_comm_a, combined_comm_b, combined_comm_c)` over num_addends=3."""
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    generators = [_point(g) for g in d["generators"]]
    hiding = _point(d["hiding"])
    rows = len(a)
    nark_matrices_hash = bytes.fromhex(d["nark_matrices_hash_hex"])
    as_matrices_hash = bytes.fromhex(d["as_matrices_hash_hex"])

    input2 = [_fr(h) for h in s["input2_r1cs_input"]]
    witness2 = [_fr(h) for h in s["input2_witness"]]
    input_len, witness_len = len(input2), len(witness2)
    nark_r = [_fr(h) for h in s["r"]]
    nark_blinders = tuple(_fr(s[k]) for k in (
        "a_blinder", "b_blinder", "c_blinder", "r_a_blinder", "r_b_blinder", "r_c_blinder",
        "blinder_1", "blinder_2"))
    as_r1cs_r_input = _fr(s["as_r1cs_r_input"])
    as_r1cs_r_witness = _fr(s["as_r1cs_r_witness"])
    as_rand = tuple(_fr(s[k]) for k in ("as_rand_1", "as_rand_2", "as_rand_3"))

    acc = s["acc_prev_instance"]
    acc_r1cs_input = [_fr(h) for h in acc["r1cs_input"]]
    r1cs_r_input = [as_r1cs_r_input] * input_len
    r1cs_r_witness = [as_r1cs_r_witness] * witness_len

    accw = s["acc_prev_witness"]
    acc_blinded_witness = jnp.asarray(np.array([_fr(h) for h in accw["r1cs_blinded_witness"]], dtype=cv.fr))
    acc_sigma_abc = jnp.asarray(np.array(
        [_fr(accw["sigma_a"]), _fr(accw["sigma_b"]), _fr(accw["sigma_c"])], dtype=cv.fr))
    r1cs_r_witness_arr = jnp.asarray(np.array(r1cs_r_witness, dtype=cv.fr))
    as_rand_arr = jnp.asarray(np.array(list(as_rand), dtype=cv.fr))

    input2_bytes = _fr_bytes(input2)
    acc_bytes = _fr_bytes(acc_r1cs_input)
    r1cs_r_input_bytes = _fr_bytes(r1cs_r_input)

    bases_h = curve.stack_affine(cv, list(generators[:rows]) + [hiding])
    acc_comms = curve.stack_affine(cv, [
        _point(acc["comm_a"]), _point(acc["comm_b"]), _point(acc["comm_c"]),
        _point(acc["hp_comm_1"]), _point(acc["hp_comm_2"]), _point(acc["hp_comm_3"])])

    @frx.jit
    def core(bases_h: frx.Array, acc_comms: frx.Array) -> tuple:
        fr_one = jnp.asarray(np.array([1], dtype=cv.fr))
        nk = nark.prove_zk_core(cv, a, b, c, input2, witness2, bases_h, params,
                                nark_matrices_hash, nark_r, *nark_blinders)
        gamma = nk.gamma

        # The fold's AS proof-randomness commitments comm_r_M = commit(M·z_r, as_r_M).
        def _mz(matrix: nark.Matrix, zv: frx.Array) -> frx.Array:
            return field.matvec(jnp.asarray(nark.to_dense(cv, matrix, zv.shape[0])), zv)
        zr = jnp.asarray(np.array(r1cs_r_input + [as_r1cs_r_witness] * witness_len, dtype=cv.fr))
        comm_r_a = curve.commit_hiding(cv, _mz(a, zr), as_rand[0], bases_h)
        comm_r_b = curve.commit_hiding(cv, _mz(b, zr), as_rand[1], bases_h)
        comm_r_c = curve.commit_hiding(cv, _mz(c, zr), as_rand[2], bases_h)

        # input₂'s gamma-blinded NARK commitments (the addend the input contributes).
        one_gamma = jnp.concatenate([fr_one, gamma])
        blinded_comm_a = lax.msm(one_gamma, jnp.stack([nk.comm_a, nk.comm_r_a]))
        blinded_comm_b = lax.msm(one_gamma, jnp.stack([nk.comm_b, nk.comm_r_b]))
        blinded_comm_c = lax.msm(one_gamma, jnp.stack([nk.comm_c, nk.comm_r_c]))

        # beta over num_addends=3: as_sponge absorbs the accumulator instance, then
        # the input instance, then the proof randomness; squeeze 2 challenges.
        acc_inst_fe = r1cs_nark_as._acc_instance_fe(cv, acc_bytes, acc_comms)
        inst_fe = jnp.concatenate([
            jnp.asarray(absorbable.u8_batch_field_array(cv, input2_bytes)),
            absorbable.point_to_field_array_frx(cv, jnp.stack([nk.comm_a, nk.comm_b, nk.comm_c])),
            jnp.asarray(absorbable.option_flag(cv, True)),
            absorbable.point_to_field_array_frx(
                cv, jnp.stack([nk.comm_r_a, nk.comm_r_b, nk.comm_r_c, nk.comm_1, nk.comm_2])),
        ])
        pr_fe = jnp.concatenate([
            jnp.asarray(absorbable.option_flag(cv, True)),
            jnp.asarray(absorbable.u8_batch_field_array(cv, r1cs_r_input_bytes)),
            absorbable.point_to_field_array_frx(cv, jnp.stack([comm_r_a, comm_r_b, comm_r_c])),
        ])
        beta = r1cs_nark_as._beta_challenges_frx(  # (3,) = [1, c₁, c₂]
            cv, params, as_matrices_hash, inst_fe, pr_fe, acc_inst_fe=acc_inst_fe, num_challenges=2)

        # Fold under beta, in the order [acc, input, proof_randomness].
        combined_input = field.combine_vectors(
            jnp.asarray(np.array([acc_r1cs_input, input2, r1cs_r_input], dtype=cv.fr)), beta)
        combined_comm_a = lax.msm(beta, jnp.stack([acc_comms[0], blinded_comm_a, comm_r_a]))
        combined_comm_b = lax.msm(beta, jnp.stack([acc_comms[1], blinded_comm_b, comm_r_b]))
        combined_comm_c = lax.msm(beta, jnp.stack([acc_comms[2], blinded_comm_c, comm_r_c]))

        # Witness combine (no sponge — reuses beta): blinded witness + sigma_{a,b,c}
        # over the same [acc, input, proof_randomness] order. The fold's proof
        # randomness contributes the AS sigmas (as_rand_1/2/3); the input contributes
        # the NARK sigma_{a,b,c}; sigma_o is NARK-only and does not enter the AS fold.
        combined_blinded_witness = field.combine_vectors(
            jnp.stack([acc_blinded_witness, nk.blinded_witness, r1cs_r_witness_arr]), beta)
        combined_sigmas = acc_sigma_abc * beta[0] + nk.sigma_abc * beta[1] + as_rand_arr * beta[2]
        return (combined_input, combined_comm_a, combined_comm_b, combined_comm_c,
                combined_blinded_witness, combined_sigmas)

    return core(bases_h, acc_comms)


class AsFoldZkTest(absltest.TestCase):
    def test_as_fold_instance_matches_arkworks(self) -> None:
        d = json.loads(_FIXTURE.read_text())
        params = _params()
        for s in d["seeds"]:
            gi, gw = s["golden_instance"], s["golden_witness"]
            cin, cca, ccb, ccc, cbw, csig = _combined_instance(d, s, params)

            # Combined instance: r1cs_input (canonical-LE fr bytes) + comm_a/b/c.
            got_input = [np.asarray(v).tobytes().hex() for v in cin]
            want_input = [cv.fr(_fr(h)).tobytes().hex() for h in gi["r1cs_input"]]
            self.assertEqual(got_input, want_input, (
                f"[seed {s['seed']}] combined r1cs_input diverged:\n got  {got_input}\n want {want_input}"))
            for name, got_pt, want in (("comm_a", cca, gi["comm_a"]),
                                        ("comm_b", ccb, gi["comm_b"]),
                                        ("comm_c", ccc, gi["comm_c"])):
                got_hex = curve.point_to_bytes(cv, np.asarray(got_pt)).hex()
                want_hex = curve.point_to_bytes(cv, _point(want)).hex()
                self.assertEqual(got_hex, want_hex, (
                    f"[seed {s['seed']}] combined {name} diverged:\n got  {got_hex}\n want {want_hex}"))

            # Combined witness: blinded witness + sigma_{a,b,c}.
            got_bw = [np.asarray(v).tobytes().hex() for v in cbw]
            want_bw = [cv.fr(_fr(h)).tobytes().hex() for h in gw["r1cs_blinded_witness"]]
            self.assertEqual(got_bw, want_bw, (
                f"[seed {s['seed']}] combined blinded_witness diverged:\n got  {got_bw}\n want {want_bw}"))
            got_sig = [np.asarray(csig[i]).tobytes().hex() for i in range(3)]
            want_sig = [cv.fr(_fr(gw[k])).tobytes().hex() for k in ("sigma_a", "sigma_b", "sigma_c")]
            self.assertEqual(got_sig, want_sig, (
                f"[seed {s['seed']}] combined sigmas diverged:\n got  {got_sig}\n want {want_sig}"))

            print(f"  [seed {s['seed']}] combined instance + witness byte-matches arkworks "
                  f"(num_addends=3, beta=[1,c1,c2])")
        print("ALL SLICE-4 AS-FOLD INSTANCE+WITNESS CHECKS PASSED")


if __name__ == "__main__":
    absltest.main()
