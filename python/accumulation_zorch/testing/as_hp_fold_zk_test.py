"""Slice-4 byte-match: the HP-level old-accumulator fold (num_addends=3).

The AS-level fold (`as_fold_zk_test`) byte-matches the combined `r1cs_input` /
`comm_{a,b,c}` and the AS witness, but leaves the accumulator's **HP instance**
(`hp_comm_1/2/3`) and **HP witness** (`hp_a_vec`, `hp_b_vec`, `hp_rand_*`) to this
slice. Those are produced by `ASForHadamardProducts::prove` run over the fold's
two HP inputs.

The HP fold is the same num_inputs=2 machinery as the single-input zk HP prove
(`hp_as.prove_zk_core`, which pads one real input with a zero placeholder), with
the second input promoted from the zero placeholder to the **real old
accumulator**: its HP instance `(hp_comm_1, hp_comm_2, hp_comm_3)`, opening vectors
`(hp_a_vec, hp_b_vec)`, and randomness `(hp_rand_1, hp_rand_2, hp_rand_3)` — read
from `acc_prev` in the fixture. The HP input order is `[new_input, old_acc]`
(`inputs.chain(old_accumulators)` in `prove_with_backend`).

The new input's HP instance/witness are re-derived in jax from input₂'s NARK
(the gamma-blinded `comm_a/comm_b/comm_prod` and the `M·z` openings over
`z = r1cs_input ‖ blinded_witness`), exactly as `r1cs_nark_as._build_zk_core`
does for the single-input AS prove. The fold's fresh HP hiding randomness
(`hp_hiding_a/b`, `hp_rand_1/2/3`) is replayed from the dump.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/as_hp_fold_zk_test.py
"""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from accumulation_zorch import absorbable, curve, hp_as, jcurve, jfield, nark, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_FIXTURE = _TESTDATA / "as_fold_zk_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"

HP_AS_PROTOCOL_NAME = b"AS-FOR-HP-2020"


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _hp_fold(d: Any, s: Any, params: Any) -> tuple:
    """Run the HP-level fold for one seed, returning the combined HP
    `(instance (3,), a_open, b_open, rand (3,))`."""
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    generators = [_point(g) for g in d["generators"]]
    hiding = _point(d["hiding"])
    rows = len(a)
    supported_num_elems = d["supported_num_elems"]
    nark_matrices_hash = bytes.fromhex(d["nark_matrices_hash_hex"])

    input2 = [_fr(h) for h in s["input2_r1cs_input"]]
    witness2 = [_fr(h) for h in s["input2_witness"]]
    nark_r = [_fr(h) for h in s["r"]]
    nark_blinders = tuple(_fr(s[k]) for k in (
        "a_blinder", "b_blinder", "c_blinder", "r_a_blinder", "r_b_blinder", "r_c_blinder",
        "blinder_1", "blinder_2"))
    hp_hiding_a, hp_hiding_b = _fr(s["hp_hiding_a"]), _fr(s["hp_hiding_b"])
    hp_rand = tuple(_fr(s[k]) for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3"))

    # The old accumulator's HP input: instance commitments + opening vectors +
    # randomness, read straight from acc_prev (the second HP input of the fold).
    acc = s["acc_prev_instance"]
    accw = s["acc_prev_witness"]
    old_a = jnp.asarray(np.array([_fr(h) for h in accw["hp_a_vec"]], dtype=cv.fr))
    old_b = jnp.asarray(np.array([_fr(h) for h in accw["hp_b_vec"]], dtype=cv.fr))
    old_rand = jnp.asarray(np.array(
        [_fr(accw["hp_rand_1"]), _fr(accw["hp_rand_2"]), _fr(accw["hp_rand_3"])], dtype=cv.fr))

    bases_h = jcurve.stack_affine(cv, list(generators[:rows]) + [hiding])
    id_pt = jcurve.stack_affine(cv, [cv.g1((0, 0))])
    old_hp_comms = jcurve.stack_affine(
        cv, [_point(acc["hp_comm_1"]), _point(acc["hp_comm_2"]), _point(acc["hp_comm_3"])])

    @jax.jit
    def core(bases_h: jax.Array, id_pt: jax.Array, old_hp_comms: jax.Array,
             old_a: jax.Array, old_b: jax.Array, old_rand: jax.Array) -> tuple:
        fr_one = jnp.asarray(np.array([1], dtype=cv.fr))
        nk = nark.prove_zk_core(cv, a, b, c, input2, witness2, bases_h, params,
                                nark_matrices_hash, nark_r, *nark_blinders)
        gamma = nk.gamma

        # input₂'s HP instance: the gamma-blinded NARK commitments
        # (comm_prod folds comm_1, comm_2 at gamma, gamma²).
        one_gamma = jnp.concatenate([fr_one, gamma])
        blinded_comm_a = jcurve.msm(one_gamma, jnp.stack([nk.comm_a, nk.comm_r_a]))
        blinded_comm_b = jcurve.msm(one_gamma, jnp.stack([nk.comm_b, nk.comm_r_b]))
        comm_prod = jcurve.msm(jnp.concatenate([fr_one, gamma, gamma * gamma]),
                               jnp.stack([nk.comm_c, nk.comm_1, nk.comm_2]))

        # input₂'s HP opening: M·z over z = r1cs_input ‖ blinded_witness; the HP
        # input randomness is the NARK (sigma_a, sigma_b, sigma_o).
        def _mz(matrix: nark.Matrix, zv: jax.Array) -> jax.Array:
            return jfield.matvec(jnp.asarray(nark.to_dense(cv, matrix, zv.shape[0])), zv)
        zw = jnp.concatenate([jnp.asarray(np.array(input2, dtype=cv.fr)), nk.blinded_witness])
        new_rand = jnp.stack([nk.sigma_abc[0], nk.sigma_abc[1], nk.sigma_o])

        hp_sponge = absorbable.fork(cv, sponge.new_sponge(params), HP_AS_PROTOCOL_NAME)
        hp = hp_as.prove_zk_core(
            cv, bases_h, id_pt, jnp.stack([blinded_comm_a, blinded_comm_b, comm_prod]),
            _mz(a, zw), _mz(b, zw), new_rand, supported_num_elems, params,
            hp_hiding_a, hp_hiding_b, hp_rand[0], hp_rand[1], hp_rand[2],
            old_inst=old_hp_comms, old_a=old_a, old_b=old_b, old_rand=old_rand,
            base_sponge=hp_sponge)
        return hp.instance, hp.a_open, hp.b_open, hp.rand

    return core(bases_h, id_pt, old_hp_comms, old_a, old_b, old_rand)


def test_hp_fold_matches_arkworks() -> None:
    d = json.loads(_FIXTURE.read_text())
    params = _params()
    for s in d["seeds"]:
        gi, gw = s["golden_instance"], s["golden_witness"]
        instance, a_open, b_open, rand = _hp_fold(d, s, params)

        # Combined HP instance: hp_comm_1/2/3.
        for name, idx, want in (("hp_comm_1", 0, gi["hp_comm_1"]),
                                ("hp_comm_2", 1, gi["hp_comm_2"]),
                                ("hp_comm_3", 2, gi["hp_comm_3"])):
            got_hex = curve.point_to_bytes(cv, np.asarray(instance[idx])).hex()
            want_hex = curve.point_to_bytes(cv, _point(want)).hex()
            assert got_hex == want_hex, (
                f"[seed {s['seed']}] combined {name} diverged:\n got  {got_hex}\n want {want_hex}")

        # Combined HP witness: a_vec, b_vec, randomness (rand_1, rand_2, rand_3).
        got_a = [np.asarray(v).tobytes().hex() for v in a_open]
        want_a = [cv.fr(_fr(h)).tobytes().hex() for h in gw["hp_a_vec"]]
        assert got_a == want_a, (
            f"[seed {s['seed']}] combined hp_a_vec diverged:\n got  {got_a}\n want {want_a}")
        got_b = [np.asarray(v).tobytes().hex() for v in b_open]
        want_b = [cv.fr(_fr(h)).tobytes().hex() for h in gw["hp_b_vec"]]
        assert got_b == want_b, (
            f"[seed {s['seed']}] combined hp_b_vec diverged:\n got  {got_b}\n want {want_b}")
        got_r = [np.asarray(rand[i]).tobytes().hex() for i in range(3)]
        want_r = [cv.fr(_fr(gw[k])).tobytes().hex() for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3")]
        assert got_r == want_r, (
            f"[seed {s['seed']}] combined hp_rand diverged:\n got  {got_r}\n want {want_r}")

        print(f"  [seed {s['seed']}] combined HP instance + witness byte-matches arkworks "
              f"(num_addends=3, old-accumulator fold)")
    print("ALL SLICE-4 HP-FOLD CHECKS PASSED")


def main() -> None:
    print("slice-4 HP-level old-accumulator fold byte-match (Pallas, num_addends=3):")
    test_hp_fold_matches_arkworks()


if __name__ == "__main__":
    main()
