"""R1CS-NARK-AS **decider** byte-match (#10): the decider recomputes the six
size-`n` Pedersen commitments — `comm_{a,b,c} = commit(M·z, σ_M)` and the
`hp_as` decide check's `test_comm_{1,2,3} = commit(a_vec, ρ₁), commit(b_vec, ρ₂),
commit(a_vec∘b_vec, ρ₃)` — and accepts iff they equal the accumulator's stored
commitments. This is the decider's GPU-value work (the fused-core target, the
R1CS-NARK counterpart of the IPA-PC decider's size-`d` MSM).

Fixture-driven and faithful to arkworks: each fixture's accumulator is the
**unmodified arkworks** `(acc.instance ‖ acc.witness)`, emitted only for an
accumulator arkworks decides `true` (the `decide` flag, asserted by the dump).
The golden `comm_{a,b,c}` / `hp_instance.comm_{1,2,3}` live in `acc_instance_hex`
as compressed points; the decider inputs (`r1cs_input`, `r1cs_blinded_witness`,
the HP witness `a_vec`/`b_vec`, and the zk randomizers `σ` / `ρ`) live as scalars
in `acc_instance_hex` / `acc_witness_hex`. We parse the scalars and slice the
golden commitment *bytes* (no point decompression), recompute the commitments
with the curve-generic port, and compare byte-for-byte — over BOTH Pasta cycle
curves and both the no-zk and zk accumulators.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:as_decide_test
"""

import json
import struct
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, nark, r1cs_nark_as

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (curve, no-zk fixture, zk fixture)
_CURVES = [
    (curve.PALLAS, _TESTDATA / "as_fixtures.json", _TESTDATA / "as_zk_fixtures.json"),
    (curve.VESTA, _TESTDATA / "as_vesta_fixtures.json", _TESTDATA / "as_zk_vesta_fixtures.json"),
]

_FR_BYTES = 32
_POINT_BYTES = 33  # arkworks compressed affine


def _matrix(rows: Any) -> nark.Matrix:
    return [[(int.from_bytes(bytes.fromhex(coeff), "little"), idx) for coeff, idx in row] for row in rows]


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((int.from_bytes(bytes.fromhex(p["x_le_hex"]), "little"),
                  int.from_bytes(bytes.fromhex(p["y_le_hex"]), "little")))


def _take_fr_vec(buf: bytes, off: int) -> tuple[list[int], int]:
    """A `Vec<Fr>` CanonicalSerialize: `u64` LE length, then each element 32B LE."""
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    vals = [int.from_bytes(buf[off + i * _FR_BYTES: off + (i + 1) * _FR_BYTES], "little") for i in range(n)]
    return vals, off + n * _FR_BYTES


def _take_point_hex(buf: bytes, off: int) -> tuple[str, int]:
    """A 33-byte compressed affine point, returned as hex (compared byte-for-byte,
    no decompression needed)."""
    return buf[off:off + _POINT_BYTES].hex(), off + _POINT_BYTES


def _parse_instance(buf: bytes) -> tuple[list[int], list[str]]:
    """`AccumulatorInstance`: `r1cs_input` (`Vec<Fr>`), then the golden commitments
    `comm_a ‖ comm_b ‖ comm_c` and the embedded HP instance `comm_1 ‖ comm_2 ‖
    comm_3` (33B each). Returns `(r1cs_input, [comm_a, comm_b, comm_c, comm_1,
    comm_2, comm_3] as hex)`."""
    r1cs_input, off = _take_fr_vec(buf, 0)
    comms = []
    for _ in range(6):
        h, off = _take_point_hex(buf, off)
        comms.append(h)
    return r1cs_input, comms


def _parse_witness(buf: bytes, zk: bool) -> tuple[list[int], list[int], list[int],
                                                  tuple[int, int, int] | None,
                                                  tuple[int, int, int] | None]:
    """`AccumulatorWitness`: `r1cs_blinded_witness` (`Vec<Fr>`), the HP witness
    (`a_vec`, `b_vec`, then `Some(ρ₁,ρ₂,ρ₃)` / `None` randomness), then
    `Some(σ_a,σ_b,σ_c)` / `None` accumulator randomness."""
    blinded_witness, off = _take_fr_vec(buf, 0)
    hp_a_vec, off = _take_fr_vec(buf, off)
    hp_b_vec, off = _take_fr_vec(buf, off)

    def _take_opt_triple(off: int) -> tuple[tuple[int, int, int] | None, int]:
        flag = buf[off]
        off += 1
        if flag == 0:
            return None, off
        triple = tuple(int.from_bytes(buf[off + i * _FR_BYTES: off + (i + 1) * _FR_BYTES], "little")
                       for i in range(3))
        return triple, off + 3 * _FR_BYTES  # type: ignore[return-value]

    hp_rand, off = _take_opt_triple(off)
    sigmas, off = _take_opt_triple(off)
    if zk:
        assert hp_rand is not None and sigmas is not None, "zk witness must carry Some randomness"
    else:
        assert hp_rand is None and sigmas is None, "no-zk witness must carry None randomness"
    return blinded_witness, hp_a_vec, hp_b_vec, sigmas, hp_rand


class AsDecideTest(absltest.TestCase):
    def _check(self, cv: curve.Curve, fixture: Path, zk: bool) -> None:
        d = json.loads(fixture.read_text())
        a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
        generators = [_point(cv, g) for g in d["generators"]]
        hiding = _point(cv, d["hiding"]) if zk else None
        mode = "zk" if zk else "no-zk"

        for seed_entry in d["seeds"]:
            self.assertTrue(seed_entry["decide"],
                            f"[{cv.name}, {mode}] fixture seed {seed_entry['seed']} not arkworks-decided")
            r1cs_input, want_comms = _parse_instance(bytes.fromhex(seed_entry["acc_instance_hex"]))
            blinded_witness, hp_a_vec, hp_b_vec, sigmas, hp_rand = _parse_witness(
                bytes.fromhex(seed_entry["acc_witness_hex"]), zk)

            acc = r1cs_nark_as.Accumulator(
                r1cs_input=r1cs_input, blinded_witness=blinded_witness,
                hp_a_vec=hp_a_vec, hp_b_vec=hp_b_vec, sigmas=sigmas, hp_rand=hp_rand)
            got = r1cs_nark_as.decide(cv, a, b, c, generators, hiding, acc)
            got_hex = [curve.point_to_bytes(cv, p).hex() for p in got]

            labels = ["comm_a", "comm_b", "comm_c", "hp_comm_1", "hp_comm_2", "hp_comm_3"]
            for label, g, w in zip(labels, got_hex, want_comms):
                self.assertEqual(g, w, f"[{cv.name}, {mode}] seed {seed_entry['seed']} {label}: {g} != {w}")
            print(f"  [{cv.name}, {mode}] seed {seed_entry['seed']}: decider recomputes "
                  f"comm_{{a,b,c}} + HP test_comm_{{1,2,3}} byte-matching arkworks "
                  f"({len(r1cs_input)}+{len(blinded_witness)} vars)")

    def test_decide_no_zk_matches_arkworks(self) -> None:
        for cv, no_zk_fixture, _ in _CURVES:
            self._check(cv, no_zk_fixture, zk=False)

    def test_decide_zk_matches_arkworks(self) -> None:
        for cv, _, zk_fixture in _CURVES:
            self._check(cv, zk_fixture, zk=True)

    def test_mutation_breaks_match(self) -> None:
        """A perturbed blinded witness must diverge from the golden commitments —
        guards against a decider that ignores its inputs."""
        cv, no_zk_fixture, _ = _CURVES[0]
        d = json.loads(no_zk_fixture.read_text())
        a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
        generators = [_point(cv, g) for g in d["generators"]]
        seed_entry = d["seeds"][0]
        _, want_comms = _parse_instance(bytes.fromhex(seed_entry["acc_instance_hex"]))
        r1cs_input, _ = _parse_instance(bytes.fromhex(seed_entry["acc_instance_hex"]))
        blinded_witness, hp_a_vec, hp_b_vec, _, _ = _parse_witness(
            bytes.fromhex(seed_entry["acc_witness_hex"]), zk=False)

        bad = list(blinded_witness)
        bad[0] = (bad[0] + 1) % cv.fr_modulus
        acc = r1cs_nark_as.Accumulator(
            r1cs_input=r1cs_input, blinded_witness=bad,
            hp_a_vec=hp_a_vec, hp_b_vec=hp_b_vec, sigmas=None, hp_rand=None)
        got = r1cs_nark_as.decide(cv, a, b, c, generators, None, acc)
        got_comm_a = curve.point_to_bytes(cv, got[0]).hex()
        self.assertNotEqual(got_comm_a, want_comms[0],
                            "a perturbed blinded witness must change comm_a")
        print("  mutation check: a perturbed blinded witness diverges from the golden comm_a")


if __name__ == "__main__":
    absltest.main()
