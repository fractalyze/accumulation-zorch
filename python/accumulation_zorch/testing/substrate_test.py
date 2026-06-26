"""Slice-1 byte-match: field + point serialization vs arkworks golden bytes.

Reads the fixtures dumped by `cargo run --example dump_fixtures`
(`python/testdata/substrate_fixtures.json`) and asserts that the Python
field/curve serializers reproduce arkworks `CanonicalSerialize` byte-for-byte,
that `pallas_g1_affine` is `ark_pallas::Affine`, and that zk_dtypes' CPU group
ops match the Rust-computed points.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=. \
      python accumulation_zorch/testing/substrate_test.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from accumulation_zorch import curve, field

cv = curve.PALLAS

_FIXTURES = (
    Path(__file__).resolve().parents[2] / "testdata" / "substrate_fixtures.json"
)


def _load() -> Any:
    return json.loads(_FIXTURES.read_text())


def test_field_serialization() -> None:
    data = _load()
    dtypes = {"fq": cv.fq, "fr": cv.fr}
    for which, dtype in dtypes.items():
        for entry in data["fields"][which]:
            value = int(entry["value"], 16)  # Rust dumps into_repr() as BE hex
            got = dtype(value).tobytes().hex()
            assert got == entry["canonical_hex"], (
                f"{which}/{entry['label']}: {got} != {entry['canonical_hex']}"
            )
            # round-trip: rebuilding from the canonical bytes is stable
            v2 = field.fe_value(dtype(value))
            assert v2 == value, f"{which}/{entry['label']} value round-trip: {v2} != {value}"
    # hard sanity anchors independent of the dump
    assert cv.fq(1).tobytes().hex() == "01" + "00" * 31
    print(f"  field serialization OK ({len(data['fields']['fq'])} Fq + "
          f"{len(data['fields']['fr'])} Fr)")


def test_point_serialization_and_dtype_mapping() -> None:
    data = _load()
    for entry in data["points"]:
        x = int.from_bytes(bytes.fromhex(entry["x_le_hex"]), "little")
        y = int.from_bytes(bytes.fromhex(entry["y_le_hex"]), "little")
        pt = cv.g1((x, y))
        got = curve.point_to_bytes(cv, pt).hex()
        assert got == entry["canonical_hex"], (
            f"point/{entry['label']}: {got} != {entry['canonical_hex']}"
        )
    # pallas_g1_affine IS ark_pallas::Affine: its 1·G generator must serialize
    # to the dumped generator bytes. The zk_dtypes affine dtype reads an int as
    # the scalar `k·G`, so `[1]` is the generator.
    gen = next(p for p in data["points"] if p["label"] == "generator")
    generator = np.array([1], dtype=cv.g1)
    assert curve.point_to_bytes(cv, generator).hex() == gen["canonical_hex"], (
        "pallas_g1_affine generator != ark_pallas generator — dtype mapping wrong"
    )
    print(f"  point serialization + dtype mapping OK ({len(data['points'])} points)")


def test_cpu_group_ops_match_arkworks() -> None:
    data = _load()
    by = {p["label"]: p["canonical_hex"] for p in data["points"]}
    g = np.array([1], dtype=cv.g1)  # the generator 1·G
    two_g = g + g  # CPU point add
    assert curve.point_to_bytes(cv, two_g).hex() == by["two_g"], "G+G != 2G (CPU add)"
    k = g * np.array([12345], dtype=cv.fr)  # CPU scalar mul: 12345·G
    assert curve.point_to_bytes(cv, k).hex() == by["k12345_g"], "12345·G mismatch (CPU mul)"
    g_plus_2g = g + g * np.array([2], dtype=cv.fr)  # G + 2·G
    assert curve.point_to_bytes(cv, g_plus_2g).hex() == by["g_plus_2g"], "G+2G mismatch"
    print("  CPU group ops (add, scalar-mul) match arkworks OK")


def _point(entry: Any) -> Any:
    x = int.from_bytes(bytes.fromhex(entry["x_le_hex"]), "little")
    y = int.from_bytes(bytes.fromhex(entry["y_le_hex"]), "little")
    return cv.g1((x, y))


def test_pedersen_commit_matches_arkworks() -> None:
    ped = _load()["pedersen"]
    generators = [_point(g) for g in ped["generators"]]
    hiding = _point(ped["hiding"])
    elems = [int(e) for e in ped["elems"]]
    for case in ped["cases"]:
        r = case["randomizer"]
        if r is None:
            commit = curve.pedersen_commit(cv, generators, elems)
            tag = "no-hiding"
        else:
            commit = curve.pedersen_commit(
                cv, generators, elems, hiding=hiding, randomizer=int(r)
            )
            tag = f"hiding(r={r})"
        got = curve.point_to_bytes(cv, commit).hex()
        assert got == case["result_canonical_hex"], (
            f"pedersen {tag}: {got} != {case['result_canonical_hex']}"
        )
    print(f"  Pedersen commit (CPU MSM) matches arkworks OK "
          f"({len(ped['cases'])} cases, {len(elems)} elems)")


def main() -> None:
    print("slice-1 substrate byte-match:")
    test_field_serialization()
    test_point_serialization_and_dtype_mapping()
    test_cpu_group_ops_match_arkworks()
    test_pedersen_commit_matches_arkworks()
    print("ALL SLICE-1 SUBSTRATE CHECKS PASSED")


if __name__ == "__main__":
    main()
