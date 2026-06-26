"""jax/zk_dtypes port of the arkworks Pasta accumulation prover (zorch#303).

A byte-for-byte reimplementation of `accumulation-zorch`'s Rust prover
(`r1cs_nark_as` + `hp_as` + `r1cs_nark`) and its classic ark-sponge Fiat-Shamir
layer, validated against Rust-dumped golden bytes (`src/oracle.rs`). Built
bottom-up, each layer gated by a byte-match: field+serialize → group+commit →
poseidon+sponge → provers.
"""
