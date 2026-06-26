# Vendored: zorch Poseidon sponge

A vendored subset of [`fractalyze/zorch`](https://github.com/fractalyze/zorch)
@ `8d651b4` — just the Poseidon duplex-sponge closure the accumulation prover's
Fiat-Shamir needs (`hash/duplex_sponge`, `hash/permutation`, `hash/poseidon/{params,poseidon,linear}`,
`_composite`, `fusion`). Pure-Python over `jax`; no other `zorch` modules are pulled
in. Vendored so this repo's Python is self-contained — no private-repo checkout on
`PYTHONPATH`. Licensed Apache-2.0 (see `LICENSE`).

Regenerate from an upstream `zorch` checkout:

```bash
cp <zorch>/zorch/{_composite,fusion}.py            python/zorch/
cp <zorch>/zorch/hash/{duplex_sponge,permutation}.py python/zorch/hash/
cp <zorch>/zorch/hash/poseidon/{params,poseidon,linear}.py python/zorch/hash/poseidon/
```
