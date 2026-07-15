//! A thin Rust consumer of the frx-exported fused GPU accumulation prover core.
//!
//! The whole prover — the arkworks `ark-accumulation` `r1cs_nark_as` + `hp_as`
//! native prove path over the Pasta cycle — is authored in Python/FRX
//! (`python/accumulation_zorch/`), exported to a single fused StableHLO core, and
//! run here as one PJRT call (see [`fused`]). Rust feeds the committer key plus
//! the assignment/randomness and re-serializes the output; it does not
//! re-implement the prover. The byte-match oracle is the pristine arkworks prover
//! (the `ark-accumulation` dev-dependency at `../accumulation`), driven by the
//! fixture generators in `examples/` and `tests/recursion_step.rs`.

// The default build carries no unsafe code. The GPU path (feature `gpu`) needs
// unsafe FFI into the xla-pjrt plugin, so under that feature the crate-wide ban
// is downgraded to `deny` and the `gpu` / `fused` modules get a scoped exception.
#![cfg_attr(not(feature = "gpu"), forbid(unsafe_code))]
#![cfg_attr(feature = "gpu", deny(unsafe_code))]

/// The `ark_pallas` ↔ zk_dtypes Pasta byte bridge for the GPU MSM boundary.
pub mod wire;

/// The persistent PJRT client + Pasta cycle-curve abstraction (feature `gpu`):
/// the one leaked client the fused consumer compiles + runs its core on, plus
/// each curve's PJRT buffer-type tags.
#[cfg(feature = "gpu")]
#[allow(unsafe_code)]
pub mod gpu;

/// Thin consumer of the fused frx-exported prove core (feature `gpu`): loads the
/// general prover `.mlirbc` (the assignment + randomness are runtime inputs) and
/// runs the whole prove — every commitment, the NARK + HP cores, and all three
/// Fiat-Shamir sponges — as one PJRT call.
#[cfg(feature = "gpu")]
#[allow(unsafe_code)]
pub mod fused;
