//! The pluggable multi-scalar-multiplication seam.
//!
//! Every Pedersen commitment in the arkworks prove path is a multi-scalar
//! multiplication (MSM) over a [`CommitterKey`]'s generators. Upstream the
//! prover calls [`PedersenCommitment::commit`] directly; here that single call
//! is routed through [`MsmBackend`] so the inner MSM can be swapped for a GPU
//! backend without touching the surrounding prover logic.
//!
//! [`CpuBackend`] forwards verbatim to [`PedersenCommitment::commit`], so a
//! prove run parameterized by `CpuBackend` is byte-identical to the unmodified
//! arkworks prover (the oracle). The GPU backend lands in Slice 2.

use ark_ec::AffineCurve;
use ark_poly_commit::trivial_pc::{CommitterKey, PedersenCommitment};

/// Computes a (optionally hiding) Pedersen commitment to `elems` under `ck`.
///
/// The contract mirrors [`PedersenCommitment::commit`] exactly:
/// `comm = <ck.generators, elems>` (an MSM, zipped to the shorter length) plus,
/// when `randomizer` is `Some`, `randomizer · ck.hiding_generator`. The seam is
/// static (no `&self`) because the CPU backend is stateless and a future GPU
/// backend holds its PJRT session in a `thread_local!`.
pub trait MsmBackend<G: AffineCurve> {
    /// Commits to `elems` under `ck`, optionally hiding with `randomizer`.
    fn commit(
        ck: &CommitterKey<G>,
        elems: &[G::ScalarField],
        randomizer: Option<G::ScalarField>,
    ) -> G;
}

/// The reference backend: forwards to arkworks' [`PedersenCommitment::commit`].
///
/// This is the byte-identical oracle a prove run is validated against.
pub struct CpuBackend;

impl<G: AffineCurve> MsmBackend<G> for CpuBackend {
    #[inline]
    fn commit(
        ck: &CommitterKey<G>,
        elems: &[G::ScalarField],
        randomizer: Option<G::ScalarField>,
    ) -> G {
        PedersenCommitment::<G>::commit(ck, elems, randomizer)
    }
}
