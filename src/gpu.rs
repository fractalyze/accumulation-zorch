//! The persistent PJRT client + the Pasta cycle-curve abstraction for the fused
//! GPU prover.
//!
//! [`session`] is the one leaked PJRT client the whole process shares (creating a
//! second aborts; dropping it `dlclose`s the plugin mid CUDA/absl teardown →
//! SIGSEGV, so it is never dropped). [`PastaCurve`] bundles what differs between
//! Pallas and Vesta at the plugin boundary — the short-Weierstrass parameters and
//! the PJRT buffer-type tags the fused core's inputs carry (`G1_AFFINE` for affine
//! points, `SF` for the scalar field, `BF` for the base / sponge field). The fused
//! consumer ([`crate::fused`]) compiles + runs its exported core on this client and
//! tags its inputs via these consts.
//!
//! (The per-MSM `GpuBackend` strategy — one `lax.msm` dispatch per Pedersen
//! commitment — was retired once the fused jax-exported core became
//! the sole GPU prove path; the CPU faithful-copy prover stays as the arkworks
//! oracle + fixture generator.)

use std::cell::Cell;

use ark_ec::models::SWModelParameters;
use xla_pjrt::Session;

// The shared xla-pjrt crate is curve-agnostic and exports no curve tags, so name
// the plugin's Pasta `PJRT_Buffer_Type` enum variants directly. (xla-pjrt's
// build.rs sets `prepend_enum_name(false)`, so the `sys` const is the
// header-faithful single-prefixed tag, not the bindgen-doubled name.)
const PALLAS_SF: xla_pjrt::sys::PJRT_Buffer_Type = xla_pjrt::sys::PJRT_Buffer_Type_PALLAS_SF;
const PALLAS_G1_AFFINE: xla_pjrt::sys::PJRT_Buffer_Type =
    xla_pjrt::sys::PJRT_Buffer_Type_PALLAS_G1_AFFINE;
const VESTA_SF: xla_pjrt::sys::PJRT_Buffer_Type = xla_pjrt::sys::PJRT_Buffer_Type_VESTA_SF;
const VESTA_G1_AFFINE: xla_pjrt::sys::PJRT_Buffer_Type =
    xla_pjrt::sys::PJRT_Buffer_Type_VESTA_G1_AFFINE;

/// A Pasta cycle curve, bundling what differs between Pallas and Vesta at the PJRT
/// plugin boundary: the short-Weierstrass parameters and the buffer-type tags the
/// fused core's inputs carry so the plugin routes them to the curve's thunks.
pub trait PastaCurve {
    /// The short-Weierstrass parameters (`Affine = GroupAffine<Params>`).
    type Params: SWModelParameters;
    /// The PJRT buffer-type tag for this curve's affine G1 points — what a fused
    /// core's affine inputs (`crate::fused`) carry so the plugin routes them to
    /// the curve's MSM thunk.
    const G1_AFFINE: xla_pjrt::sys::PJRT_Buffer_Type;
    /// The PJRT buffer-type tag for this curve's scalar field (`fr`) — what a
    /// general fused core's `fr` runtime inputs (the witness / public input /
    /// randomness lifted to runtime in the general prover) carry so the plugin types them.
    const SF: xla_pjrt::sys::PJRT_Buffer_Type;
    /// The PJRT buffer-type tag for this curve's **base** field (`fq`, the
    /// Poseidon / Fiat-Shamir sponge field) — what the zk general core's
    /// pre-encoded `u8_batch` runtime inputs carry. On the Pasta cycle
    /// `Pallas.fq == Vesta.fr`, so this is the *opposite* curve's `SF` tag.
    const BF: xla_pjrt::sys::PJRT_Buffer_Type;
}

/// Pallas.
pub struct Pallas;
impl PastaCurve for Pallas {
    type Params = ark_pallas::PallasParameters;
    const G1_AFFINE: xla_pjrt::sys::PJRT_Buffer_Type = PALLAS_G1_AFFINE;
    const SF: xla_pjrt::sys::PJRT_Buffer_Type = PALLAS_SF;
    // Pallas's base field fq == Vesta's scalar field, so its sponge-field buffers
    // carry the VESTA_SF tag.
    const BF: xla_pjrt::sys::PJRT_Buffer_Type = VESTA_SF;
}

/// Vesta.
pub struct Vesta;
impl PastaCurve for Vesta {
    type Params = ark_vesta::VestaParameters;
    const G1_AFFINE: xla_pjrt::sys::PJRT_Buffer_Type = VESTA_G1_AFFINE;
    const SF: xla_pjrt::sys::PJRT_Buffer_Type = VESTA_SF;
    // Vesta's base field fq == Pallas's scalar field, so its sponge-field buffers
    // carry the PALLAS_SF tag.
    const BF: xla_pjrt::sys::PJRT_Buffer_Type = PALLAS_SF;
}

thread_local! {
    /// One persistent PJRT client for the process, leaked. Creating a second
    /// client aborts, so the GPU path stays single-threaded; the client must
    /// also never be dropped, since dropping it `dlclose`s the plugin while its
    /// CUDA/absl thread-local destructors are still registered, segfaulting at
    /// thread teardown (bellman-zorch's `Box::leak` precedent).
    static SESSION: Cell<Option<&'static Session>> = const { Cell::new(None) };
}

/// The process's persistent PJRT client, loaded + leaked on first use. Shared
/// (`pub(crate)`) so the fused-core consumer (`crate::fused`) runs on the same
/// one client — a second client per process aborts.
pub(crate) fn session() -> &'static Session {
    SESSION.with(|cell| {
        if let Some(s) = cell.get() {
            return s;
        }
        // Safety: single-threaded GPU path → one client per process.
        let s: &'static Session = Box::leak(Box::new(unsafe { Session::new() }));
        cell.set(Some(s));
        s
    })
}
