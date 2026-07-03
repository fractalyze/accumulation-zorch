# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Vendored subset of zorch (Apache-2.0, see LICENSE; pin recorded in ``VENDOR``):
the Poseidon duplex-sponge + fused-region helpers the accumulation prover's
Fiat-Shamir transcript builds on, plus the ``pcs/ipa`` IPA-PC prover/verifier (with
``transcript``, ``poly``, ``utils``) that the ``ipa_pc_as`` accumulation reuses.
Pure-Python over `jax`."""
