/* Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Auto-generated from zkx_data.proto and zk_dtypes - DO NOT EDIT */

#ifndef ZKX_PJRT_C_PJRT_C_API_DATA_TYPES_H_
#define ZKX_PJRT_C_PJRT_C_API_DATA_TYPES_H_

typedef enum {
  PJRT_Buffer_Type_INVALID = 0,
  PJRT_Buffer_Type_PRED = 1,
  PJRT_Buffer_Type_S2 = 3,
  PJRT_Buffer_Type_S4 = 4,
  PJRT_Buffer_Type_S8 = 5,
  PJRT_Buffer_Type_S16 = 6,
  PJRT_Buffer_Type_S32 = 7,
  PJRT_Buffer_Type_S64 = 8,
  PJRT_Buffer_Type_U2 = 10,
  PJRT_Buffer_Type_U4 = 11,
  PJRT_Buffer_Type_U8 = 12,
  PJRT_Buffer_Type_U16 = 13,
  PJRT_Buffer_Type_U32 = 14,
  PJRT_Buffer_Type_U64 = 15,
  PJRT_Buffer_Type_U128 = 57,
  PJRT_Buffer_Type_U256 = 58,
  PJRT_Buffer_Type_TOKEN = 18,
  PJRT_Buffer_Type_BABYBEAR = 35,
  PJRT_Buffer_Type_BABYBEAR_MONT = 36,
  PJRT_Buffer_Type_BABYBEARX4 = 41,
  PJRT_Buffer_Type_BABYBEARX4_MONT = 42,
  PJRT_Buffer_Type_GOLDILOCKS = 39,
  PJRT_Buffer_Type_GOLDILOCKS_MONT = 40,
  PJRT_Buffer_Type_GOLDILOCKSX3 = 45,
  PJRT_Buffer_Type_GOLDILOCKSX3_MONT = 46,
  PJRT_Buffer_Type_KOALABEAR = 33,
  PJRT_Buffer_Type_KOALABEAR_MONT = 34,
  PJRT_Buffer_Type_KOALABEARX4 = 43,
  PJRT_Buffer_Type_KOALABEARX4_MONT = 44,
  PJRT_Buffer_Type_MERSENNE31 = 37,
  PJRT_Buffer_Type_MERSENNE31X2 = 47,
  PJRT_Buffer_Type_BN254_SF = 19,
  PJRT_Buffer_Type_BN254_SF_MONT = 20,
  PJRT_Buffer_Type_BN254_G1_AFFINE = 21,
  PJRT_Buffer_Type_BN254_G1_AFFINE_MONT = 22,
  PJRT_Buffer_Type_BN254_G1_JACOBIAN = 23,
  PJRT_Buffer_Type_BN254_G1_JACOBIAN_MONT = 24,
  PJRT_Buffer_Type_BN254_G1_XYZZ = 25,
  PJRT_Buffer_Type_BN254_G1_XYZZ_MONT = 26,
  PJRT_Buffer_Type_BN254_G2_AFFINE = 27,
  PJRT_Buffer_Type_BN254_G2_AFFINE_MONT = 28,
  PJRT_Buffer_Type_BN254_G2_JACOBIAN = 29,
  PJRT_Buffer_Type_BN254_G2_JACOBIAN_MONT = 30,
  PJRT_Buffer_Type_BN254_G2_XYZZ = 31,
  PJRT_Buffer_Type_BN254_G2_XYZZ_MONT = 32,
  /* ZKX Pasta G1 (Pallas + Vesta) — added for the M2/M3 GPU accumulation MSM
     (zkx#706, zkx#715). This vendored snapshot is hand-extended to match the
     generated plugin enum: gen_pjrt_data_types.py emits each EC buffer-type
     with the same value as its zkx_data.proto PrimitiveType, so these equal the
     proto values. Pasta is pairing-free, so G1 only (no G2). The proto also
     defines the _JACOBIAN/_XYZZ forms (65–68, 73–76); only the standard-form
     SF/G1_AFFINE pairs used at the MSM boundary are vendored here. */
  PJRT_Buffer_Type_PALLAS_SF = 61,
  PJRT_Buffer_Type_PALLAS_SF_MONT = 62,
  PJRT_Buffer_Type_PALLAS_G1_AFFINE = 63,
  PJRT_Buffer_Type_PALLAS_G1_AFFINE_MONT = 64,
  PJRT_Buffer_Type_VESTA_SF = 69,
  PJRT_Buffer_Type_VESTA_SF_MONT = 70,
  PJRT_Buffer_Type_VESTA_G1_AFFINE = 71,
  PJRT_Buffer_Type_VESTA_G1_AFFINE_MONT = 72,
  PJRT_Buffer_Type_BINARY_FIELD_T0 = 49,
  PJRT_Buffer_Type_BINARY_FIELD_T1 = 50,
  PJRT_Buffer_Type_BINARY_FIELD_T2 = 51,
  PJRT_Buffer_Type_BINARY_FIELD_T3 = 52,
  PJRT_Buffer_Type_BINARY_FIELD_T4 = 53,
  PJRT_Buffer_Type_BINARY_FIELD_T5 = 54,
  PJRT_Buffer_Type_BINARY_FIELD_T6 = 55,
  PJRT_Buffer_Type_BINARY_FIELD_T7 = 56,
} PJRT_Buffer_Type;

#endif  // ZKX_PJRT_C_PJRT_C_API_DATA_TYPES_H_
