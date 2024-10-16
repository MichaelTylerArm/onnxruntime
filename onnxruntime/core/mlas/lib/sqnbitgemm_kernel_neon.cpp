/*++

Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include <arm_neon.h>

#include <cassert>

#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_neon.h"
#include "sqnbitgemm_q8_block.h"

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"

namespace sqnbitgemm_neon
{

namespace
{

//
// Quantized B data packing function implementation.
//

size_t
SQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, BlkLen, kai_dt_bf16);
}

void
SQ4BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool has_zp_input,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct& packed_quant_b,
    MLAS_THREADPOOL* ThreadPool
)
{
    std::byte* PackedQuantBDataBegin = packed_quant_b.PackedQuantBData;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
    const size_t kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
    const size_t sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();

    kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_dt_bf16;

    const size_t scales_len = N * K / BlkLen;
    uint16_t *scales = reinterpret_cast<uint16_t *>(alloca(scales_len * sizeof(uint16_t)));
    for (size_t i = 0; i < scales_len; i++) {
        scales[i] = kai_cast_bf16_f32(QuantBScaleBegin[i]);
    }

    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1, N, K, nr, kr, sr, BlkLen,
            reinterpret_cast<const uint8_t*>(QuantBDataBegin), K / 2,
            nullptr, scales, K * sizeof(uint16_t) / BlkLen,
            PackedQuantBDataBegin, 0, &params);
}

//
// Workspace size calculation function implementation.
//

size_t
SQ4BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            size_t mr;
            size_t kr;
            size_t sr;
            if (M == 1) {
                mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
                kr = kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
                sr = kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod();
            } else {
                mr = kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
                kr = kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
                sr = kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm();
            }
            return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
        }
        default: {
            return 0;
        }
    }
}

size_t
SQ4BitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

}  // namespace

}  // namespace sqnbitgemm_neon

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchNeon = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = sqnbitgemm_neon::SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBDataAndBlkSum = sqnbitgemm_neon::SQ4BitGemmPackQuantBDataAndBlkSum;

    d.SQ4BitGemmPerGemmWorkspaceSize = sqnbitgemm_neon::SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = sqnbitgemm_neon::SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = sqnbitgemm_neon::SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_neon::Q4BitBlkDequantBForSgemm_CompFp32;

    d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_neon::SQ4BitGemmKernel_CompInt8;
    d.QuantizeARow_CompInt8 = sqnbitgemm_neon::QuantizeARow_CompInt8;

    return d;
}();
