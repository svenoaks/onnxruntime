// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/math/unary_elementwise_ops.cc"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    Gelu);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime