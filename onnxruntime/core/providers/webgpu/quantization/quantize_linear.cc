// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/util/math.h"
#include "core/providers/webgpu/quantization/quantize_linear.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {
const std::vector<MLDataType>& DequantizeLinearConstraints() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<int32_t>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    10, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    13, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    19, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);

ONNX_OPERATOR_KERNEL_EX(
    DequantizeLinear,
    kOnnxDomain,
    23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DequantizeLinearConstraints())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    DequantizeLinear);


Status DequantizeLinearProgram::GenerateShaderCode(ShaderHelper& shader) const {
  return Status::OK();
}

Status DequantizeLinear::ComputeInternal(ComputeContext& context) const {
  DequantizeLinearProgram program{axis_, block_size_};
  /*
  const Tensor* input_tensor = context.Input<Tensor>(0);
  auto const& input_shape = input_tensor->Shape();


  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(mode_)), dim_value_zero)
      .AddUniformVariables({{gsl::span<const int32_t>(lower_DequantizeLinears.data(), lower_DequantizeLinears.size())}, {output_size}, {value_uint32}});
*/
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
