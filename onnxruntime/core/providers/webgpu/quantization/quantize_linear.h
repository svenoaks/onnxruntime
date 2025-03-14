// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class DequantizeLinearProgram final : public Program<DequantizeLinearProgram> {
 public:
  DequantizeLinearProgram(const int64_t axis, const int64_t block_size) : Program<DequantizeLinearProgram>{"DequantizeLinear"},
                                                                          axis_{axis},
                                                                          block_size_{block_size} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"axis", ProgramUniformVariableDataType::Int32},
                                          {"block_size", ProgramUniformVariableDataType::Int32});

 private:
  int64_t axis_;
  int64_t block_size_;
};

class DequantizeLinear final : public WebGpuKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : WebGpuKernel(info) {
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
};

}  // namespace webgpu
}  // namespace onnxruntime
