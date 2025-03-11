// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/reduction/reduction_kernel_base.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
namespace onnxruntime {
namespace webgpu {
// reduceOpSpecificCode is a 3-element array of strings that represent the op specific code for the reduce operation.
// The first element is the loop header, the second element is the loop body, and the third element is the loop footer.
// The loop header is the code that is executed before the loop starts. The loop body is the code that is executed for each element in the loop.
// The loop footer is the code that is executed after the loop ends.
typedef std::array<std::string, 3> ReduceOpSpecificCode;
class ReduceKernelProgram final : public Program<ReduceKernelProgram> {
 public:
  ReduceKernelProgram(std::string name, bool keepdims, bool no_op_with_empty_axes, const InlinedVector<uint32_t>& axes, ReduceOpSpecificCode code) : Program{name}, keepdims_(keepdims), no_op_with_empty_axes_(no_op_with_empty_axes), axes_(axes.begin(), axes.end()), code_(code) {}
  Status GenerateShaderCode(ShaderHelper& wgpuShaderModuleAddRef) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"no_op_with_empty_axes", ProgramUniformVariableDataType::Uint32},
                                          {"axes", ProgramUniformVariableDataType::Uint32},
                                          {"axes_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool keepdims_;
  const bool no_op_with_empty_axes_;
  InlinedVector<uint32_t> axes_;
  ReduceOpSpecificCode code_;
};

template <bool allow_multi_axes = true>
class ReduceKernel : public WebGpuKernel, public ReduceKernelBase<allow_multi_axes> {
 protected:
  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  using ReduceKernelBase<allow_multi_axes>::select_last_index_;

  ReduceKernel(const OpKernelInfo& info, std::string name, optional<int64_t> keepdims_override = {})
      : WebGpuKernel(info),
        ReduceKernelBase<allow_multi_axes>(info, keepdims_override),
        name_(name) {
  }
  Status ComputeInternal(ComputeContext& ctx) const;
  virtual ReduceOpSpecificCode GetOpSpecificCode(const Tensor* input_tensor, size_t axes_size) const = 0;

 private:
  std::string name_;
};

class ReduceMean final : public ReduceKernel<true> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info, "ReduceMean") {}
  ReduceOpSpecificCode GetOpSpecificCode(const Tensor* input_tensor, size_t axes_size) const override;
  Status ComputeInternal(ComputeContext& ctx) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
