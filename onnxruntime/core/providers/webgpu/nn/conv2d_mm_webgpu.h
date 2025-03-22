// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
class Conv2dMMProgram final : public Program<Conv2dMMProgram> {
 public:
  Conv2dMMProgram(const Activation& activation, uint32_t tile_a_outer, uint32_t tile_b_outer, uint32_t tile_inner, bool fit_a_outer, bool fit_b_outer, bool fit_inner, bool is_channels_last, bool is_vec4, bool has_bias, std::vector<uint32_t>&& element_size, InlinedVector<int64_t>&& elements_per_thread) : Program("Conv2dMM"),
                                                                                                                                                                                                                                                                                                                activation_(activation),
                                                                                                                                                                                                                                                                                                                tile_a_outer_(tile_a_outer),
                                                                                                                                                                                                                                                                                                                tile_b_outer_(tile_b_outer),
                                                                                                                                                                                                                                                                                                                tile_inner_(tile_inner),
                                                                                                                                                                                                                                                                                                                fit_a_outer_(fit_a_outer),
                                                                                                                                                                                                                                                                                                                fit_b_outer_(fit_b_outer),
                                                                                                                                                                                                                                                                                                                fit_inner_(fit_inner),
                                                                                                                                                                                                                                                                                                                is_channels_last_(is_channels_last),
                                                                                                                                                                                                                                                                                                                is_vec4_(is_vec4),
                                                                                                                                                                                                                                                                                                                has_bias_(has_bias),
                                                                                                                                                                                                                                                                                                                element_size_(std::move(element_size)),
                                                                                                                                                                                                                                                                                                                elements_per_thread_(std::move(elements_per_thread)) {}

  std::string Conv2dCommonSnippet(const Activation& activation, uint32_t inner_element_size_x = 4, uint32_t inner_element_size_w = 4, uint32_t inner_element_size = 4, std::string data_type = "f32") const;
  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dim_a_outer", ProgramUniformVariableDataType::Int32},
      {"dim_b_outer", ProgramUniformVariableDataType::Int32},
      {"dim_inner", ProgramUniformVariableDataType::Int32},
      {"pads", ProgramUniformVariableDataType::Int32},
      {"strides", ProgramUniformVariableDataType::Int32},
      {"dilations", ProgramUniformVariableDataType::Int32});

 private:
  const Activation& activation_;
  uint32_t tile_a_outer_;
  uint32_t tile_b_outer_;
  uint32_t tile_inner_;
  bool fit_a_outer_;
  bool fit_b_outer_;
  bool fit_inner_;
  bool is_channels_last_;
  bool is_vec4_;
  bool has_bias_;
  std::vector<uint32_t> element_size_;
  InlinedVector<int64_t> elements_per_thread_;
};

Conv2dMMProgram CreateConv2dMMProgram(const Activation& activation, const std::vector<const Tensor*>& inputs, std::vector<uint32_t> pads, std::vector<uint32_t> strides, std::vector<uint32_t> dilations, Tensor* output, uint32_t dim_a_outer, uint32_t dim_b_outer, uint32_t dim_inner, bool is_channels_last);

}  // namespace webgpu
}  // namespace onnxruntime
