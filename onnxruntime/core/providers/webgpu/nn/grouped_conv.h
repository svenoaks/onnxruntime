// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

class GroupedConvProgram final : public Program<GroupedConvProgram> {
 public:
  GroupedConvProgram(const ConvAttributes& conv_attrs, bool has_bias, bool is_channels_last) : Program("GroupedConv"), conv_attrs_(conv_attrs), has_bias_(has_bias), is_channels_last_(is_channels_last) {
  }
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"output_channels_per_group", ProgramUniformVariableDataType::Uint32},
      {"components", ProgramUniformVariableDataType::Uint32});

 private:
  ConvAttributes conv_attrs_;
  bool has_bias_;
  bool is_channels_last_;
};

}  // namespace webgpu
}  // namespace onnxruntime
