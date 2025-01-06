// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    1, 9,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    Slice);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    10, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).InputMemoryType(OrtMemTypeCPU, 1).InputMemoryType(OrtMemTypeCPU, 2).InputMemoryType(OrtMemTypeCPU, 3).InputMemoryType(OrtMemTypeCPU, 4),
    Slice);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).InputMemoryType(OrtMemTypeCPU, 1).InputMemoryType(OrtMemTypeCPU, 2).InputMemoryType(OrtMemTypeCPU, 3).InputMemoryType(OrtMemTypeCPU, 4),
    Slice);

ONNX_OPERATOR_KERNEL_EX(
    Slice,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()).InputMemoryType(OrtMemTypeCPU, 1).InputMemoryType(OrtMemTypeCPU, 2).InputMemoryType(OrtMemTypeCPU, 3).InputMemoryType(OrtMemTypeCPU, 4),
    Slice);

Status SliceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  std::cout << "generate shader code" << std::endl;
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var input_indices: input_indices_t;\n"
                            << "var carry = 0u;\n";

  for (auto i = input.Rank() - 1; i >= 0; i--) {
    std::string input_shape_i = absl::StrCat("input_shape_", i);
    std::string steps_i = absl::StrCat("steps_", i);
    std::string starts_i = absl::StrCat("starts_", i);
    std::string output_index_i = absl::StrCat("output_index_", i);
    std::string input_index_i = absl::StrCat("input_index_", i);

    shader.MainFunctionBody() << "let " << input_shape_i << " = " << input.IndicesGet("uniforms.input_shape", i) << ";\n"
                              << "let " << steps_i << " = " << input.IndicesGet("uniforms.steps", i) << ";\n"
                              << "let " << starts_i << " = " << input.IndicesGet("uniforms.starts", i) << ";\n"
                              << "var " << output_index_i << " = " << output.IndicesGet("output_indices", i) << ";\n"
                              << "var " << input_index_i << " = " << output_index_i << " * " << steps_i << " + " << starts_i << " + carry;\n"
                              << "carry = " << input_index_i << " / " << input_shape_i << ";\n"
                              << input_index_i << " = " << input_index_i << " % " << input_shape_i << ";\n"
                              << "if (" << input.IndicesGet("uniforms.signs", i) << " < 0) {\n"
                              << "  " << input_index_i << " = " << input_shape_i << " - " << input_index_i << " - 1u + " << starts_i << ";\n"
                              << "}\n"
                              << input.IndicesSet("input_indices", i, input_index_i) << ";\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", input.GetByIndices("input_indices"));

  std::cout << "shader code generated" << std::endl;
  return Status::OK();
}

Status Slice::ComputeInternal(ComputeContext& context) const {
  // READ INPUTS
  std::cout << "read input" << std::endl;
  const Tensor* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int64_t input_rank = static_cast<int64_t>(input_shape.NumDimensions());

  std::cout << "read starts/ends from either attr or input" << std::endl;

  auto starts_raw = hasStartsAttr ? gsl::make_span(attr_starts_) : context.Input(1)->DataAsSpan<int64_t>();
  auto ends_raw = hasEndsAttr ? gsl::make_span(attr_ends_) : context.Input(2)->DataAsSpan<int64_t>();

  ORT_ENFORCE(starts_raw.size() == ends_raw.size(), "starts and ends must have the same size");

  int input_count = context.InputCount();

  const Tensor* axes_tensor = nullptr;
  const Tensor* steps_tensor = nullptr;

  std::cout << "read axes and steps from input" << std::endl;

  if (input_count >= 4) {
    // axes provided as input
    axes_tensor = context.Input(3);
  }

  if (input_count == 5) {
    // steps provided as input
    steps_tensor = context.Input(4);
  }

  std::cout << "inject defaults if axes or steps not provided" << std::endl;

  std::vector<int64_t> axes_default;
  if (axes_tensor == nullptr) {
    // if axes not provided, set to [0, ..., len(starts)-1]
    for (size_t i = 0; i < starts_raw.size(); i++) {
      axes_default.push_back(i);
    }
  }
  auto axes_raw = hasAxesAttr ? gsl::make_span(attr_axes_) : (axes_tensor == nullptr ? gsl::make_span(axes_default) : axes_tensor->DataAsSpan<int64_t>());

  std::vector<int64_t> steps_default;
  if (steps_tensor == nullptr) {
    // if steps not provided, set to [1, ..., 1] of len(starts)
    for (size_t i = 0; i < starts_raw.size(); i++) {
      steps_default.push_back(1);
    }
  }
  auto steps_raw = steps_tensor == nullptr ? gsl::make_span(steps_default) : steps_tensor->DataAsSpan<int64_t>();

  std::cout << "ORIGINAL INPUTS" << std::endl;
  std::cout << "input shape: " << input_shape << std::endl;
  std::cout << "starts: ";
  for (auto start : starts_raw) {
    std::cout << start << " ";
  }
  std::cout << std::endl;
  std::cout << "ends: ";
  for (auto end : ends_raw) {
    std::cout << end << " ";
  }
  std::cout << std::endl;
  std::cout << "axes: ";
  for (auto axis : axes_raw) {
    std::cout << axis << " ";
  }
  std::cout << std::endl;
  std::cout << "steps: ";
  for (auto step : steps_raw) {
    std::cout << step << " ";
  }
  std::cout << std::endl;

  // PROCESS INPUTS
  std::cout << "processing inputs" << std::endl;
  std::cout << "process axes" << std::endl;

  std::vector<uint32_t> axes;
  for (unsigned int i = 0; i < axes_raw.size(); i++) {
    int64_t val = axes_raw[i];
    if (val < 0) {
      val += input_rank;
    }
    axes.push_back(static_cast<int32_t>(val));
  }

  std::cout << "process starts" << std::endl;
  std::vector<uint32_t> starts;
  for (unsigned int i = 0; i < starts_raw.size(); i++) {
    int64_t val = starts_raw[i];
    std::cout << "val: " << val << std::endl;
    if (val < 0) {
      val += input_shape[axes[i]];
    }
    std::cout << "val after handling negative: " << val << std::endl;

    std::cout << "steps raw i: " << steps_raw[i] << std::endl;
    if (steps_raw[i] < 0) {
      std::cout << "steps raw < 0" << std::endl;
      std::cout << "axes raw i: " << axes[i] << std::endl;
      std::cout << "input shape axes raw i: " << input_shape[axes[i]] << std::endl;
      val = std::max(static_cast<int64_t>(0), std::min(val, static_cast<int64_t>(input_shape[axes[i]] - 1)));
    } else {
      std::cout << "steps raw >= 0" << std::endl;
      std::cout << "axes raw i: " << axes[i] << std::endl;
      std::cout << "input shape axes raw i: " << input_shape[axes[i]] << std::endl;
      val = std::max(static_cast<int64_t>(0), std::min(val, static_cast<int64_t>(input_shape[axes[i]])));
    }
    std::cout << "val after clamping: " << val << std::endl;
    starts.push_back(static_cast<uint32_t>(val));
  }

  std::cout << "process ends" << std::endl;

  std::vector<uint32_t> ends;
  for (unsigned int i = 0; i < ends_raw.size(); i++) {
    int64_t val = ends_raw[i];
    if (val < 0) {
      val += input_shape[axes[i]];
    }
    if (steps_raw[i] < 0) {
      val = std::max(static_cast<int64_t>(0), std::min(val, static_cast<int64_t>(input_shape[axes[i]] - 1)));
    } else {
      val = std::max(static_cast<int64_t>(0), std::min(val, static_cast<int64_t>(input_shape[axes[i]])));
    }
    ends.push_back(static_cast<uint32_t>(val));
  }

  std::cout << "process steps with INT_MAX" << std::endl;

  // temporary steps vector to handle negative steps
  std::vector<int32_t> steps_tmp;
  for (unsigned int i = 0; i < steps_raw.size(); i++) {
    if (steps_raw[i] >= std::numeric_limits<int32_t>::max()) {
      steps_tmp.push_back(std::numeric_limits<int32_t>::max());
    } else {
      steps_tmp.push_back(static_cast<int32_t>(steps_raw[i]));
    }
  }

  std::cout << "insert missing dimensions" << std::endl;

  if (static_cast<int64_t>(axes.size()) != input_rank) {
    for (uint32_t i = 0; i < input_rank; i++) {
      int idx = -1;
      for (unsigned int j = 0; j < axes_raw.size(); j++) {
        if (axes_raw[j] == i) {
          idx = j;
          break;
        }
      }
      if (idx == -1) {
        axes.insert(axes.begin() + i, i);
        starts.insert(starts.begin() + i, 0);
        ends.insert(ends.begin() + i, static_cast<uint32_t>(input_shape[i]));
        steps_tmp.insert(steps_tmp.begin() + i, 1);
      }
    }
  }

  std::cout << "retain the sign of the steps" << std::endl;

  // retain the sign of the steps
  std::vector<int32_t> signs;
  for (unsigned int i = 0; i < steps_tmp.size(); i++) {
    signs.push_back(steps_tmp[i] < 0 ? -1 : (steps_tmp[i] > 0 ? 1 : 0));
  }

  std::cout << "convert negative steps to positive steps and reverse starts and ends" << std::endl;

  // Convert negative steps to positive steps and reverse starts and ends
  for (unsigned int i = 0; i < steps_tmp.size(); i++) {
    if (steps_tmp[i] < 0) {
      float numSteps = static_cast<float>((static_cast<float>(ends[i]) - static_cast<float>(starts[i])) / static_cast<float>(steps_tmp[i]));
      float newEnd = static_cast<float>(starts[i]);
      float newStart = newEnd + numSteps * static_cast<float>(steps_tmp[i]);

      starts[i] = static_cast<uint32_t>(newStart);
      ends[i] = static_cast<uint32_t>(newEnd);
      steps_tmp[i] = static_cast<int32_t>(-steps_tmp[i]);
    }
  }

  std::cout << "final steps vector" << std::endl;

  // final steps vector of type unsigned int
  std::vector<uint32_t> steps;
  for (unsigned int i = 0; i < steps_tmp.size(); i++) {
    steps.push_back(static_cast<uint32_t>(steps_tmp[i]));
  }

  std::cout << "PROCESSED INPUTS" << std::endl;
  std::cout << "starts: ";
  for (auto start : starts) {
    std::cout << start << " ";
  }
  std::cout << std::endl;
  std::cout << "ends: ";
  for (auto end : ends) {
    std::cout << end << " ";
  }
  std::cout << std::endl;
  std::cout << "axes: ";
  for (auto axis : axes) {
    std::cout << axis << " ";
  }
  std::cout << std::endl;
  std::cout << "steps: ";
  for (auto step : steps) {
    std::cout << step << " ";
  }
  std::cout << std::endl;

  std::cout << "reorder inputs in order of axis" << std::endl;

  std::vector<int32_t> signs_reordered;
  std::vector<uint32_t> steps_reordered, starts_reordered;
  for (unsigned int i = 0; i < axes.size(); i++) {
    signs_reordered.push_back(0);
    steps_reordered.push_back(0);
    starts_reordered.push_back(0);
  }
  for (unsigned int i = 0; i < axes.size(); i++) {
    int32_t dim = axes[i];
    signs_reordered[dim] = signs[i];
    steps_reordered[dim] = steps[i];
    starts_reordered[dim] = starts[i];
  }

  std::cout << "REORDERED INPUTS" << std::endl;
  std::cout << "signs_reordered: ";
  for (auto sign : signs_reordered) {
    std::cout << sign << " ";
  }
  std::cout << std::endl;
  std::cout << "steps_reordered: ";
  for (auto step : steps_reordered) {
    std::cout << step << " ";
  }
  std::cout << std::endl;
  std::cout << "starts_reordered: ";
  for (auto start : starts_reordered) {
    std::cout << start << " ";
  }
  std::cout << std::endl;

  std::cout << "calculate output dims" << std::endl;

  // calculate output dims
  std::vector<int64_t> output_dims;
  for (unsigned int i = 0; i < axes.size(); i++) {
    int32_t dim = axes[i];
    float tmp = ceil((static_cast<float>(ends[dim]) - static_cast<float>(starts[dim])) / static_cast<float>(steps[dim]));
    if (tmp < 0)
      output_dims.push_back(0);
    else
      output_dims.push_back(static_cast<int64_t>(tmp));
  }

  TensorShape output_shape(output_dims);

  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = static_cast<uint32_t>(output_shape.Size());

  if (output_size == 0) {
    std::cout << "output size is 0" << std::endl;
    return Status::OK();
  }

  std::cout << "run program" << std::endl;

  SliceProgram program{};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{output_size}, {starts_reordered}, {steps_reordered}, {signs_reordered}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime