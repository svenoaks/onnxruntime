
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/depth_to_space.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_DEPTH_TO_SPACE_VERSIONED_KERNEL(start, end, domain, is_nhwc) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                        \
      DepthToSpace,                                                         \
      domain,                                                               \
      start,                                                                \
      end,                                                                  \
      kWebGpuExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),                \
      DepthToSpace<is_nhwc>);

#define WEBGPU_DEPTH_TO_SPACE_KERNEL(version, domain, is_nhwc) \
  ONNX_OPERATOR_KERNEL_EX(                                     \
      DepthToSpace,                                            \
      domain,                                                  \
      version,                                                 \
      kWebGpuExecutionProvider,                                \
      (*KernelDefBuilder::Create())                            \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),   \
      DepthToSpace<is_nhwc>);

WEBGPU_DEPTH_TO_SPACE_VERSIONED_KERNEL(11, 12, kOnnxDomain, false)
WEBGPU_DEPTH_TO_SPACE_KERNEL(13, kOnnxDomain, false)

WEBGPU_DEPTH_TO_SPACE_VERSIONED_KERNEL(11, 12, kMSInternalNHWCDomain, true)
WEBGPU_DEPTH_TO_SPACE_KERNEL(13, kMSInternalNHWCDomain, true)

int64_t GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

void AppendPermFunction(std::ostream& os, const ShaderVariableHelper& input, int64_t* perm) {
  os << "fn perm(i: input_indices_t) -> input_indices_t {\n"
     << "  var a: input_indices_t;\n";
  for (int i = 0; i < input.Rank(); ++i) {
    os << "  " << input.IndicesSet("a", std::to_string(perm[i]), "i[" + std::to_string(i) + "]);\n");
  }
  os << "  return a;\n"
     << "}\n";
}

Status DepthToSpaceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input");
  const ShaderVariableHelper& output = shader.AddOutput("output");

  AppendPermFunction(shader.AdditionalImplementation(), input, perm_);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "let aIndices = perm(indices);\n"
                            << output.SetByOffset("global_idx", input.GetByOffset("aIndices")) << ";\n";

  return Status::OK();
}

template <bool is_nchw>
Status DepthToSpace<is_nchw>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const TensorShape input_shape = input->Shape();

  int64_t n, c, h, w;
  int64_t shape[6];
  int64_t perm[6];
  if (is_nchw) {
    n = input_shape[0];
    c = input_shape[1];
    h = input_shape[2];
    w = input_shape[3];

    if (is_dcr_) {
      shape[0] = n;
      shape[1] = blocksize_;
      shape[2] = blocksize_;
      shape[3] = c / (blocksize_ * blocksize_);
      shape[4] = h;
      shape[5] = w;

      perm[0] = 0;
      perm[1] = 3;
      perm[2] = 4;
      perm[3] = 1;
      perm[4] = 5;
      perm[5] = 2;
    } else {
      shape[0] = n;
      shape[1] = c / (blocksize_ * blocksize_);
      shape[2] = blocksize_;
      shape[3] = blocksize_;
      shape[4] = h;
      shape[5] = w;

      perm[0] = 0;
      perm[1] = 1;
      perm[2] = 4;
      perm[3] = 2;
      perm[4] = 5;
      perm[5] = 3;
    }
  } else {
    n = input_shape[0];
    h = input_shape[1];
    w = input_shape[2];
    c = input_shape[3];

    if (is_dcr_) {
      shape[0] = n;
      shape[1] = h;
      shape[2] = w;
      shape[3] = blocksize_;
      shape[4] = blocksize_;
      shape[5] = c / (blocksize_ * blocksize_);

      perm[0] = 0;
      perm[1] = 1;
      perm[2] = 3;
      perm[3] = 2;
      perm[4] = 4;
      perm[5] = 5;
    } else {
      shape[0] = n;
      shape[1] = h;
      shape[2] = w;
      shape[3] = c / (blocksize_ * blocksize_);
      shape[4] = blocksize_;
      shape[5] = blocksize_;

      perm[0] = 0;
      perm[1] = 1;
      perm[2] = 4;
      perm[3] = 2;
      perm[4] = 5;
      perm[5] = 3;
    }
  }

  TensorShape override_shape(gsl::make_span(shape));
  int64_t components = GetMaxComponents(c);

  // Calculate the final 4D output shape
  int64_t output_shape[4];
  if (is_nchw) {
    output_shape[0] = n;
    output_shape[1] = c / (blocksize_ * blocksize_);
    output_shape[2] = h * blocksize_;
    output_shape[3] = w * blocksize_;
  } else {
    output_shape[0] = n;
    output_shape[1] = h * blocksize_;
    output_shape[2] = w * blocksize_;
    output_shape[3] = c / (blocksize_ * blocksize_);
  }
  TensorShape final_output_shape(gsl::make_span(output_shape, 4));

  auto* output = context.Output(0, final_output_shape);

  // Map from the reshaped 6D input to 4D output
  TensorShape output_reshape(gsl::make_span(shape, 6));
  int64_t output_size = output->Shape().Size() / components;

  if (output_size == 0) {
    return Status::OK();
  }

  DepthToSpaceProgram program{perm};
  program
      .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, override_shape, static_cast<int>(components)})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, output_reshape, static_cast<int>(components)})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariable({static_cast<uint32_t>(output_size)});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
