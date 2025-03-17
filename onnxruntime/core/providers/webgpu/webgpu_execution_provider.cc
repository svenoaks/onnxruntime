// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_execution_provider.h"

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#endif

#include "allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/function_utils.h"
#include "core/graph/indexed_sub_graph.h"

#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/external_data_loader.h"
#include "core/providers/webgpu/webgpu_profiler.h"

namespace onnxruntime {

namespace webgpu {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    return Info().GetDataTransferManager().CopyTensor(*X, *Y);
  }
};

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, Op)>

#define KERNEL_CREATE_INFO_TYPED(Start, type, Op) \
  BuildKernelCreateInfo<                          \
      ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, type, Op)>

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Abs);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Abs);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Neg);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Neg);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Floor);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Floor);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Ceil);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Ceil);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Reciprocal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Reciprocal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Sqrt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Sqrt);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Exp);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Exp);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 12, Erf);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Erf);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Sigmoid);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Sigmoid);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, HardSigmoid);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Log);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Log);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, Sin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, Cos);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, Tan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, Asin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, Acos);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, Atan);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, Sinh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, Cosh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, Asinh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, Acosh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, Atanh);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Tanh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Tanh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, Not);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 8, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 12, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, Cast);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, Cast);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, float, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, float, Clip);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, Clip);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, Clip);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Clip);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, Elu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 15, LeakyRelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, LeakyRelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, ThresholdedRelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, Gelu);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMax);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMean);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceMean);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMean);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceMean);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMin);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMin);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMin);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceProd);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceProd);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceProd);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceProd);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ReduceSum);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL1);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL1);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL1);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL1);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL2);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL2);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL2);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL2);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSum);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSum);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSumSquare);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceSumSquare);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceSumSquare);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSumExp);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSumExp);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSumExp);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 12, Add);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Add);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, Add);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 12, Sub);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Sub);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, Sub);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 12, Mul);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Mul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, Mul);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 12, Div);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Div);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, Div);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 11, Pow);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, Pow);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 14, Pow);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, Pow);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 10, Equal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Equal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, Equal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, Equal);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, Greater);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 12, Greater);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Greater);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 15, GreaterOrEqual);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, GreaterOrEqual);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, Less);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 12, Less);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Less);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 15, LessOrEqual);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, LessOrEqual);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 14, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, 18, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Shape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Shape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Shape);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 18, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Reshape);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, Reshape);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Squeeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Squeeze);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Unsqueeze);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 15, Where);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, Where);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Transpose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Transpose);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, DepthToSpace);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, DepthToSpace);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, DepthToSpace);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, Conv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, Conv);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ConvTranspose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, ConvTranspose);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, ConvTranspose);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, ConvTranspose);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 7, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 8, 9, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 7, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 8, 9, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 9, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, AveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 9, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, AveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, AveragePool);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalAveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalAveragePool);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalMaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalMaxPool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MatMul);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, ArgMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, float, ArgMin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, ArgMin);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Softmax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Softmax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Softmax);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 3, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Concat);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 1, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Split);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, Split);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 8, 12, Expand);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Expand);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Resize);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, 17, Resize);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 18, 18, Resize);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 19, Resize);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Gather);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gather);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, GatherElements);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, GatherElements);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 9, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Slice);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Slice);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Slice);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Tile);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Tile);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 17, LayerNormalization);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, InstanceNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 6, InstanceNormalization);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, float, Range);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, int32_t, Range);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, float, Einsum);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Pad);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Pad);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, If);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, If);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 13, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 14, BatchNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 8, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 9, 13, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 14, 14, BatchNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 15, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 13, CumSum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, CumSum);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 12, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 12, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 12, int32_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, int32_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, int8_t, DequantizeLinear);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, int32_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, uint8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, int8_t, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, int32_t, DequantizeLinear);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

      // element-wise operators
      // unary - math
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Abs),
      KERNEL_CREATE_INFO(13, Abs),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Neg),
      KERNEL_CREATE_INFO(13, Neg),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Floor),
      KERNEL_CREATE_INFO(13, Floor),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Ceil),
      KERNEL_CREATE_INFO(13, Ceil),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Reciprocal),
      KERNEL_CREATE_INFO(13, Reciprocal),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Sqrt),
      KERNEL_CREATE_INFO(13, Sqrt),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Exp),
      KERNEL_CREATE_INFO(13, Exp),
      KERNEL_CREATE_INFO_VERSIONED(9, 12, Erf),
      KERNEL_CREATE_INFO(13, Erf),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Sigmoid),
      KERNEL_CREATE_INFO(13, Sigmoid),
      KERNEL_CREATE_INFO(6, HardSigmoid),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Log),
      KERNEL_CREATE_INFO(13, Log),

      KERNEL_CREATE_INFO(7, Sin),
      KERNEL_CREATE_INFO(7, Cos),
      KERNEL_CREATE_INFO(7, Tan),
      KERNEL_CREATE_INFO(7, Asin),
      KERNEL_CREATE_INFO(7, Acos),
      KERNEL_CREATE_INFO(7, Atan),
      KERNEL_CREATE_INFO(9, Sinh),
      KERNEL_CREATE_INFO(9, Cosh),
      KERNEL_CREATE_INFO(9, Asinh),
      KERNEL_CREATE_INFO(9, Acosh),
      KERNEL_CREATE_INFO(9, Atanh),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Tanh),
      KERNEL_CREATE_INFO(13, Tanh),
      KERNEL_CREATE_INFO(1, Not),

      KERNEL_CREATE_INFO_VERSIONED(6, 8, Cast),
      KERNEL_CREATE_INFO_VERSIONED(9, 12, Cast),
      KERNEL_CREATE_INFO_VERSIONED(13, 18, Cast),
      KERNEL_CREATE_INFO(19, Cast),

      // // activations
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, float, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, float, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Clip)>,
      KERNEL_CREATE_INFO(6, Elu),
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Relu),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Relu),
      KERNEL_CREATE_INFO(14, Relu),
      KERNEL_CREATE_INFO_VERSIONED(6, 15, LeakyRelu),
      KERNEL_CREATE_INFO(16, LeakyRelu),
      KERNEL_CREATE_INFO(10, ThresholdedRelu),
      KERNEL_CREATE_INFO(20, Gelu),

      // // binary - math
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Add),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Add),
      KERNEL_CREATE_INFO(14, Add),
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Sub),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Sub),
      KERNEL_CREATE_INFO(14, Sub),
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Mul),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Mul),
      KERNEL_CREATE_INFO(14, Mul),
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Div),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Div),
      KERNEL_CREATE_INFO(14, Div),
      KERNEL_CREATE_INFO_VERSIONED(7, 11, Pow),
      KERNEL_CREATE_INFO_VERSIONED(12, 12, Pow),
      KERNEL_CREATE_INFO_VERSIONED(13, 14, Pow),
      KERNEL_CREATE_INFO(15, Pow),
      KERNEL_CREATE_INFO_VERSIONED(7, 10, Equal),
      KERNEL_CREATE_INFO_VERSIONED(11, 12, Equal),
      KERNEL_CREATE_INFO_VERSIONED(13, 18, Equal),
      KERNEL_CREATE_INFO(19, Equal),
      KERNEL_CREATE_INFO_VERSIONED(7, 8, Greater),
      KERNEL_CREATE_INFO_VERSIONED(9, 12, Greater),
      KERNEL_CREATE_INFO(13, Greater),
      KERNEL_CREATE_INFO_VERSIONED(12, 15, GreaterOrEqual),
      KERNEL_CREATE_INFO(16, GreaterOrEqual),
      KERNEL_CREATE_INFO_VERSIONED(7, 8, Less),
      KERNEL_CREATE_INFO_VERSIONED(9, 12, Less),
      KERNEL_CREATE_INFO(13, Less),
      KERNEL_CREATE_INFO_VERSIONED(12, 15, LessOrEqual),
      KERNEL_CREATE_INFO(16, LessOrEqual),

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, 18, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Shape)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 5, 12, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 18, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, Reshape)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMax)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceMean)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Unsqueeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Unsqueeze)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMin)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceProd)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ReduceSum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL1)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL2)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceSumSquare)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceSumSquare)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSumExp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSumExp)>,

      KERNEL_CREATE_INFO_VERSIONED(9, 15, Where),
      KERNEL_CREATE_INFO(16, Where),

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Transpose)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, DepthToSpace)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, DepthToSpace)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, Conv)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, ConvTranspose)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 7, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 8, 9, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 7, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 8, 9, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 9, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 9, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, AveragePool)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalAveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalAveragePool)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalMaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalMaxPool)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, MatMul)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MatMul)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, ArgMax)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, float, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, ArgMin)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Softmax)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 3, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Concat)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 1, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Split)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, Split)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 8, 12, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Expand)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gather)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, GatherElements)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, GatherElements)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, 17, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 18, 18, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 19, Resize)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 9, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Slice)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, Flatten)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Tile)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 17, LayerNormalization)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, InstanceNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 6, InstanceNormalization)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, float, Range)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, int32_t, Range)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, float, Einsum)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Pad)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, If)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, If)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, If)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, If)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 13, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 14, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 8, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 9, 13, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 14, 14, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 15, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 13, CumSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, CumSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 12, uint8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 12, int8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 12, int32_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, uint8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, int8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, int32_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, uint8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, int8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, int32_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, uint8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, int8_t, DequantizeLinear)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, int32_t, DequantizeLinear)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

#ifndef DISABLE_CONTRIB_OPS
  Status status = ::onnxruntime::contrib::webgpu::RegisterWebGpuContribKernels(*kernel_registry);
  ORT_ENFORCE(status.IsOK(), "Failed to register WebGPU contrib kernels: " + status.ErrorMessage());
#endif

  return kernel_registry;
}

}  // namespace webgpu

using namespace webgpu;

WebGpuExecutionProvider::WebGpuExecutionProvider(int context_id,
                                                 WebGpuContext& context,
                                                 WebGpuExecutionProviderConfig&& config)
    : IExecutionProvider{kWebGpuExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)},
      context_id_{context_id},
      context_{context},
      preferred_data_layout_{config.data_layout},
      force_cpu_node_names_{std::move(config.force_cpu_node_names)},
      enable_graph_capture_{config.enable_graph_capture} {}

std::vector<AllocatorPtr> WebGpuExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo gpuBufferAllocatorCreationInfo([&](int) {
    return std::make_unique<webgpu::GpuBufferAllocator>(context_);
  },
                                                       0, false);
  auto preferred_allocators = std::vector<AllocatorPtr>{CreateAllocator(gpuBufferAllocatorCreationInfo)};
  allocator_ = reinterpret_cast<webgpu::GpuBufferAllocator*>(preferred_allocators[0].get());
  return preferred_allocators;
}

std::vector<std::unique_ptr<ComputeCapability>> WebGpuExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& /* graph_optimizer_registry */,
    IResourceAccountant* /* resource_accountant */) const {
  InlinedVector<NodeIndex> candidates;
  // `tenative_candidates` is a subset of `candidates`.
  InlinedVector<NodeIndex> tenative_candidates;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      // If the node was added by layout transformer, do not move it to CPU
      if (node.GetExecutionProviderType() == kWebGpuExecutionProvider) {
        candidates.push_back(node.Index());
      }
      continue;
    }

    const KernelCreateInfo* webgpu_kernel_def = kernel_lookup.LookUpKernel(node);
    // none of the provided registries has a webgpu kernel for this node
    if (webgpu_kernel_def == nullptr) {
      LOGS(*GetLogger(), INFO) << "webgpu kernel not found in registries for Op type: "
                               << node.OpType() << " node name: " << node.Name();
      continue;
    }

    // TODO: currently this lookup is O(N). If the list becomes large we should optimize this.
    if (std::find(force_cpu_node_names_.cbegin(),
                  force_cpu_node_names_.cend(),
                  node.Name()) != force_cpu_node_names_.cend()) {
      LOGS(*GetLogger(), INFO) << "Force CPU execution for node: " << node.Name();
      continue;
    }
    candidates.push_back(node.Index());
    tenative_candidates.push_back(node.Index());
  }

  auto cpu_nodes = GetCpuPreferredNodes(graph, kernel_lookup, tenative_candidates, *GetLogger());
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    if (cpu_nodes.count(node_index) > 0) {
      continue;
    }

    auto sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.emplace_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

std::shared_ptr<KernelRegistry> WebGpuExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = webgpu::RegisterKernels();

  return registry;
}

std::unique_ptr<onnxruntime::IDataTransfer> WebGpuExecutionProvider::GetDataTransfer() const {
  return std::make_unique<webgpu::DataTransfer>(context_);
}

#if defined(__wasm__)
std::unique_ptr<onnxruntime::IExternalDataLoader> WebGpuExecutionProvider::GetExternalDataLoader() const {
  return std::make_unique<webgpu::ExternalDataLoader>();
}
#endif

WebGpuExecutionProvider::~WebGpuExecutionProvider() {
  WebGpuContextFactory::ReleaseContext(context_id_);
}

std::unique_ptr<profiling::EpProfiler> WebGpuExecutionProvider::GetProfiler() {
  auto profiler = std::make_unique<WebGpuProfiler>(context_);
  profiler_ = profiler.get();
  return profiler;
}

Status WebGpuExecutionProvider::OnSessionInitializationEnd() {
  if (allocator_ != nullptr) {
    allocator_->OnSessionInitializationEnd();
  }
  return Status::OK();
}

Status WebGpuExecutionProvider::OnRunStart(const onnxruntime::RunOptions& /*run_options*/) {
  if (context_.ValidationMode() >= ValidationMode::Basic) {
    context_.PushErrorScope();
  }

  if (profiler_->Enabled()) {
    context_.StartProfiling();
  }

  if (IsGraphCaptureEnabled() && IsGraphCaptureAllowed() && !IsGraphCaptured(0)) {
    ORT_NOT_IMPLEMENTED("graph capture not implemented");
  }
  return Status::OK();
}

Status WebGpuExecutionProvider::OnRunEnd(bool /* sync_stream */, const onnxruntime::RunOptions& /*run_options*/) {
  if (IsGraphCaptureEnabled() && !IsGraphCaptured(0)) {
    if (IsGraphCaptureAllowed()) {
      ORT_NOT_IMPLEMENTED("graph capture not implemented");
      // is_graph_captured_ = true;
    } else {
      IncrementRegularRunCountBeforeGraphCapture();
    }
  }

  context_.Flush();

  if (profiler_->Enabled()) {
    context_.CollectProfilingData(profiler_->Events());
  }

  context_.OnRunEnd();

  if (context_.ValidationMode() >= ValidationMode::Basic) {
    return context_.PopErrorScope();
  } else {
    return Status::OK();
  }
}

bool WebGpuExecutionProvider::IsGraphCaptureEnabled() const {
  return enable_graph_capture_;
}

bool WebGpuExecutionProvider::IsGraphCaptured(int) const {
  return is_graph_captured_;
}

Status WebGpuExecutionProvider::ReplayGraph(int) {
  ORT_ENFORCE(IsGraphCaptured(0));
  ORT_ENFORCE(false);
  return Status::OK();
}

bool WebGpuExecutionProvider::IsGraphCaptureAllowed() const {
  return regular_run_count_before_graph_capture_ >= min_num_runs_before_cuda_graph_capture_;
}

void WebGpuExecutionProvider::IncrementRegularRunCountBeforeGraphCapture() {
  ++regular_run_count_before_graph_capture_;
}
}  // namespace onnxruntime
