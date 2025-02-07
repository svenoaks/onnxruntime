# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set (CSHARP_ROOT ${PROJECT_SOURCE_DIR}/../csharp)

if (onnxruntime_USE_CUDA)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_CUDA;")
endif()

if (onnxruntime_USE_DNNL)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_DNNL;")
endif()

if (onnxruntime_USE_DML)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_DML;")
endif()

if (onnxruntime_USE_MIGRAPHX)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_MIGRAPHX;")
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_NNAPI;")
endif()

if (onnxruntime_USE_OPENVINO)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_OPENVINO;")
endif()

if (onnxruntime_USE_ROCM)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_ROCM;")
endif()

if (onnxruntime_USE_TENSORRT)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_TENSORRT;")
endif()

if (onnxruntime_USE_XNNPACK)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_XNNPACK;")
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "__TRAINING_ENABLED_NATIVE_BUILD__;")
endif()

# generate Directory.Build.props
set(DIRECTORY_BUILD_PROPS_COMMENT "WARNING: This is a generated file, please do not check it in!")
configure_file(${CSHARP_ROOT}/Directory.Build.props.in
               ${CSHARP_ROOT}/Directory.Build.props
               @ONLY)
