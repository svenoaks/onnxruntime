#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ONNX_CONFIG>"
  exit 1
fi

ONNX_CONFIG="$1"

./build.sh --config MinSizeRel \
--use_xcode \
--ios \
--apple_sysroot iphoneos \
--osx_arch arm64 \
--apple_deploy_target 16.0 \
--build_shared_lib \
--parallel \
--compile_no_warning_as_error \
--skip_tests \
--minimal_build \
--disable_ml_ops \
--include_ops_by_config "$ONNX_CONFIG" \
--enable_reduced_operator_type_support

           
