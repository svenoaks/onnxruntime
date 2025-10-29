#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ONNX_CONFIG> [ABI]"
  echo "  ABI options: arm64-v8a, armeabi-v7a, x86, x86_64, all"
  echo "  Default: arm64-v8a"
  exit 1
fi

ONNX_CONFIG="$1"
ABI="${2:-arm64-v8a}"

ANDROID_SDK_ROOT="/Users/steve/Library/Android/sdk"
ANDROID_NDK_PATH="/Users/steve/Library/Android/sdk/ndk/28.2.13676358"

build_for_abi() {
  local abi=$1
  local abi_build_dir="build/Android_${abi}"
  
  echo "========================================"
  echo "Building for ABI: $abi"
  echo "Build directory: $abi_build_dir"
  echo "========================================"
  
  mamba run -n demucsonnx ./build.sh --config Release \
  --build_dir "$abi_build_dir" \
  --android \
  --android_sdk_path "$ANDROID_SDK_ROOT" \
  --android_ndk_path "$ANDROID_NDK_PATH" \
  --android_abi "$abi" \
  --android_api 21 \
  --android_cpp_shared \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_tests \
  --minimal_build \
  --disable_ml_ops \
  --include_ops_by_config "$ONNX_CONFIG" \
  --enable_reduced_operator_type_support \
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5
  
  if [ $? -eq 0 ]; then
    echo "✓ Successfully built for $abi"
    echo "  Output: $abi_build_dir/Release/libonnxruntime.so"
  else
    echo "✗ Failed to build for $abi"
    return 1
  fi
}

if [ "$ABI" == "all" ]; then
  echo "Building for all ABIs: arm64-v8a, armeabi-v7a, x86, x86_64"
  for abi in arm64-v8a armeabi-v7a x86 x86_64; do
    build_for_abi "$abi" || exit 1
  done
  echo ""
  echo "All builds completed successfully!"
  echo "Output files:"
  for abi in arm64-v8a armeabi-v7a x86 x86_64; do
    so_file="build/Android_${abi}/Release/libonnxruntime.so"
    if [ -f "$so_file" ]; then
      echo "  $abi:"
      ls -lh "$so_file"
      file "$so_file"
    fi
  done
else
  build_for_abi "$ABI"
fi
  

  