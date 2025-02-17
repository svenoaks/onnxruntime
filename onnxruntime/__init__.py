# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""

__version__ = "1.21.0"
__author__ = "Microsoft"

# we need to do device version validation (for example to check Cuda version for an onnxruntime-training package).
# in order to know whether the onnxruntime package is for training it needs
# to do import onnxruntime.training.ortmodule first.
# onnxruntime.capi._pybind_state is required before import onnxruntime.training.ortmodule.
# however, import onnxruntime.capi._pybind_state will already raise an exception if a required Cuda version
# is not found.
# here we need to save the exception and continue with Cuda version validation in order to post
# meaningful messages to the user.
# the saved exception is raised after device version validation.
try:
    from onnxruntime.capi._pybind_state import (
        ExecutionMode,  # noqa: F401
        ExecutionOrder,  # noqa: F401
        GraphOptimizationLevel,  # noqa: F401
        LoraAdapter,  # noqa: F401
        ModelMetadata,  # noqa: F401
        NodeArg,  # noqa: F401
        OrtAllocatorType,  # noqa: F401
        OrtArenaCfg,  # noqa: F401
        OrtMemoryInfo,  # noqa: F401
        OrtMemType,  # noqa: F401
        OrtSparseFormat,  # noqa: F401
        RunOptions,  # noqa: F401
        SessionIOBinding,  # noqa: F401
        SessionOptions,  # noqa: F401
        create_and_register_allocator,  # noqa: F401
        create_and_register_allocator_v2,  # noqa: F401
        disable_telemetry_events,  # noqa: F401
        enable_telemetry_events,  # noqa: F401
        get_all_providers,  # noqa: F401
        get_available_providers,  # noqa: F401
        get_build_info,  # noqa: F401
        get_device,  # noqa: F401
        get_version_string,  # noqa: F401
        has_collective_ops,  # noqa: F401
        set_default_logger_severity,  # noqa: F401
        set_default_logger_verbosity,  # noqa: F401
        set_seed,  # noqa: F401
    )

    import_capi_exception = None
except Exception as e:
    import_capi_exception = e

from onnxruntime.capi import onnxruntime_validation

if import_capi_exception:
    raise import_capi_exception

from onnxruntime.capi.onnxruntime_inference_collection import (
    AdapterFormat,  # noqa: F401
    InferenceSession,  # noqa: F401
    IOBinding,  # noqa: F401
    OrtDevice,  # noqa: F401
    OrtValue,  # noqa: F401
    SparseTensor,  # noqa: F401
)

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:  # noqa: SIM105
    from . import experimental  # noqa: F401
except ImportError:
    pass


package_name, version, cuda_version = onnxruntime_validation.get_package_name_and_version_info()

if version:
    __version__ = version

onnxruntime_validation.check_distro_info()


def preload_dlls(cuda: bool = True, cudnn: bool = True, msvc: bool = True, torch_safe:bool = True, verbose: bool = False):
    """Preload CUDA 12.x and cuDNN 9.x DLLs in Windows or Linux, and MSVC runtime DLLs in Windows.

       When you installed PyTorch that is compatible with CUDA 12.x, set `torch_safe` to True is recommended.
       Note that there is no need to call this function if `import torch` is done before `import onnxruntime`.

    Args:
        cuda (bool, optional): enable loading CUDA DLLs. Defaults to True.
        cudnn (bool, optional): enable loading cuDNN DLLs. Defaults to True.
        msvc (bool, optional): enable loading MSVC DLLs in Windows. Defaults to True.
        torch_safe (bool, optional): enable loading CUDA and cuDNN DLLs from PyTorch in Windows. Defaults to True.
        verbose (bool, optional): allow printing more information to console for debugging purpose. Defaults to False.
    """
    import ctypes
    import os
    import sys
    import platform
    if platform.system() not in ["Windows", "Linux"]:
        return

    is_windows = platform.system() == "Windows"
    if is_windows and msvc:
        try:
            ctypes.CDLL("vcruntime140.dll")
            ctypes.CDLL("msvcp140.dll")
            if platform.machine() != "ARM64":
                ctypes.CDLL("vcruntime140_1.dll")
        except OSError:
            print("Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.")
            print("It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe.")


    def get_package_version(package:str):
        from importlib.metadata import version, PackageNotFoundError
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = None
        return package_version

    if verbose:
        if cuda_version:
            # Print version of installed packages that is related to CUDA or cuDNN DLLs.
            packages = ["torch",
                        "nvidia-cuda-runtime-cu12",
                        "nvidia-cudnn-cu12",
                        "nvidia-cublas-cu12",
                        "nvidia-cufft-cu12",
                        "nvidia-curand-cu12",
                        "nvidia-cuda-nvrtc-cu12",
                        "nvidia-nvjitlink-cu12"]
            print("Related packages and installed version:")
            for package in packages:
                print(f"\t{package}: {get_package_version(package)}")

        # List onnxruntime* packages to help identify that multiple onnxruntime packages were installed.
        from importlib.metadata import distributions
        for dist in distributions():
            if dist.metadata['Name'].startswith('onnxruntime'):
                print(f"\t{dist.metadata['Name']}=={dist.version}")

    if not (cuda_version and cuda_version.startswith("12.")) and (cuda or cudnn):
        print(f"\033[33mWARNING: {package_name} is not built with CUDA 12.x support. "
              "Please install a version that support it, or call preload_dlls with cuda=False and cudnn=False.\033[0m")

    if cuda_version and cuda_version.startswith("12.") and (cuda or cudnn):
        torch_version = get_package_version("torch")
        if torch_safe and torch_version and "+cu12" not in torch_version:
            print(f"\033[33mWARNING: the installed PyTorch {torch_version} does not support CUDA 12.x. "
                  f"Please install PyTorch for CUDA 12.x to be compatible with {package_name}.\033[0m")

        if is_windows:
            if 'torch' in sys.modules and not torch_safe and torch_version and "+cu" in torch_version and (cuda or cudnn):
                print("\033[33mWARNING: onnxruntime.preload_dlls is called with torch_safe=False when torch has been imported.")
                print("Either set torch_safe=False, or not call preload_dlls in this case.\033[0m")

            # In torch_safe mode, we will load same cuda and cudnn DLLs that used by PyTorch.
            if torch_safe and torch_version and "+cu12" in torch_version:
                # All paths are relative to the site-packages directory.
                cuda_dll_paths = [
                    ("torch", "lib", "cublasLt64_12.dll"),
                    ("torch", "lib", "cublas64_12.dll"),
                    ("torch", "lib", "cufft64_11.dll"),
                    ("torch", "lib", "cudart64_12.dll"),
                ]
                # Some cudnn DLLs are loaded dynamically. It is better to add all DLLs used by ORT to this list.
                # See https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/api/overview.html
                cudnn_dll_paths = [
                    ("torch", "lib", "cudnn_graph64_9.dll"),
                    ("torch", "lib", "cudnn64_9.dll"),
                ]
            else:
                cuda_dll_paths = [
                    ("nvidia", "cublas", "bin", "cublasLt64_12.dll"),
                    ("nvidia", "cublas", "bin", "cublas64_12.dll"),
                    ("nvidia", "cufft", "bin", "cufft64_11.dll"),
                    ("nvidia", "cuda_runtime", "bin", "cudart64_12.dll"),
                ]
                cudnn_dll_paths = [
                    ("nvidia", "cudnn", "bin", "cudnn_graph64_9.dll"),
                    ("nvidia", "cudnn", "bin", "cudnn64_9.dll"),
                ]
        else:  # Linux
            # cublas64 depends on cublasLt64, so cublasLt64 should be loaded first.
            cuda_dll_paths = [
                ("nvidia", "cublas", "lib", "libcublasLt.so.12"),
                ("nvidia", "cublas", "lib", "libcublas.so.12"),
                ("nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12"),
                ("nvidia", "curand", "lib", "libcurand.so.10"),
                ("nvidia", "cufft", "lib", "libcufft.so.11"),
                ("nvidia", "cuda_runtime", "lib", "libcudart.so.12"),
            ]
            cudnn_dll_paths = [
                ("nvidia", "cudnn", "lib", "libcudnn_graph.so.9"),
                ("nvidia", "cudnn", "lib", "libcudnn.so.9"),
            ]

        # Try load DLLs from site packages.
        site_packages_path = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        dll_paths = (cuda_dll_paths if cuda else []) + (cudnn_dll_paths if cudnn else [])
        loaded_dlls = []
        for relative_path in dll_paths:
            dll_path = os.path.join(site_packages_path, *relative_path)
            if os.path.isfile(dll_path):
                try:
                    _ = ctypes.CDLL(dll_path)
                    loaded_dlls.append(relative_path[-1])
                    # Add DLL directory to search path for cuDNN.
                    if relative_path[-1] in ["cudnn64_9.dll"]:
                        os.add_dll_directory(os.path.dirname(dll_path))
                except Exception as e:
                    print(f"Failed to load {dll_path}: {e}")

        # Try load DLLs with default path settings.
        has_failure = False
        for relative_path in dll_paths:
            dll_filename = relative_path[-1]
            if dll_filename not in loaded_dlls:
                try:
                    _ = ctypes.CDLL(dll_filename)
                except Exception as e:
                    has_failure = True
                    print(f"Failed to load {dll_filename}: {e}")

        if has_failure:
            print("Please follow https://onnxruntime.ai/docs/install/#cuda-and-cudnn to install CUDA and CuDNN.")

    if verbose:

        def is_target_dll(path: str):
            target_keywords = ["cufft", "cublas", "cudart", "nvrtc", "curand", "cudnn", "vcruntime140", "msvcp140"]
            return any(keyword in path for keyword in target_keywords)

        import psutil

        p = psutil.Process(os.getpid())
        print("----List of loaded DLLs----")
        for lib in p.memory_maps():
            if is_target_dll(lib.path.lower()):
                print(lib.path)
