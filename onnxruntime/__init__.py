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


def _get_package_root(package_name, directory_name=None):
    root_directory_name = directory_name or package_name
    import importlib.metadata
    try:
        dist = importlib.metadata.distribution(package_name)
        files = dist.files or []

        # Find the first file that matches the package name and ends with '__init__.py'
        for file in files:
            if file.name.endswith('__init__.py') and root_directory_name in file.parts:
                return file.locate().parent
        else:
            print(f"Could not determine the installation path for package '{package_name}'.")
    except importlib.metadata.PackageNotFoundError:
        print(f"Package '{package_name}' not found.")

    return None

def _get_nvidia_dll_paths(is_windows:bool, cuda: bool = True, cudnn: bool = True):
    if is_windows:
        # Path is relative to site-packages directory.
        cuda_dll_paths = [
            ("nvidia", "cublas", "bin", "cublasLt64_12.dll"),
            ("nvidia", "cublas", "bin", "cublas64_12.dll"),
            ("nvidia", "cufft", "bin", "cufft64_11.dll"),
            ("nvidia", "cuda_runtime", "bin", "cudart64_12.dll"),
        ]
        cudnn_dll_paths = [
            ("nvidia", "cudnn", "bin", "cudnn_engines_runtime_compiled64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_engines_precompiled64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_heuristic64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_ops64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_adv64_9.dll"),
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
            ("nvidia", "cudnn", "lib", "libcudnn_engines_runtime_compiled.so.9"),
            ("nvidia", "cudnn", "lib", "libcudnn_engines_precompiled.so.9"),
            ("nvidia", "cudnn", "lib", "libcudnn_heuristic.so.9"),
            ("nvidia", "cudnn", "lib", "libcudnn_ops.so.9"),
            ("nvidia", "cudnn", "bin", "libcudnn_adv.so.9"),
            ("nvidia", "cudnn", "lib", "libcudnn_graph.so.9"),
            ("nvidia", "cudnn", "lib", "libcudnn.so.9"),
        ]

    # Try load DLLs from site packages.
    return (cuda_dll_paths if cuda else []) + (cudnn_dll_paths if cudnn else [])

def preload_dlls(cuda: bool = True, cudnn: bool = True, msvc: bool = True, directory=None, verbose: bool = False):
    """Preload CUDA 12.x and cuDNN 9.x DLLs in Windows or Linux, and MSVC runtime DLLs in Windows.

       When you installed PyTorch that is compatible with CUDA 12.x, set `torch_safe` to True is recommended.
       Note that there is no need to call this function if `import torch` is done before `import onnxruntime`.

    Args:
        cuda (bool, optional): enable loading CUDA DLLs. Defaults to True.
        cudnn (bool, optional): enable loading cuDNN DLLs. Defaults to True.
        msvc (bool, optional): enable loading MSVC DLLs in Windows. Defaults to True.
        directory(str, optional): a directory contains CUDA or cuDNN DLLs. It can be an absolute path,
           or a path relative to the directory of this file. If its value is None, we will try load from lib directory
           of PyTorch in Windows or nvidia site packages.
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
        is_torch_for_cuda_12 = torch_version and "+cu12" in torch_version
        is_cuda_cudnn_imported_by_torch = False
        if 'torch' in sys.modules:
            is_cuda_cudnn_imported_by_torch = is_torch_for_cuda_12
            if not is_torch_for_cuda_12:
                print(f"\033[33mWARNING: the installed PyTorch {torch_version} does not support CUDA 12.x. "
                    f"Please install PyTorch for CUDA 12.x to be compatible with {package_name}.\033[0m")

        if is_windows and is_torch_for_cuda_12 and not directory:
            torch_root = _get_package_root("torch")
            if torch_root:
                directory = os.path.join(torch_root, "lib")

        base_directory = directory or ".."
        if not os.path.isabs(base_directory):
            base_directory = os.path.join(os.path.dirname(__file__), base_directory)
        base_directory = os.path.normpath(base_directory)
        if not os.path.isdir(base_directory):
            raise RuntimeError(f"Invalid paramter of directory={directory}. The directory does not exist!")

        if is_cuda_cudnn_imported_by_torch:
            if verbose:
                print("Skip loading CUDA and cuDNN DLLs since torch is imported.")
        else:
            dll_paths = _get_nvidia_dll_paths(is_windows, cuda, cudnn)
            loaded_dlls = []
            # if is_windows:
            #     kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
            #     with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
            #     prev_error_mode = kernel32.SetErrorMode(0x0001)

            #     kernel32.LoadLibraryW.restype = ctypes.c_void_p
            #     if with_load_library_flags:
            #         kernel32.LoadLibraryExW.restype = ctypes.c_void_p

            #     path_patched = False
            #     for relative_path in dll_paths:
            #         dll_path = os.path.join(base_directory, relative_path[-1]) if directory else os.path.join(base_directory, *relative_path)
            #         is_loaded = False
            #         if with_load_library_flags:
            #             res = kernel32.LoadLibraryExW(dll_path, None, 0x00001100)
            #             last_error = ctypes.get_last_error()
            #             if res is None and last_error != 126:
            #                 err = ctypes.WinError(last_error)
            #                 err.strerror += (
            #                     f' Error loading "{dll_path}" or one of its dependencies.'
            #                 )
            #                 raise err
            #             elif res is not None:
            #                 is_loaded = True

            #         if not is_loaded:
            #             if not path_patched:
            #                 os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
            #                 path_patched = True
            #             res = kernel32.LoadLibraryW(dll_path)
            #             if res is None:
            #                 err = ctypes.WinError(ctypes.get_last_error())
            #                 err.strerror += (
            #                     f' Error loading "{dll_path}" or one of its dependencies.'
            #                 )
            #                 raise err

            #     kernel32.SetErrorMode(prev_error_mode)
            for relative_path in dll_paths:
                dll_path = os.path.join(base_directory, relative_path[-1]) if directory else os.path.join(base_directory, *relative_path)
                if os.path.isfile(dll_path):
                    try:
                        _ = ctypes.CDLL(dll_path)
                        loaded_dlls.append(relative_path[-1])
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
