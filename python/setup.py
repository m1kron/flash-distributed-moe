from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Dirs
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]  # /home/REPO/flash-moe/flash-moe
repo_root = project_root.parent  # /home/REPO/flash-moe
out_dir = repo_root / "build"  # place .so and find core lib here
flash_moe_lib_name = "flashMoe"
flash_moe_lib_dir = str(out_dir)


class BuildExtensionToBuildDir(BuildExtension):
    def finalize_options(self):
        super().finalize_options()
        self.build_lib = str(out_dir)
        self.build_temp = str(out_dir)


# TODO: Use CppExtension, but frist make public flashMoe header hip headers free.
setup(
    name="flashMoeLauncher",
    version="0.0.1",
    ext_modules=[
        CUDAExtension(
            name="flashMoeLauncher",
            sources=[str(project_root / "python" / "pyMoeLauncherTorch.cpp")],
            include_dirs=[str(project_root / "include")],
            libraries=[flash_moe_lib_name],
            library_dirs=[flash_moe_lib_dir],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": [], "hipcc": []},
            define_macros=[("TORCH_EXTENSION_NAME", "flashMoeLauncher")],
            extra_link_args=[
                f"-Wl,-rpath,{out_dir}",  # absolute path fallback
                "-Wl,-rpath,$ORIGIN",  # prefer co-located libs
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtensionToBuildDir},
)

# CUDAExtension automatically hipifies files...
os.remove(str(project_root / "python" / "pyMoeLauncherTorch_hip.cpp"))
