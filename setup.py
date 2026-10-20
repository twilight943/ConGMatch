import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


compile_args = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]


setup(
    name="congmatch_fused_cpp",
    ext_modules=[
        CppExtension(
            name="_association_fused_cpp",
            sources=[
                "association_fused_candidates_cpp.cpp",
                "association_fused_build_cpp.cpp",
            ],
            extra_compile_args={"cxx": compile_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
