from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='mixed_cache_ops',
    ext_modules=[
        CUDAExtension(
            name='mixed_cache_ops',
            sources=['mixed_cache.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

#python mixed_cache_setup.py build_ext --inplace
