from setuptools import setup, Extension
import setuptools
import numpy as np
import sys
import os

extra_compile_args = []
extra_link_args = []
include_dirs_omp = []

if sys.platform == 'darwin':
    homebrew_prefix = "/opt/homebrew" if os.path.exists("/opt/homebrew") else "/usr/local"
    omp_include_path = os.path.join(homebrew_prefix, 'opt', 'libomp', 'include')
    omp_lib_path = os.path.join(homebrew_prefix, 'opt', 'libomp', 'lib')

    if os.path.exists(omp_include_path):
        extra_compile_args = ['-Xpreprocessor', '-fopenmp']
        extra_link_args = [f'-L{omp_lib_path}', '-lomp']
        include_dirs_omp = [omp_include_path]
    else:
        print("Warning: libomp not found. Please install with 'brew install libomp'")
elif sys.platform.startswith('linux'):
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-lgomp']

def create_extension(name, c_file):
    return Extension(
        name,
        [c_file],
        include_dirs=[np.get_include()] + include_dirs_omp,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

def main():
    setup(
        name="statisco",
        version="0.1.1",
        description="Processing functions module",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        package_dir={'statisco': 'statisco'},
        author="Hector Miranda",
        author_email="hectorsucre13@gmail.com",
        install_requires=[
            "numpy>=1.26.2",
        ],
        ext_modules=[
            create_extension("statisco.statistics", "statisco/c_src/statistics.c"),
            create_extension("statisco.finance", "statisco/c_src/finance.c"),
            create_extension("statisco.indicators.MAs", "statisco/c_src/indicators/MAs.c"),
            create_extension("statisco.indicators.ATRs", "statisco/c_src/indicators/ATRs.c"),
            create_extension("statisco.preprocessing.normalization", "statisco/c_src/preprocessing/normalization.c"),
            create_extension("statisco.utils", "statisco/c_src/utils.c"),
        ],
        zip_safe=False,
    )

if __name__ == "__main__":
    main()

