from setuptools import setup, Extension
import setuptools
import numpy as np

def main():
    setup(
        name="statisco",
        version="0.1.1",
        description="Processing functions module",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages("./statisco/c_src"),
        package_dir={'statisco': 'statisco'},
        author="Hector Miranda",
        author_email="hectorsucre13@gmail.com",
        install_requires=[
            "numpy>=1.26.2",
        ],
        ext_modules=[
            Extension(
                "statisco.statistics",
                ["statisco/c_src/statistics.c"],
                include_dirs=[np.get_include()],
                # extra_compile_args=['-fopenmp'],
                extra_link_args=['-lgomp'],
            ),
            Extension(
                "statisco.finance",
                ["statisco/c_src/finance.c"],
                include_dirs=[np.get_include()],
                # extra_compile_args=['-fopenmp'],
                extra_link_args=['-lgomp'],
            ),
            Extension(
                "statisco.indicators.MAs",
                ["statisco/c_src/indicators/MAs.c"],
                include_dirs=[np.get_include()],
                # extra_compile_args=['-fopenmp'],
                extra_link_args=['-lgomp'],
                ),
            Extension(
                "statisco.indicators.ATRs",
                ["statisco/c_src/indicators/ATRs.c"],
                include_dirs=[np.get_include()],
                # extra_compile_args=['-fopenmp'],
                extra_link_args=['-lgomp'],
            ),
        ],
        zip_safe=False,
    )

if __name__ == "__main__":
    main()

