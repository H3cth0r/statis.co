from distutils.core import setup, Extension
import numpy as np
import shutil
import os

def main():
    omp_include =  "/opt/homebrew/opt/libomp/include" if os.uname().sysname == 'Darwin' else "usr/lib/gcc/x86_64-pc-linux-gnu/13.2.1/include"
    setup(name="stocksOps",
            version="1.0.0",
            description="processingFunctions module",
            author="Hector Miranda",
            author_email="hectorsucre13@gmail.com",
            ext_modules=[
                Extension("processingFunctions",
                          ["processingFunctions.c"],
                          include_dirs=[np.get_include(), omp_include],
                ),
                Extension("statistics",
                          ["statistics.c"],
                          include_dirs=[np.get_include(), omp_include],
                ),
                Extension("finance",
                          ["finance.c"],
                          include_dirs=[np.get_include(), omp_include],
                ),
                Extension("MAs",
                          ["indicators/MAs.c"],
                          include_dirs=[np.get_include(), omp_include],
                ),
            ],
    )
    platform = 'cpython-39-darwin' if os.uname().sysname == 'Darwin' else 'cpython-311-x86_64-linux-gnu' 
    shutil.move(f'processingFunctions.{platform}.so', os.path.join('..', f'processingFunctions.{platform}.so'))
    shutil.move(f'statistics.{platform}.so', os.path.join('..', f'statistics.{platform}.so'))
    shutil.move(f'finance.{platform}.so', os.path.join('..', f'finance.{platform}.so'))
    shutil.move(f'MAs.{platform}.so', os.path.join('../indicators', f'MAs.{platform}.so'))

if __name__ == "__main__":
    main()
