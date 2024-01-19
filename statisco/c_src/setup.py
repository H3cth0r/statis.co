from distutils.core import setup, Extension
import numpy as np
import shutil
import os

def main():
    omp_include =  "/opt/homebrew/opt/libomp/include" if os.uname().sysname == 'Darwin' else "usr/lib/gcc/x86_64-pc-linux-gnu/13.2.1/include"
    setup(name="statis.co",
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
                Extension("ATRs",
                          ["indicators/ATRs.c"],
                          include_dirs=[np.get_include(), omp_include],
                ),
                Extension("minmax_scaler",
                          ["preprocessing/normalization/minmax_scaler.c"],
                          include_dirs=[np.get_include(), omp_include],
                ),
            ],
    )
    platform = 'cpython-39-darwin' if os.uname().sysname == 'Darwin' else 'cpython-311-x86_64-linux-gnu' 
    shutil.move(f'processingFunctions.{platform}.so', os.path.join('..', f'processingFunctions.{platform}.so'))
    # Statistical Tools
    shutil.move(f'statistics.{platform}.so', os.path.join('..', f'statistics.{platform}.so'))
    # Financial Tools
    shutil.move(f'finance.{platform}.so', os.path.join('..', f'finance.{platform}.so'))
    # Indicators
    shutil.move(f'MAs.{platform}.so', os.path.join('../indicators', f'MAs.{platform}.so'))
    shutil.move(f'ATRs.{platform}.so', os.path.join('../indicators', f'ATRs.{platform}.so'))
    # Preprocessing Tools
    shutil.move(f'minmax_scaler.{platform}.so', os.path.join('../preprocessing/normalization', f'minmax_scaler.{platform}.so'))

if __name__ == "__main__":
    main()
