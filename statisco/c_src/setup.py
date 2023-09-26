from distutils.core import setup, Extension
import numpy as np
import shutil
import os

def main():
    setup(name="stocksOps",
            version="1.0.0",
            description="processingFunctions module",
            author="Hector Miranda",
            author_email="hectorsucre13@gmail.com",
            ext_modules=[
                Extension("processingFunctions",
                          ["processingFunctions.c"],
                          include_dirs=[np.get_include(), "/opt/homebrew/opt/libomp/include"],
                )
            ]
    )
    shutil.move('processingFunctions.cpython-39-darwin.so', os.path.join('..', 'processingFunctions.cpython-39-darwin.so'))

if __name__ == "__main__":
    main()