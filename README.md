![logo](https://github.com/H3cth0r/statis.co/blob/main/resources/logo.png)

## What is Statis.Co?
[Statis.Co](https://github.com/H3cth0r/statis.co) is a Python module encompassing diverse financial tools and functionalities, 
including indicators, statistical calculations, and connections to data sources. 
The module is crafted in pure C, seamlessly integrated with Numpy's API, and 
employs parallelization tools such as OpenMP. The primary goal is to facilitate 
blazing-fast calculations and empower the module to adeptly handle substantial workloads.
To ensure robust memory management and prevent memory leaks, we utilize tools such as 
Valgrind and Bloomberg's Memray for in-depth analysis of memory usage.


## Why Statis.Co?
In today's finance industry, there's a growing demand for optimized tools to handle 
massive workloads swiftly. As financial data complexity rises, cutting-edge technologies 
play a pivotal role in processing, analyzing, and deriving insights for market analysis, 
risk assessment, and investment strategies. Staying competitive in finance now hinges on 
the ability to make informed decisions at unprecedented speeds. Notably, we've achieved 
over a 70% optimization in execution time compared to pure Python NumPy calculations.


## Installation
From PyPi. Check the [PyPi](https://pypi.org/project/Statis.Co/) repo:
```
pip install statisco
```

From source:
```
python setup.py sdist bdist_wheel
```

## To develop and Test from the source
```
python3 statisco/c_src/setup.py build_ext --inplace
```
If you'd like to contribute, please contact me via GitHub.

## Requirements
To compile the C extension, the primary requirements include the installation of Numpy and GCC.


## Usage
This is a usage example:
```python
from statisco.statistics import closingReturns, mean

msft["MyCloseReturns"] = closingReturns(msft["Adj Close"])
myMean = mean(msft["MyCloseReturns"])

print(f"myMean: {myMean}")
msft.head()
```
For a more in-depth understanding of usage, refer to the detailed examples provided in the 
[test notebook](https://github.com/H3cth0r/statis.co/blob/main/test.ipynb).
We'll be working on developing a documentation site.

## TODO
This is what I’ll be working on:
- Documentation
- More indicators
- Automatization Classes
- Basic ML models integration

## LICENSE
[Statis.Co](https://github.com/H3cth0r/statis.co) by [Hector Miranda](https://github.com/H3cth0r) is licensed under [Attribution-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-sa/4.0/?ref=chooser-v1) ![CC](https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1) ![BY](https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1) ![SA](https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1).

