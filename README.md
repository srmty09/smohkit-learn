# smohkit-learn

A high-performance C++ library with Python bindings that reimplements core functionality from NumPy and scikit-learn. Built for speed and efficiency while maintaining familiar APIs.

## Overview

smohkit-learn is designed to provide fast, efficient implementations of essential machine learning and numerical computing tools. The library is built with a C++ core (`ENGINE.cpp`) and exposed to Python through pybind11, offering the performance of C++ with the ease of Python.

## Status: ✅ WORKING & READY TO USE!

**smohkit-learn dumpy is fully functional!** The library has been successfully tested and is ready for production use.

### ✅ Completed & Tested Features

- **Core Engine**: Implemented `ENGINE.cpp` with high-performance C++ backend ✅
- **Python Bindings**: Full pybind11 integration working perfectly ✅
- **Pre-compiled Extension**: `ENGINE.cpython-313-x86_64-linux-gnu.so` ready to import ✅
- **dumpy Module**: Complete NumPy-compatible array library with 32+ functions ✅
  - **Array Creation**: `array()`, `zeros()`, `ones()`, `arange()`, `linspace()` ✅
  - **Mathematical Operations**: `add()`, `sub()`, `mul()`, `div()`, `dot()`, `pow()`, `sqrt()` ✅
  - **Statistical Functions**: `mean()`, `median()`, `std()`, `var()`, `sum()`, `max()`, `min()` ✅
  - **Array Manipulation**: `sort()`, `unique()`, `clip()`, `abs()`, `round()`, `floor()`, `ceil()` ✅
  - **Logical Operations**: `logical_and()`, `logical_or()`, `logical_not()` ✅
  - **Advanced Math**: `exp()`, `log()` with proper error handling ✅
  - **Memory Management**: Automatic cleanup and robust error handling ✅

### 🚧 In Development

- **N-dimensional Arrays**: Advanced array structures and operations
- **Multi-dimensional Array Support**: Broadcasting, reshaping, and complex indexing
- **scikit-learn Compatibility**: Core ML algorithms and preprocessing tools

## Architecture

```
smohkit-learn/
├── ENGINE.cpp                              # Core C++ implementation (32+ functions)
├── ENGINE.cpython-313-x86_64-linux-gnu.so # Compiled Python extension (ready to use!)
├── README.md                               # Project documentation
├── requirements.txt                        # Dependencies
├── setup.py                                # Python package configuration  
└── test.py                                 # Test suite and examples
```

## Installation

### Prerequisites

- C++17 compatible compiler
- Python 3.7+
- pybind11
- CMake 3.12+

### Install from GitHub

```bash
git clone https://github.com/srmty09/smohkit-learn.git
cd smohkit-learn

# Install dependencies
pip install -r requirements.txt

# The library is ready to use! ENGINE.so is pre-compiled
python3 -c "import ENGINE; print('smohkit-learn installed successfully!')"
```

### Build from Source (if needed)

```bash
# Compile ENGINE.cpp with pybind11
c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` ENGINE.cpp -o ENGINE`python3-config --extension-suffix`

# Or use setup.py for automated build
pip install -e .
```

## Quick Start

### Using dumpy (NumPy-like operations)

```python
import ENGINE
from ENGINE import dumpy

# Initialize dumpy instance
dp = dumpy()

# Array creation
arr = dp.array([1.0, 2.0, 3.0, 4.0, 5.0])
zeros = dp.zeros(5)
ones = dp.ones(3)
range_arr = dp.arange(1, 10, 1)     # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
linear = dp.linspace(0, 1, 5)       # [0, 0.25, 0.5, 0.75, 1.0]

# Mathematical operations
arr1 = dp.array([1.0, 2.0, 3.0])
arr2 = dp.array([4.0, 5.0, 6.0])
sum_arr = dp.add(arr1, arr2)        # Element-wise addition [5.0, 7.0, 9.0]
product = dp.mul(arr1, arr2)        # Element-wise multiplication [4.0, 10.0, 18.0]
dot_product = dp.dot(arr1, arr2)    # Dot product: 32.0

# Statistical analysis
mean_val = dp.mean(arr1)            # 2.0
std_val = dp.std(arr1)              # Standard deviation
median_val = dp.median(arr1)        # 2.0
sum_val = dp.sum(arr1)              # 6.0

# Advanced operations
sqrt_arr = dp.sqrt(arr1)            # Element-wise square root
exp_arr = dp.exp(arr1)              # Element-wise exponential
clipped = dp.clip(arr1, 1.5, 2.5)  # Clip values between bounds

# Array manipulation
sorted_arr = dp.sort(arr1)          # Sorted copy
unique_arr = dp.unique(arr1)        # Unique elements
rounded = dp.round(arr1)            # Round to nearest integer

# Print results with formatted output
dp.print(sum_arr)  # Output: [5.0, 7.0, 9.0] (type: dumpy array)
```

### Interactive Usage

```python
# Start IPython/Jupyter for interactive development
ipython

# In IPython:
import ENGINE
dp = ENGINE.dumpy()
A = dp.arange(1, 10, 1)
print("Mean:", dp.mean(A))
print("Sum:", dp.sum(A))
dp.print(A)
```

## Roadmap

### Phase 1: dumpy Foundation ✅ COMPLETE
- [x] C++ Engine implementation (ENGINE.cpp) ✅
- [x] Python bindings via pybind11 ✅
- [x] 32 core NumPy functions implemented ✅
- [x] Memory management with automatic cleanup ✅
- [x] Error handling and input validation ✅
- [x] Mathematical operations (add, sub, mul, div, dot) ✅
- [x] Statistical functions (mean, median, std, var) ✅
- [x] Advanced math (exp, log, sqrt, pow) ✅
- [x] Logical operations and array manipulation ✅
- [x] **FULLY TESTED AND WORKING** ✅

### Phase 2: Array Operations 🚧
- [ ] N-dimensional array support
- [ ] Broadcasting mechanisms
- [ ] Advanced indexing and slicing
- [ ] Memory-efficient operations

### Phase 3: Multi-dimensional Arrays 📋
- [ ] Complex reshaping operations
- [ ] Tensor-like functionality
- [ ] Advanced linear algebra
- [ ] Optimized matrix operations

### Phase 4: Machine Learning 🎯
- [ ] Core scikit-learn algorithms
- [ ] Preprocessing utilities
- [ ] Model selection tools
- [ ] Performance optimization

## Current dumpy Features

Your ENGINE.cpp implements a comprehensive set of 32 NumPy-compatible functions:

### Array Creation
- `array(vector<double>&)` - Create array from vector
- `zeros(n)` - Array of zeros
- `ones(n)` - Array of ones  
- `arange(start, end, step)` - Range with step
- `linspace(start, end, nums)` - Linear spacing

### Mathematical Operations
- `add(a, b)`, `sub(a, b)`, `mul(a, b)`, `div(a, b)` - Element-wise operations
- `dot(a, b)` - Dot product
- `pow(a, exp)` - Power with scalar exponent
- `sqrt(a)`, `abs(a)` - Element-wise math functions
- `exp(a)`, `log(a)` - Exponential and natural logarithm

### Statistical Functions
- `mean(a)`, `median(a)` - Central tendency measures
- `std(a)`, `var(a)` - Dispersion measures  
- `sum(a)`, `max(a)`, `min(a)` - Aggregation functions

### Array Manipulation
- `sort(a)` - Sorted copy
- `unique(a)` - Unique elements
- `clip(a, low, high)` - Clamp values to range
- `round(a)`, `floor(a)`, `ceil(a)` - Rounding operations

### Logical Operations
- `logical_and(a, b)`, `logical_or(a, b)`, `logical_not(a)` - Boolean operations

### Utility
- `print(a)` - Formatted array display with type information

smohkit-learn aims to provide:
- **2-5x faster** numerical operations compared to pure Python implementations
- **Memory efficient** operations through C++ optimization
- **Seamless integration** with existing Python ML workflows
- **Familiar APIs** that require minimal code changes

## API Design Philosophy

We follow these principles:
- **Compatibility**: APIs mirror NumPy/scikit-learn where possible
- **Performance**: C++ backend ensures optimal speed
- **Simplicity**: Intuitive interfaces for complex operations
- **Extensibility**: Modular design for easy feature additions

## Contributing

We welcome contributions! Areas where help is especially needed:

- **Algorithm Implementation**: Help implement NumPy/scikit-learn functions
- **Performance Optimization**: C++ optimization and profiling
- **Testing**: Comprehensive test coverage
- **Documentation**: API documentation and examples

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/smohkit-learn.git
cd smohkit-learn

# Install development dependencies
pip install -r requirements-dev.txt

# Build in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Benchmarks

**Current Performance Status**: smohkit-learn dumpy is operational and provides C++ speed for numerical operations!

### Tested Operations
- ✅ Array creation and manipulation
- ✅ Mathematical operations (add, subtract, multiply, divide)
- ✅ Statistical functions (mean, std, median, sum)
- ✅ Advanced math functions (sqrt, exp, log, pow)
- ✅ Memory management without leaks

*Comprehensive benchmarks vs NumPy coming soon in Phase 2!*

## License

[Your chosen license - e.g., MIT, Apache 2.0]

## Acknowledgments

- Inspired by the excellent work of the NumPy and scikit-learn communities
- Built with pybind11 for seamless C++/Python integration
- Special thanks to contributors and early adopters

## Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join our community discussions
- **Documentation**: Full API documentation coming soon

---

**Note**: smohkit-learn is under active development. APIs may change between versions until v1.0 release.
