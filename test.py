#!/usr/bin/env python3
"""
Comprehensive test suite for dumpy library with NumPy comparison
"""

import numpy as np
import time
import sys
try:
    import ENGINE  # Your compiled dumpy module
except ImportError:
    print("Error: ENGINE module not found. Make sure to compile the C++ extension first.")
    sys.exit(1)

def compare_arrays(dumpy_result, numpy_result, tolerance=1e-10):
    """Compare dumpy array with numpy array"""
    if hasattr(dumpy_result, '__iter__'):
        # Convert dumpy array to list for comparison
        dumpy_list = list(dumpy_result) if not isinstance(dumpy_result, list) else dumpy_result
    else:
        dumpy_list = [dumpy_result]
    
    if hasattr(numpy_result, '__iter__'):
        numpy_list = numpy_result.tolist() if hasattr(numpy_result, 'tolist') else list(numpy_result)
    else:
        numpy_list = [numpy_result]
    
    if len(dumpy_list) != len(numpy_list):
        return False, f"Length mismatch: dumpy={len(dumpy_list)}, numpy={len(numpy_list)}"
    
    for i, (d, n) in enumerate(zip(dumpy_list, numpy_list)):
        if abs(d - n) > tolerance:
            return False, f"Value mismatch at index {i}: dumpy={d}, numpy={n}, diff={abs(d-n)}"
    
    return True, "Match"

def test_basic_operations():
    """Test basic array operations"""
    print("=" * 60)
    print("TESTING BASIC OPERATIONS")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    # Test data
    data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    data2 = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    # Create arrays
    print("\n1. Array Creation:")
    dumpy_arr1 = dp.array(data1)
    dumpy_arr2 = dp.array(data2)
    numpy_arr1 = np.array(data1)
    numpy_arr2 = np.array(data2)
    
    print("   Dumpy array 1:", end=" ")
    dp.print(dumpy_arr1)
    print("   NumPy array 1:", numpy_arr1)
    
    # Test zeros
    print("\n2. Zeros array (size 5):")
    dumpy_zeros = dp.zeros(5)
    numpy_zeros = np.zeros(5)
    print("   Dumpy zeros:", end=" ")
    dp.print(dumpy_zeros)
    print("   NumPy zeros:", numpy_zeros)
    match, msg = compare_arrays(dumpy_zeros, numpy_zeros)
    print(f"   Match: {match} - {msg}")
    
    # Test ones
    print("\n3. Ones array (size 4):")
    dumpy_ones = dp.ones(4)
    numpy_ones = np.ones(4)
    print("   Dumpy ones:", end=" ")
    dp.print(dumpy_ones)
    print("   NumPy ones:", numpy_ones)
    match, msg = compare_arrays(dumpy_ones, numpy_ones)
    print(f"   Match: {match} - {msg}")
    
    # Test arange
    print("\n4. Arange (0, 10, 2):")
    dumpy_arange = dp.arange(0, 10, 2)
    numpy_arange = np.arange(0, 10, 2)
    print("   Dumpy arange:", end=" ")
    dp.print(dumpy_arange)
    print("   NumPy arange:", numpy_arange)
    match, msg = compare_arrays(dumpy_arange, numpy_arange)
    print(f"   Match: {match} - {msg}")
    
    # Test linspace
    print("\n5. Linspace (0, 1, 6):")
    dumpy_linspace = dp.linspace(0, 1, 6)
    numpy_linspace = np.linspace(0, 1, 6)
    print("   Dumpy linspace:", end=" ")
    dp.print(dumpy_linspace)
    print("   NumPy linspace:", numpy_linspace)
    match, msg = compare_arrays(dumpy_linspace, numpy_linspace)
    print(f"   Match: {match} - {msg}")

def test_arithmetic_operations():
    """Test arithmetic operations"""
    print("\n" + "=" * 60)
    print("TESTING ARITHMETIC OPERATIONS")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    data2 = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    dumpy_arr1 = dp.array(data1)
    dumpy_arr2 = dp.array(data2)
    numpy_arr1 = np.array(data1)
    numpy_arr2 = np.array(data2)
    
    # Addition
    print("\n1. Addition:")
    dumpy_add = dp.add(dumpy_arr1, dumpy_arr2)
    numpy_add = numpy_arr1 + numpy_arr2
    print("   Dumpy add:", end=" ")
    dp.print(dumpy_add)
    print("   NumPy add:", numpy_add)
    match, msg = compare_arrays(dumpy_add, numpy_add)
    print(f"   Match: {match} - {msg}")
    
    # Subtraction
    print("\n2. Subtraction:")
    dumpy_sub = dp.sub(dumpy_arr1, dumpy_arr2)
    numpy_sub = numpy_arr1 - numpy_arr2
    print("   Dumpy sub:", end=" ")
    dp.print(dumpy_sub)
    print("   NumPy sub:", numpy_sub)
    match, msg = compare_arrays(dumpy_sub, numpy_sub)
    print(f"   Match: {match} - {msg}")
    
    # Multiplication
    print("\n3. Multiplication:")
    dumpy_mul = dp.mul(dumpy_arr1, dumpy_arr2)
    numpy_mul = numpy_arr1 * numpy_arr2
    print("   Dumpy mul:", end=" ")
    dp.print(dumpy_mul)
    print("   NumPy mul:", numpy_mul)
    match, msg = compare_arrays(dumpy_mul, numpy_mul)
    print(f"   Match: {match} - {msg}")
    
    # Division
    print("\n4. Division:")
    dumpy_div = dp.div(dumpy_arr1, dumpy_arr2)
    numpy_div = numpy_arr1 / numpy_arr2
    print("   Dumpy div:", end=" ")
    dp.print(dumpy_div)
    print("   NumPy div:", numpy_div)
    match, msg = compare_arrays(dumpy_div, numpy_div)
    print(f"   Match: {match} - {msg}")
    
    # Dot product
    print("\n5. Dot product:")
    dumpy_dot = dp.dot(dumpy_arr1, dumpy_arr2)
    numpy_dot = np.dot(numpy_arr1, numpy_arr2)
    print(f"   Dumpy dot: {dumpy_dot}")
    print(f"   NumPy dot: {numpy_dot}")
    match, msg = compare_arrays(dumpy_dot, numpy_dot)
    print(f"   Match: {match} - {msg}")

def test_statistical_operations():
    """Test statistical operations"""
    print("\n" + "=" * 60)
    print("TESTING STATISTICAL OPERATIONS")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    dumpy_arr = dp.array(data)
    numpy_arr = np.array(data)
    
    print("Test data:", data)
    
    # Mean
    print("\n1. Mean:")
    dumpy_mean = dp.mean(dumpy_arr)
    numpy_mean = np.mean(numpy_arr)
    print(f"   Dumpy mean: {dumpy_mean}")
    print(f"   NumPy mean: {numpy_mean}")
    match, msg = compare_arrays(dumpy_mean, numpy_mean)
    print(f"   Match: {match} - {msg}")
    
    # Median
    print("\n2. Median:")
    dumpy_median = dp.median(dumpy_arr)
    numpy_median = np.median(numpy_arr)
    print(f"   Dumpy median: {dumpy_median}")
    print(f"   NumPy median: {numpy_median}")
    match, msg = compare_arrays(dumpy_median, numpy_median)
    print(f"   Match: {match} - {msg}")
    
    # Standard deviation
    print("\n3. Standard deviation:")
    dumpy_std = dp.std(dumpy_arr)
    numpy_std = np.std(numpy_arr)
    print(f"   Dumpy std: {dumpy_std}")
    print(f"   NumPy std: {numpy_std}")
    match, msg = compare_arrays(dumpy_std, numpy_std)
    print(f"   Match: {match} - {msg}")
    
    # Variance
    print("\n4. Variance:")
    dumpy_var = dp.var(dumpy_arr)
    numpy_var = np.var(numpy_arr)
    print(f"   Dumpy var: {dumpy_var}")
    print(f"   NumPy var: {numpy_var}")
    match, msg = compare_arrays(dumpy_var, numpy_var)
    print(f"   Match: {match} - {msg}")
    
    # Min/Max
    print("\n5. Min/Max:")
    dumpy_min = dp.min(dumpy_arr)
    dumpy_max = dp.max(dumpy_arr)
    numpy_min = np.min(numpy_arr)
    numpy_max = np.max(numpy_arr)
    print(f"   Dumpy min: {dumpy_min}, max: {dumpy_max}")
    print(f"   NumPy min: {numpy_min}, max: {numpy_max}")
    match_min, _ = compare_arrays(dumpy_min, numpy_min)
    match_max, _ = compare_arrays(dumpy_max, numpy_max)
    print(f"   Min match: {match_min}, Max match: {match_max}")
    
    # Sum
    print("\n6. Sum:")
    dumpy_sum = dp.sum(dumpy_arr)
    numpy_sum = np.sum(numpy_arr)
    print(f"   Dumpy sum: {dumpy_sum}")
    print(f"   NumPy sum: {numpy_sum}")
    match, msg = compare_arrays(dumpy_sum, numpy_sum)
    print(f"   Match: {match} - {msg}")

def test_mathematical_functions():
    """Test mathematical functions"""
    print("\n" + "=" * 60)
    print("TESTING MATHEMATICAL FUNCTIONS")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    data = [1.0, 4.0, 9.0, 16.0, 25.0]
    dumpy_arr = dp.array(data)
    numpy_arr = np.array(data)
    
    print("Test data:", data)
    
    # Square root
    print("\n1. Square root:")
    dumpy_sqrt = dp.sqrt(dumpy_arr)
    numpy_sqrt = np.sqrt(numpy_arr)
    print("   Dumpy sqrt:", end=" ")
    dp.print(dumpy_sqrt)
    print("   NumPy sqrt:", numpy_sqrt)
    match, msg = compare_arrays(dumpy_sqrt, numpy_sqrt)
    print(f"   Match: {match} - {msg}")
    
    # Power
    print("\n2. Power (^2):")
    dumpy_pow = dp.pow(dumpy_arr, 2.0)
    numpy_pow = np.power(numpy_arr, 2.0)
    print("   Dumpy pow:", end=" ")
    dp.print(dumpy_pow)
    print("   NumPy pow:", numpy_pow)
    match, msg = compare_arrays(dumpy_pow, numpy_pow)
    print(f"   Match: {match} - {msg}")
    
    # Absolute value
    data_neg = [-2.5, -1.0, 0.0, 1.0, 2.5]
    dumpy_arr_neg = dp.array(data_neg)
    numpy_arr_neg = np.array(data_neg)
    
    print("\n3. Absolute value:")
    print("   Test data:", data_neg)
    dumpy_abs = dp.abs(dumpy_arr_neg)
    numpy_abs = np.abs(numpy_arr_neg)
    print("   Dumpy abs:", end=" ")
    dp.print(dumpy_abs)
    print("   NumPy abs:", numpy_abs)
    match, msg = compare_arrays(dumpy_abs, numpy_abs)
    print(f"   Match: {match} - {msg}")
    
    # Exponential
    data_small = [0.0, 1.0, 2.0]
    dumpy_arr_small = dp.array(data_small)
    numpy_arr_small = np.array(data_small)
    
    print("\n4. Exponential:")
    print("   Test data:", data_small)
    dumpy_exp = dp.exp(dumpy_arr_small)
    numpy_exp = np.exp(numpy_arr_small)
    print("   Dumpy exp:", end=" ")
    dp.print(dumpy_exp)
    print("   NumPy exp:", numpy_exp)
    match, msg = compare_arrays(dumpy_exp, numpy_exp)
    print(f"   Match: {match} - {msg}")
    
    # Logarithm
    data_pos = [1.0, 2.0, 3.0, 4.0, 5.0]
    dumpy_arr_pos = dp.array(data_pos)
    numpy_arr_pos = np.array(data_pos)
    
    print("\n5. Natural logarithm:")
    print("   Test data:", data_pos)
    dumpy_log = dp.log(dumpy_arr_pos)
    numpy_log = np.log(numpy_arr_pos)
    print("   Dumpy log:", end=" ")
    dp.print(dumpy_log)
    print("   NumPy log:", numpy_log)
    match, msg = compare_arrays(dumpy_log, numpy_log)
    print(f"   Match: {match} - {msg}")

def test_logical_operations():
    """Test logical operations"""
    print("\n" + "=" * 60)
    print("TESTING LOGICAL OPERATIONS")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    data1 = [0.0, 1.0, 0.0, 1.0, 1.0]
    data2 = [1.0, 1.0, 0.0, 0.0, 1.0]
    
    dumpy_arr1 = dp.array(data1)
    dumpy_arr2 = dp.array(data2)
    numpy_arr1 = np.array(data1, dtype=bool)
    numpy_arr2 = np.array(data2, dtype=bool)
    
    print("Test data 1:", data1)
    print("Test data 2:", data2)
    
    # Logical AND
    print("\n1. Logical AND:")
    dumpy_and = dp.logical_and(dumpy_arr1, dumpy_arr2)
    numpy_and = np.logical_and(numpy_arr1, numpy_arr2).astype(float)
    print("   Dumpy AND:", end=" ")
    dp.print(dumpy_and)
    print("   NumPy AND:", numpy_and)
    match, msg = compare_arrays(dumpy_and, numpy_and)
    print(f"   Match: {match} - {msg}")
    
    # Logical OR
    print("\n2. Logical OR:")
    dumpy_or = dp.logical_or(dumpy_arr1, dumpy_arr2)
    numpy_or = np.logical_or(numpy_arr1, numpy_arr2).astype(float)
    print("   Dumpy OR:", end=" ")
    dp.print(dumpy_or)
    print("   NumPy OR:", numpy_or)
    match, msg = compare_arrays(dumpy_or, numpy_or)
    print(f"   Match: {match} - {msg}")
    
    # Logical NOT
    print("\n3. Logical NOT:")
    dumpy_not = dp.logical_not(dumpy_arr1)
    numpy_not = np.logical_not(numpy_arr1).astype(float)
    print("   Dumpy NOT:", end=" ")
    dp.print(dumpy_not)
    print("   NumPy NOT:", numpy_not)
    match, msg = compare_arrays(dumpy_not, numpy_not)
    print(f"   Match: {match} - {msg}")

def test_utility_functions():
    """Test utility functions"""
    print("\n" + "=" * 60)
    print("TESTING UTILITY FUNCTIONS")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    data = [3.7, 1.2, 4.8, 2.1, 3.7, 1.2, 5.9]
    dumpy_arr = dp.array(data)
    numpy_arr = np.array(data)
    
    print("Test data:", data)
    
    # Sort
    print("\n1. Sort:")
    dumpy_sort = dp.sort(dumpy_arr)
    numpy_sort = np.sort(numpy_arr)
    print("   Dumpy sort:", end=" ")
    dp.print(dumpy_sort)
    print("   NumPy sort:", numpy_sort)
    match, msg = compare_arrays(dumpy_sort, numpy_sort)
    print(f"   Match: {match} - {msg}")
    
    # Unique
    print("\n2. Unique:")
    dumpy_unique = dp.unique(dumpy_arr)
    numpy_unique = np.unique(numpy_arr)
    print("   Dumpy unique:", end=" ")
    dp.print(dumpy_unique)
    print("   NumPy unique:", numpy_unique)
    match, msg = compare_arrays(dumpy_unique, numpy_unique)
    print(f"   Match: {match} - {msg}")
    
    # Floor, Ceil, Round
    data_decimal = [1.2, 2.7, -1.3, -2.8, 0.5]
    dumpy_decimal = dp.array(data_decimal)
    numpy_decimal = np.array(data_decimal)
    
    print("\n3. Floor/Ceil/Round:")
    print("   Test data:", data_decimal)
    
    dumpy_floor = dp.floor(dumpy_decimal)
    numpy_floor = np.floor(numpy_decimal)
    print("   Dumpy floor:", end=" ")
    dp.print(dumpy_floor)
    print("   NumPy floor:", numpy_floor)
    match, msg = compare_arrays(dumpy_floor, numpy_floor)
    print(f"   Floor match: {match} - {msg}")
    
    dumpy_ceil = dp.ceil(dumpy_decimal)
    numpy_ceil = np.ceil(numpy_decimal)
    print("   Dumpy ceil:", end=" ")
    dp.print(dumpy_ceil)
    print("   NumPy ceil:", numpy_ceil)
    match, msg = compare_arrays(dumpy_ceil, numpy_ceil)
    print(f"   Ceil match: {match} - {msg}")
    
    dumpy_round = dp.round(dumpy_decimal)
    numpy_round = np.round(numpy_decimal)
    print("   Dumpy round:", end=" ")
    dp.print(dumpy_round)
    print("   NumPy round:", numpy_round)
    match, msg = compare_arrays(dumpy_round, numpy_round)
    print(f"   Round match: {match} - {msg}")
    
    # Clip
    print("\n4. Clip (range 0-5):")
    data_clip = [-2.0, 0.5, 3.0, 7.0, 10.0]
    dumpy_clip_arr = dp.array(data_clip)
    numpy_clip_arr = np.array(data_clip)
    
    dumpy_clip = dp.clip(dumpy_clip_arr, 0.0, 5.0)
    numpy_clip = np.clip(numpy_clip_arr, 0.0, 5.0)
    print("   Test data:", data_clip)
    print("   Dumpy clip:", end=" ")
    dp.print(dumpy_clip)
    print("   NumPy clip:", numpy_clip)
    match, msg = compare_arrays(dumpy_clip, numpy_clip)
    print(f"   Match: {match} - {msg}")

def performance_test():
    """Basic performance comparison"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    dp = ENGINE.dumpy()
    
    # Large arrays for performance testing
    size = 100000
    data1 = [float(i) for i in range(size)]
    data2 = [float(i + 1) for i in range(size)]
    
    print(f"Testing with arrays of size: {size}")
    
    # Dumpy performance
    start_time = time.time()
    dumpy_arr1 = dp.array(data1)
    dumpy_arr2 = dp.array(data2)
    dumpy_result = dp.add(dumpy_arr1, dumpy_arr2)
    dumpy_sum = dp.sum(dumpy_result)
    dumpy_time = time.time() - start_time
    
    # NumPy performance
    start_time = time.time()
    numpy_arr1 = np.array(data1)
    numpy_arr2 = np.array(data2)
    numpy_result = numpy_arr1 + numpy_arr2
    numpy_sum = np.sum(numpy_result)
    numpy_time = time.time() - start_time
    
    print(f"\nDumpy time: {dumpy_time:.6f} seconds")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"Ratio (Dumpy/NumPy): {dumpy_time/numpy_time:.2f}x")
    print(f"Dumpy result sum: {dumpy_sum}")
    print(f"NumPy result sum: {numpy_sum}")
    
    match, msg = compare_arrays(dumpy_sum, numpy_sum)
    print(f"Results match: {match} - {msg}")

def main():
    """Main test function"""
    print("DUMPY vs NUMPY COMPARISON TEST SUITE")
    print("=" * 60)
    print("Testing custom dumpy library against NumPy...")
    
    try:
        test_basic_operations()
        test_arithmetic_operations()
        test_statistical_operations()
        test_mathematical_functions()
        test_logical_operations()
        test_utility_functions()
        performance_test()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        print("If you see 'Match: True' for most operations, your dumpy")
        print("library is working correctly and produces results")
        print("consistent with NumPy!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
