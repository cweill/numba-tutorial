import time
from sudoku import (
    solveSudoku, 
    solveSudokuOptimized, 
    solveSudokuUltraOptimized,
    solveSudokuBitwise, 
    generate_board
)

def benchmark_solver(solver_func, board, name):
    """Benchmark a sudoku solver"""
    board_copy = [row[:] for row in board]
    start = time.time()
    result = solver_func(board_copy)
    end = time.time()
    elapsed = (end - start) * 1000
    return elapsed

# Test with the standard board
board = generate_board()
print("Comprehensive Sudoku Solver Benchmark")
print("=" * 50)

print(f"Testing with board:")
for row in board:
    print(row)

print(f"\nRunning 20 benchmarks per solver:")
print("-" * 30)

solvers = [
    (solveSudoku, "Original"),
    (solveSudokuOptimized, "Optimized"), 
    (solveSudokuUltraOptimized, "Ultra-Optimized"),
    (solveSudokuBitwise, "Bitwise")
]

all_times = {}
n_runs = 20

for solver_func, name in solvers:
    times = []
    for i in range(n_runs):
        elapsed = benchmark_solver(solver_func, board, name)
        times.append(elapsed)
    
    all_times[name] = times
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"{name:15}: {avg_time:6.2f} ms avg (min: {min_time:5.2f}, max: {max_time:5.2f})")

print("\n" + "=" * 50)
print("Performance Comparison:")
baseline = sum(all_times["Original"]) / len(all_times["Original"])

for name in ["Optimized", "Ultra-Optimized", "Bitwise"]:
    avg_time = sum(all_times[name]) / len(all_times[name])
    speedup = baseline / avg_time
    print(f"{name:15} is {speedup:4.1f}x faster than Original")

# Find the fastest
fastest_name = min(all_times.keys(), key=lambda x: sum(all_times[x])/len(all_times[x]))
fastest_time = sum(all_times[fastest_name]) / len(all_times[fastest_name])

print(f"\n🏆 Fastest solver: {fastest_name} ({fastest_time:.2f} ms average)")

# Test with a different board to ensure consistency
test_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

print(f"\nTesting with different board (5 runs each):")
print("-" * 30)

for solver_func, name in solvers:
    times = []
    for i in range(5):
        elapsed = benchmark_solver(solver_func, test_board, name)
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    print(f"{name:15}: {avg_time:6.2f} ms avg")

print("\n" + "=" * 50)
print("Optimization techniques used in Ultra-Optimized:")
print("• Bitwise operations for faster constraint manipulation")
print("• Forward checking to detect dead ends early")
print("• Enhanced MRV (Minimum Remaining Values) heuristic")
print("• Naked singles detection with bit manipulation")
print("• Hidden singles detection for rows, columns, and boxes")
print("• Efficient state saving and restoration")