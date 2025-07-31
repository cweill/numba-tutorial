import time
from sudoku import (
    solveSudoku, 
    solveSudokuOptimized, 
    solveSudokuUltraOptimized,
    solveSudokuHyperOptimized,
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
print("🚀 FINAL SUDOKU SOLVER BENCHMARK 🚀")
print("=" * 60)

print(f"Testing with board:")
for row in board:
    print(row)

print(f"\nRunning 25 benchmarks per solver...")
print("-" * 60)

solvers = [
    (solveSudoku, "Original"),
    (solveSudokuOptimized, "Optimized"), 
    (solveSudokuUltraOptimized, "Ultra-Optimized"),
    (solveSudokuHyperOptimized, "🔥 HYPER-Optimized"),
    (solveSudokuBitwise, "Bitwise")
]

all_times = {}
n_runs = 25

for solver_func, name in solvers:
    print(f"Testing {name}...", end=" ", flush=True)
    times = []
    for i in range(n_runs):
        elapsed = benchmark_solver(solver_func, board, name)
        times.append(elapsed)
    
    all_times[name] = times
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"✓ {avg_time:6.2f} ms avg (min: {min_time:5.2f}, max: {max_time:5.2f})")

print("\n" + "=" * 60)
print("🏆 PERFORMANCE COMPARISON:")
baseline = sum(all_times["Original"]) / len(all_times["Original"])

performance_data = []
for name in ["Optimized", "Ultra-Optimized", "🔥 HYPER-Optimized", "Bitwise"]:
    avg_time = sum(all_times[name]) / len(all_times[name])
    speedup = baseline / avg_time
    performance_data.append((speedup, name, avg_time))
    print(f"{name:20} is {speedup:5.1f}x faster than Original ({avg_time:5.2f} ms)")

# Sort by performance
performance_data.sort(reverse=True)
fastest_speedup, fastest_name, fastest_time = performance_data[0]

print(f"\n🏆 CHAMPION: {fastest_name}")
print(f"⚡ {fastest_speedup:.1f}x faster than Original")
print(f"🕓 {fastest_time:.2f} ms average")

print("\n" + "=" * 60)
print("🧠 ADVANCED TECHNIQUES IN HYPER-OPTIMIZED SOLVER:")
print("✅ Bitwise constraint representation (9-bit integers)")
print("✅ Precomputed lookup tables for bit counting")
print("✅ Naked singles detection with optimized bit manipulation")
print("✅ Hidden singles for rows, columns, and boxes")
print("✅ Naked pairs/triples elimination")
print("✅ Pointing pairs (box-line reduction)")
print("✅ Enhanced MRV heuristic with early termination")
print("✅ Forward checking for immediate conflict detection")
print("✅ Optimized state saving and restoration")
print("✅ Constraint ordering for better pruning")

print(f"\n💡 The HYPER-Optimized solver combines multiple advanced")
print(f"   constraint satisfaction techniques for maximum speed!")