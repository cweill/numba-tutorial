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
print("🚀 ULTIMATE SUDOKU SOLVER BENCHMARK 🚀")
print("=" * 60)

print(f"Testing with board:")
for row in board:
    print(row)

print(f"\nRunning 30 benchmarks per solver...")
print("-" * 60)

solvers = [
    (solveSudoku, "Original"),
    (solveSudokuOptimized, "Optimized"), 
    (solveSudokuUltraOptimized, "Ultra-Optimized"),
    (solveSudokuHyperOptimized, "🔥 HYPER-Optimized"),
    (solveSudokuBitwise, "Bitwise")
]

all_times = {}
n_runs = 30

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

# Test with multiple different boards
test_boards = [
    # Easy board
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    # Medium board
    [
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0]
    ]
]

print(f"\n📊 MULTI-BOARD STRESS TEST (5 runs each):")
print("-" * 60)

board_names = ["Easy Board", "Medium Board"]
for board_idx, test_board in enumerate(test_boards):
    print(f"\n{board_names[board_idx]}:")
    board_results = {}
    
    for solver_func, name in solvers:
        times = []
        for i in range(5):
            elapsed = benchmark_solver(solver_func, test_board, name)
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        board_results[name] = avg_time
        print(f"  {name:20}: {avg_time:6.2f} ms")
    
    # Find fastest for this board
    fastest_board = min(board_results.items(), key=lambda x: x[1])
    print(f"  🏆 Fastest: {fastest_board[0]} ({fastest_board[1]:.2f} ms)")

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

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Implement naked pairs/triples elimination", "status": "completed", "priority": "high"}, {"id": "2", "content": "Add pointing pairs/box-line reduction", "status": "completed", "priority": "high"}, {"id": "3", "content": "Optimize with precomputed lookup tables", "status": "completed", "priority": "high"}, {"id": "4", "content": "Add constraint ordering and arc consistency", "status": "completed", "priority": "medium"}, {"id": "5", "content": "Implement guessing with constraint learning", "status": "pending", "priority": "medium"}, {"id": "6", "content": "Test the hyper-optimized version", "status": "in_progress", "priority": "high"}]