import time

from tqdm import tqdm

from sudoku import generate_board, solveSudoku, solveSudokuBitwise, solveSudokuOptimized


def benchmark_solver(solver_func, board, name):
    """Benchmark a sudoku solver"""
    board_copy = [row[:] for row in board]
    start = time.time()
    result = solver_func(board_copy)
    end = time.time()
    elapsed = (end - start) * 1000
    # print(f"{name}: {elapsed:.2f} ms")
    return elapsed


# Test with the standard board multiple times
board = generate_board()
n = 100
print(f"Running {n} benchmarks with standard board:")
print("=" * 40)

original_times = []
optimized_times = []
bitwise_times = []

for i in tqdm(range(n)):
    # print(f"\nRun {i+1}:")
    original_times.append(benchmark_solver(solveSudoku, board, "Original"))
    optimized_times.append(benchmark_solver(solveSudokuOptimized, board, "Optimized"))
    bitwise_times.append(benchmark_solver(solveSudokuBitwise, board, "Bitwise"))

print("\n" + "=" * 40)
print("Average times:")
print(f"Original: {sum(original_times)/len(original_times):.2f} ms")
print(f"Optimized: {sum(optimized_times)/len(optimized_times):.2f} ms")
print(f"Bitwise: {sum(bitwise_times)/len(bitwise_times):.2f} ms")

print("\nSpeedup:")
print(
    f"Optimized is {sum(original_times)/sum(optimized_times):.1f}x faster than Original"
)
print(f"Bitwise is {sum(original_times)/sum(bitwise_times):.1f}x faster than Original")
