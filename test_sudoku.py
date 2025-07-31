def solveSudoku(board):
    solvePartialSudoku(0, 0, board)
    return board


def solvePartialSudoku(row, col, board):
    currentRow = row
    currentCol = col

    if currentCol == len(board[currentRow]):
        currentRow += 1
        currentCol = 0
        if currentRow == len(board):
            return True

    if board[currentRow][currentCol] == 0:
        return tryDigitsAtPosition(currentRow, currentCol, board)

    return solvePartialSudoku(currentRow, currentCol + 1, board)


def tryDigitsAtPosition(row, col, board):
    for digit in range(1, 10):
        if isValidAtPosition(digit, row, col, board):
            board[row][col] = digit
            if solvePartialSudoku(row, col + 1, board):
                return True

    board[row][col] = 0
    return False


def isValidAtPosition(value, row, col, board):
    if value in board[row]:
        return False
    for idx in range(len(board)):
        if board[idx][col] == value:
            return False

    subgridRowStart = (row // 3) * 3
    subgridColStart = (col // 3) * 3
    for rowIdx in range(3):
        for colIdx in range(3):
            rowToCheck = subgridRowStart + rowIdx
            colToCheck = subgridColStart + colIdx
            existingValue = board[rowToCheck][colToCheck]

            if existingValue == value:
                return False

    return True


def solveSudokuOptimized(board):
    # Initialize possible values for each cell
    possible = [[set() for _ in range(9)] for _ in range(9)]

    # Set up initial possibilities
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                possible[i][j] = set(range(1, 10))

    # Remove initial constraints
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                updateConstraints(board[i][j], i, j, possible)

    # Apply initial constraint propagation
    if not propagateConstraints(board, possible):
        return board

    # Solve with backtracking using MRV heuristic
    if solveWithMRV(board, possible):
        return board
    return board


def updateConstraints(value, row, col, possible):
    """Remove value from row, column, and box constraints"""
    # Clear this cell's possibilities
    possible[row][col] = set()

    # Remove from row
    for c in range(9):
        possible[row][c].discard(value)

    # Remove from column
    for r in range(9):
        possible[r][col].discard(value)

    # Remove from 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            possible[r][c].discard(value)


def propagateConstraints(board, possible):
    """Propagate constraints by filling cells with only one possibility"""
    changed = True
    while changed:
        changed = False

        # Fill cells with only one possibility
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0 and len(possible[row][col]) == 1:
                    value = next(iter(possible[row][col]))
                    board[row][col] = value
                    updateConstraints(value, row, col, possible)
                    changed = True
                elif board[row][col] == 0 and len(possible[row][col]) == 0:
                    return False

        # Hidden singles: value that can only go in one place in a unit
        if not changed:
            changed = findHiddenSingles(board, possible)

    return True


def findHiddenSingles(board, possible):
    """Find values that can only go in one place in a row/col/box"""
    changed = False

    # Check rows
    for row in range(9):
        for value in range(1, 10):
            positions = []
            for col in range(9):
                if value in possible[row][col]:
                    positions.append(col)
            if len(positions) == 1:
                col = positions[0]
                if board[row][col] == 0:
                    board[row][col] = value
                    updateConstraints(value, row, col, possible)
                    changed = True

    # Check columns
    for col in range(9):
        for value in range(1, 10):
            positions = []
            for row in range(9):
                if value in possible[row][col]:
                    positions.append(row)
            if len(positions) == 1:
                row = positions[0]
                if board[row][col] == 0:
                    board[row][col] = value
                    updateConstraints(value, row, col, possible)
                    changed = True

    # Check boxes
    for box_r in range(3):
        for box_c in range(3):
            for value in range(1, 10):
                positions = []
                for r in range(3):
                    for c in range(3):
                        row = box_r * 3 + r
                        col = box_c * 3 + c
                        if value in possible[row][col]:
                            positions.append((row, col))
                if len(positions) == 1:
                    row, col = positions[0]
                    if board[row][col] == 0:
                        board[row][col] = value
                        updateConstraints(value, row, col, possible)
                        changed = True

    return changed


def solveWithMRV(board, possible):
    """Solve using MRV heuristic with constraint maintenance"""
    # Apply constraint propagation first
    if not propagateConstraints(board, possible):
        return False

    # Find empty cell with fewest possibilities
    min_poss = 10
    best_cell = None

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                n = len(possible[row][col])
                if n == 0:
                    return False
                if n < min_poss:
                    min_poss = n
                    best_cell = (row, col)

    if best_cell is None:
        return True  # Solved

    row, col = best_cell

    # Try each possible value
    for value in list(possible[row][col]):
        # Save state for backtracking
        old_board = [row[:] for row in board]
        old_possible = [[cell.copy() for cell in row] for row in possible]

        # Make move
        board[row][col] = value
        updateConstraints(value, row, col, possible)

        # Recurse
        if solveWithMRV(board, possible):
            return True

        # Backtrack
        for i in range(9):
            for j in range(9):
                board[i][j] = old_board[i][j]
                possible[i][j] = old_possible[i][j]

    return False


def generate_board():
    return [
        [0, 2, 0, 0, 9, 0, 1, 0, 0],
        [0, 0, 7, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 6, 0],
        [0, 0, 1, 9, 0, 4, 0, 0, 0],
        [0, 0, 0, 6, 0, 5, 0, 0, 7],
        [8, 0, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 8, 5],
        [4, 9, 0, 0, 3, 0, 0, 0, 0],
    ]


def print_board(board):
    for i, row in enumerate(board):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        print(
            " ".join(str(x) if x != 0 else "." for x in row[:3])
            + " | "
            + " ".join(str(x) if x != 0 else "." for x in row[3:6])
            + " | "
            + " ".join(str(x) if x != 0 else "." for x in row[6:])
        )


# Test both solvers
board = generate_board()
print("Original board:")
print_board(board)

print("\n" + "=" * 50)
print("Testing original solver:")
board_original = [row[:] for row in board]
result_original = solveSudoku(board_original)
print_board(result_original)

print("\n" + "=" * 50)
print("Testing optimized solver:")
board_optimized = [row[:] for row in board]
result_optimized = solveSudokuOptimized(board_optimized)
print_board(result_optimized)

# Check if results are the same
print("\n" + "=" * 50)
print("Results are identical:", result_original == result_optimized)
