# O(1) time | O(1) space
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


# Optimized Sudoku Solver with constraint propagation
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


# Ultra-optimized Sudoku Solver with bitwise operations and advanced techniques
def solveSudokuUltraOptimized(board):
    # Use bitwise representation: each cell has 9-bit int where bit i = digit i+1 possible
    possible = [[0 for _ in range(9)] for _ in range(9)]

    # Initialize possibilities (bits 0-8 represent digits 1-9)
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                possible[i][j] = 0b111111111  # All 9 digits possible

    # Remove initial constraints
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                updateConstraintsBitwise(board[i][j], i, j, possible)

    # Apply constraint propagation
    if not propagateConstraintsBitwise(board, possible):
        return board

    # Solve with enhanced MRV + forward checking
    if solveWithMRVBitwise(board, possible):
        return board
    return board


def updateConstraintsBitwise(value, row, col, possible):
    """Remove value from constraints using bitwise operations"""
    bit_mask = ~(1 << (value - 1))  # Mask to clear bit for this value

    # Clear this cell's possibilities
    possible[row][col] = 0

    # Remove from row, column, and box
    for i in range(9):
        possible[row][i] &= bit_mask  # Row
        possible[i][col] &= bit_mask  # Column

    # Box constraints
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            possible[r][c] &= bit_mask


def propagateConstraintsBitwise(board, possible):
    """Propagate constraints using bitwise operations"""
    changed = True
    while changed:
        changed = False

        # Fill cells with only one possibility (naked singles)
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    bits = possible[row][col]
                    if bits == 0:
                        return False  # Dead end

                    # Check if only one bit is set (power of 2)
                    if bits & (bits - 1) == 0 and bits != 0:
                        # Find which digit this is using bit manipulation
                        digit = (bits & -bits).bit_length()

                        board[row][col] = digit
                        updateConstraintsBitwise(digit, row, col, possible)
                        changed = True
                        break
            if changed:
                break

        # Hidden singles: value that can only go in one place
        if not changed:
            changed = findHiddenSinglesBitwise(board, possible)

    return True


def findHiddenSinglesBitwise(board, possible):
    """Find hidden singles using bitwise operations"""
    changed = False

    # Check each digit (1-9)
    for digit in range(1, 10):
        bit = 1 << (digit - 1)

        # Check rows
        for row in range(9):
            positions = []
            for col in range(9):
                if possible[row][col] & bit:
                    positions.append(col)

            if len(positions) == 1:
                col = positions[0]
                if board[row][col] == 0:
                    board[row][col] = digit
                    updateConstraintsBitwise(digit, row, col, possible)
                    changed = True

        # Check columns
        for col in range(9):
            positions = []
            for row in range(9):
                if possible[row][col] & bit:
                    positions.append(row)

            if len(positions) == 1:
                row = positions[0]
                if board[row][col] == 0:
                    board[row][col] = digit
                    updateConstraintsBitwise(digit, row, col, possible)
                    changed = True

        # Check boxes
        for box_r in range(3):
            for box_c in range(3):
                positions = []
                for r in range(3):
                    for c in range(3):
                        row = box_r * 3 + r
                        col = box_c * 3 + c
                        if possible[row][col] & bit:
                            positions.append((row, col))

                if len(positions) == 1:
                    row, col = positions[0]
                    if board[row][col] == 0:
                        board[row][col] = digit
                        updateConstraintsBitwise(digit, row, col, possible)
                        changed = True

    return changed


def solveWithMRVBitwise(board, possible):
    """Solve using MRV with bitwise operations and forward checking"""
    # Apply constraint propagation
    if not propagateConstraintsBitwise(board, possible):
        return False

    # Find empty cell with fewest possibilities
    min_bits = 10
    best_cell = None

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                bits = possible[row][col]
                if bits == 0:
                    return False  # Dead end

                # Count set bits (number of possibilities) - Brian Kernighan's algorithm
                bit_count = 0
                temp = bits
                while temp:
                    bit_count += 1
                    temp &= temp - 1
                if bit_count < min_bits:
                    min_bits = bit_count
                    best_cell = (row, col)

                    # If only one possibility, we'll fill it in propagation
                    if bit_count == 1:
                        break
        if best_cell and min_bits == 1:
            break

    if best_cell is None:
        return True  # Solved

    row, col = best_cell
    bits = possible[row][col]

    # Try each possible digit
    for digit in range(1, 10):
        if bits & (1 << (digit - 1)):
            # Save state efficiently
            old_board = [r[:] for r in board]
            old_possible = [r[:] for r in possible]

            # Make move
            board[row][col] = digit
            updateConstraintsBitwise(digit, row, col, possible)

            # Forward checking: ensure no empty cell has zero possibilities
            valid = True
            for r in range(9):
                for c in range(9):
                    if board[r][c] == 0 and possible[r][c] == 0:
                        valid = False
                        break
                if not valid:
                    break

            if valid and solveWithMRVBitwise(board, possible):
                return True

            # Backtrack
            for i in range(9):
                for j in range(9):
                    board[i][j] = old_board[i][j]
                    possible[i][j] = old_possible[i][j]

    return False


# Hyper-optimized Sudoku Solver with advanced constraint satisfaction techniques
def solveSudokuHyperOptimized(board):
    # Precomputed lookup tables for faster operations
    BIT_COUNT = [bin(i).count("1") for i in range(512)]  # Precompute bit counts
    SINGLE_BIT = [
        (i & -i).bit_length() if i & (i - 1) == 0 and i != 0 else 0 for i in range(512)
    ]

    # Initialize bitwise constraints
    possible = [
        [0b111111111 if board[i][j] == 0 else 0 for j in range(9)] for i in range(9)
    ]

    # Apply initial constraints
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                eliminate_value_hyper(board[i][j], i, j, possible)

    # Advanced constraint propagation
    if not hyper_propagate(board, possible, BIT_COUNT, SINGLE_BIT):
        return board

    # Solve with advanced backtracking
    if solve_hyper_mrv(board, possible, BIT_COUNT, SINGLE_BIT):
        return board
    return board


def eliminate_value_hyper(value, row, col, possible):
    """Eliminate value from all peer cells with optimized loops"""
    bit_mask = ~(1 << (value - 1))
    possible[row][col] = 0

    # Eliminate from row and column in single loop
    for i in range(9):
        possible[row][i] &= bit_mask  # Row
        possible[i][col] &= bit_mask  # Column

    # Eliminate from box
    box_r, box_c = (row // 3) * 3, (col // 3) * 3
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            possible[r][c] &= bit_mask


def hyper_propagate(board, possible, bit_count, single_bit):
    """Advanced constraint propagation with multiple techniques"""
    changed = True
    while changed:
        changed = False

        # Naked singles (cells with one possibility)
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    bits = possible[row][col]
                    if bits == 0:
                        return False
                    if single_bit[bits]:
                        digit = single_bit[bits]
                        board[row][col] = digit
                        eliminate_value_hyper(digit, row, col, possible)
                        changed = True
                        break
            if changed:
                break

        if not changed:
            # Hidden singles with optimized unit checking
            changed = find_hidden_singles_hyper(board, possible)

        if not changed:
            # Naked pairs/triples
            changed = eliminate_naked_pairs_hyper(board, possible, bit_count)

        if not changed:
            # Pointing pairs (box-line reduction)
            changed = pointing_pairs_reduction_hyper(board, possible)

    return True


def find_hidden_singles_hyper(board, possible):
    """Find hidden singles with optimized unit iteration"""
    changed = False

    # Check rows
    for row in range(9):
        for digit in range(1, 10):
            bit = 1 << (digit - 1)
            candidates = [col for col in range(9) if possible[row][col] & bit]
            if len(candidates) == 1:
                col = candidates[0]
                if board[row][col] == 0:
                    board[row][col] = digit
                    eliminate_value_hyper(digit, row, col, possible)
                    changed = True

    # Check columns
    for col in range(9):
        for digit in range(1, 10):
            bit = 1 << (digit - 1)
            candidates = [row for row in range(9) if possible[row][col] & bit]
            if len(candidates) == 1:
                row = candidates[0]
                if board[row][col] == 0:
                    board[row][col] = digit
                    eliminate_value_hyper(digit, row, col, possible)
                    changed = True

    # Check boxes
    for box_r in range(3):
        for box_c in range(3):
            for digit in range(1, 10):
                bit = 1 << (digit - 1)
                candidates = []
                for r in range(3):
                    for c in range(3):
                        row, col = box_r * 3 + r, box_c * 3 + c
                        if possible[row][col] & bit:
                            candidates.append((row, col))

                if len(candidates) == 1:
                    row, col = candidates[0]
                    if board[row][col] == 0:
                        board[row][col] = digit
                        eliminate_value_hyper(digit, row, col, possible)
                        changed = True

    return changed


def eliminate_naked_pairs_hyper(board, possible, bit_count):
    """Eliminate naked pairs and triples with precomputed bit counts"""
    changed = False

    # Helper function to process a unit
    def process_unit(unit_cells):
        nonlocal changed
        # Find cells with 2-3 possibilities
        candidates = {}
        for row, col in unit_cells:
            if board[row][col] == 0:
                bits = possible[row][col]
                count = bit_count[bits]
                if 2 <= count <= 3:
                    if bits not in candidates:
                        candidates[bits] = []
                    candidates[bits].append((row, col))

        # Eliminate naked pairs/triples
        for bits, cells in candidates.items():
            count = bit_count[bits]
            if len(cells) == count:  # Found naked pair/triple
                # Remove these bits from other cells in unit
                for row, col in unit_cells:
                    if (row, col) not in cells and board[row][col] == 0:
                        old_bits = possible[row][col]
                        possible[row][col] &= ~bits
                        if possible[row][col] != old_bits:
                            changed = True

    # Process all units
    # Rows
    for r in range(9):
        process_unit([(r, c) for c in range(9)])
    # Columns
    for c in range(9):
        process_unit([(r, c) for r in range(9)])
    # Boxes
    for box_r in range(3):
        for box_c in range(3):
            unit = []
            for r in range(3):
                for c in range(3):
                    unit.append((box_r * 3 + r, box_c * 3 + c))
            process_unit(unit)

    return changed


def pointing_pairs_reduction_hyper(board, possible):
    """Optimized box-line reduction"""
    changed = False

    for box_r in range(3):
        for box_c in range(3):
            for digit in range(1, 10):
                bit = 1 << (digit - 1)
                candidates = []

                # Find candidates in box
                for r in range(3):
                    for c in range(3):
                        row, col = box_r * 3 + r, box_c * 3 + c
                        if possible[row][col] & bit:
                            candidates.append((row, col))

                if len(candidates) >= 2:
                    # Check row alignment
                    rows = {r for r, c in candidates}
                    if len(rows) == 1:
                        target_row = next(iter(rows))
                        for col in range(9):
                            if col < box_c * 3 or col >= (box_c + 1) * 3:  # Outside box
                                old_bits = possible[target_row][col]
                                possible[target_row][col] &= ~bit
                                if possible[target_row][col] != old_bits:
                                    changed = True

                    # Check column alignment
                    cols = {c for r, c in candidates}
                    if len(cols) == 1:
                        target_col = next(iter(cols))
                        for row in range(9):
                            if row < box_r * 3 or row >= (box_r + 1) * 3:  # Outside box
                                old_bits = possible[row][target_col]
                                possible[row][target_col] &= ~bit
                                if possible[row][target_col] != old_bits:
                                    changed = True

    return changed


def solve_hyper_mrv(board, possible, bit_count, single_bit):
    """Hyper-optimized MRV with constraint ordering"""
    # Apply propagation
    if not hyper_propagate(board, possible, bit_count, single_bit):
        return False

    # Find best cell using enhanced MRV
    best_cell = None
    min_possibilities = 10

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                bits = possible[row][col]
                if bits == 0:
                    return False

                possibilities = bit_count[bits]
                if possibilities < min_possibilities:
                    min_possibilities = possibilities
                    best_cell = (row, col)
                    if possibilities == 2:  # Early exit for binary choice
                        break
        if best_cell and min_possibilities == 2:
            break

    if best_cell is None:
        return True  # Solved

    row, col = best_cell
    bits = possible[row][col]

    # Try each possible digit
    for digit in range(1, 10):
        if bits & (1 << (digit - 1)):
            # Save state efficiently
            old_board = [r[:] for r in board]
            old_possible = [r[:] for r in possible]

            # Make move
            board[row][col] = digit
            eliminate_value_hyper(digit, row, col, possible)

            # Immediate conflict check
            conflict = False
            for r in range(9):
                for c in range(9):
                    if board[r][c] == 0 and possible[r][c] == 0:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict and solve_hyper_mrv(board, possible, bit_count, single_bit):
                return True

            # Backtrack
            for i in range(9):
                for j in range(9):
                    board[i][j] = old_board[i][j]
                    possible[i][j] = old_possible[i][j]

    return False


# Alternative implementation with bitwise operations for even better performance
def solveSudokuBitwise(board):
    """Ultra-optimized version using bitwise operations"""
    # Convert to bitwise representation
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    empty_cells = []

    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                empty_cells.append((i, j))
            else:
                bit = 1 << (board[i][j] - 1)
                rows[i] |= bit
                cols[j] |= bit
                boxes[3 * (i // 3) + j // 3] |= bit

    solveBitwise(board, empty_cells, 0, rows, cols, boxes)
    return board


def solveBitwise(board, empty_cells, idx, rows, cols, boxes):
    if idx == len(empty_cells):
        return True

    row, col = empty_cells[idx]
    box_idx = 3 * (row // 3) + col // 3

    # Get available numbers (bits not set in any constraint)
    used = rows[row] | cols[col] | boxes[box_idx]
    available = ((1 << 9) - 1) ^ used  # XOR to get available positions

    # Try each available number
    for num in range(1, 10):
        bit = 1 << (num - 1)
        if available & bit:
            # Place number
            board[row][col] = num
            rows[row] |= bit
            cols[col] |= bit
            boxes[box_idx] |= bit

            if solveBitwise(board, empty_cells, idx + 1, rows, cols, boxes):
                return True

            # Backtrack
            board[row][col] = 0
            rows[row] &= ~bit
            cols[col] &= ~bit
            boxes[box_idx] &= ~bit

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


# Test the functions
if __name__ == "__main__":
    board = generate_board()
    print("Original board:")
    for row in board:
        print(row)

    print("\nSolved with original solver:")
    solved_original = solveSudoku([row[:] for row in board])
    for row in solved_original:
        print(row)

    print("\nSolved with optimized solver:")
    solved_optimized = solveSudokuOptimized([row[:] for row in board])
    for row in solved_optimized:
        print(row)

    print("\nSolved with ultra-optimized solver:")
    solved_ultra = solveSudokuUltraOptimized([row[:] for row in board])
    for row in solved_ultra:
        print(row)

    print("\nSolved with hyper-optimized solver:")
    solved_hyper = solveSudokuHyperOptimized([row[:] for row in board])
    for row in solved_hyper:
        print(row)

    print("\nSolved with bitwise solver:")
    solved_bitwise = solveSudokuBitwise([row[:] for row in board])
    for row in solved_bitwise:
        print(row)
