#=
test_state.jl — Verify the board math and move generation.

Run this with:
    julia test/test_state.jl

These tests ensure:
1. The starting matrix is set up correctly
2. Legal moves are generated properly
3. Apply/undo leaves the board unchanged
4. Check, checkmate, and stalemate detection work

If any of these fail, nothing else matters — the math is
built on top of correct chess rules.
=#

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Printf

include(joinpath(@__DIR__, "..", "src", "state.jl"))
using .State

passed = 0
failed = 0

function test(name, condition)
    global passed, failed
    if condition
        passed += 1
        println("  ✓ $name")
    else
        failed += 1
        println("  ✗ $name  ← FAILED")
    end
end

println("\n═══ Testing State Representation ═══\n")

# ── Test 1: Starting position matrix values ────────────────────
println("Board matrix values:")
b = new_board()

test("White king at e1 = +100.0",  piece_at(b, 1, 5) == 100.0)
test("Black king at e8 = -100.0",  piece_at(b, 8, 5) == -100.0)
test("White queen at d1 = +9.0",   piece_at(b, 1, 4) == 9.0)
test("Black queen at d8 = -9.0",   piece_at(b, 8, 4) == -9.0)
test("White pawn at e2 = +1.0",    piece_at(b, 2, 5) == 1.0)
test("Black pawn at e7 = -1.0",    piece_at(b, 7, 5) == -1.0)
test("Empty square e4 = 0.0",      piece_at(b, 4, 5) == 0.0)
test("White rook at a1 = +5.0",    piece_at(b, 1, 1) == 5.0)
test("Black knight at b8 = -3.0",  piece_at(b, 8, 2) == -3.0)

# ── Test 2: Matrix symmetry ───────────────────────────────────
# The starting position should be symmetric: for each White piece
# at (r,f), there's a Black piece of equal magnitude at (9-r,f).
println("\nMatrix symmetry:")
symmetric = true
for r in 1:2, f in 1:8
    if abs(b.grid[r, f]) != abs(b.grid[9-r, f])
        global symmetric = false
        break
    end
end
test("Starting position is symmetric", symmetric)

# Total charge should be zero (equal material)
total = sum(b.grid)
test("Total matrix charge = 0 (balanced)", abs(total) < 0.001)

# ── Test 3: Legal moves from starting position ────────────────
println("\nMove generation:")
moves = generate_moves(b)
test("Starting position has 20 legal moves", length(moves) == 20)

# All moves should be by White (it's White's turn)
all_white = all(m -> piece_color(b, m.from_rank, m.from_file) == WHITE, moves)
test("All moves are by White", all_white)

# ── Test 4: Apply and undo ────────────────────────────────────
println("\nApply/Undo invariance:")
b2 = new_board()
original_grid = copy(b2.grid)

# Apply e2-e4 (pawn from rank 2 file 5 to rank 4 file 5)
e2e4 = Move(2, 5, 4, 5)
undo = apply_move!(b2, e2e4)

test("After e4: e2 is empty", is_empty(b2, 2, 5))
test("After e4: e4 has pawn", piece_at(b2, 4, 5) == PAWN)
test("After e4: it's Black's turn", b2.turn == BLACK)

undo_move!(b2, e2e4, undo)

test("After undo: grid is restored", b2.grid == original_grid)
test("After undo: it's White's turn again", b2.turn == WHITE)

# ── Test 5: Check detection ───────────────────────────────────
println("\nCheck detection:")
# Set up a position where Black king is in check
b3 = new_board()
b3.grid .= 0.0  # Clear the board
b3.grid[1, 5] = KING      # White king on e1
b3.grid[8, 5] = -KING     # Black king on e8
b3.grid[4, 5] = QUEEN     # White queen on e4 — checking Black king!
b3.turn = BLACK
sync_board!(b3)

test("Black is in check (queen on e4)", is_in_check(b3, BLACK))
test("White is NOT in check", !is_in_check(b3, WHITE))

# ── Test 6: Checkmate detection ───────────────────────────────
println("\nCheckmate detection:")
b4 = new_board()
b4.grid .= 0.0
b4.grid[1, 1] = KING      # White king on a1
b4.grid[8, 8] = -KING     # Black king on h8
b4.grid[7, 1] = ROOK      # White rook on a7
b4.grid[8, 2] = QUEEN     # White queen on b8 — checkmate!
b4.turn = BLACK
b4.castling = [false, false, false, false]
sync_board!(b4)

test("Black is in checkmate", is_checkmate(b4))

# ── Test 7: Stalemate detection ───────────────────────────────
println("\nStalemate detection:")
b5 = new_board()
b5.grid .= 0.0
b5.grid[1, 1] = KING      # White king on a1
b5.grid[8, 8] = -KING     # Black king on h8 (corner)
b5.grid[7, 6] = QUEEN     # White queen on f7 — stalemate!
b5.turn = BLACK
b5.castling = [false, false, false, false]
sync_board!(b5)

test("Black is stalemated", is_stalemate(b5))
test("Black is NOT in checkmate", !is_checkmate(b5))

# ── Test 8: En passant ────────────────────────────────────────
println("\nEn passant:")
b6 = new_board()
b6.grid .= 0.0
b6.grid[1, 5] = KING
b6.grid[8, 5] = -KING
b6.grid[5, 5] = PAWN       # White pawn on e5
b6.grid[7, 4] = -PAWN      # Black pawn on d7
b6.turn = BLACK
b6.castling = [false, false, false, false]
sync_board!(b6)

# Black plays d7-d5
d7d5 = Move(7, 4, 5, 4)
undo6 = apply_move!(b6, d7d5)

test("En passant target set after d5", b6.en_passant == (6, 4))

# White should now be able to capture en passant
moves6 = generate_moves(b6)
ep_moves = filter(m -> m.is_en_passant, moves6)
test("White has en passant capture available", length(ep_moves) == 1)

# ── Test 9: is_game_over terminates correctly ─────────────────
println("\nis_game_over:")

# 50-move rule: position with halfmove = 100 must be game over
b7 = new_board()
b7.halfmove = 100
test("is_game_over true at halfmove=100", is_game_over(b7))

# Normal starting position is not game over
b8 = new_board()
test("is_game_over false at start", !is_game_over(b8))

# ── Summary ────────────────────────────────────────────────────
println("\n═══════════════════════════════════")
println("  Results: $passed passed, $failed failed")
if failed == 0
    println("  All tests passed! ✓")
else
    println("  SOME TESTS FAILED ✗")
end
println("═══════════════════════════════════\n")
failed == 0 || exit(1)

# Show the starting board
println("Starting position as matrix:")
print_board(new_board())

# Show the raw matrix values
println("Raw matrix (positive=White, negative=Black):")
b = new_board()
for r in 8:-1:1
    print("  rank $r: ")
    for f in 1:8
        v = b.grid[r, f]
        if v == 0.0
            print("   .  ")
        else
            @printf("%+6.1f", v)
        end
    end
    println()
end
println()
