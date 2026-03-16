#=
test_search.jl — Verify search correctness.

Run with:
    julia test/test_search.jl

Tests:
  1. Perft — counts leaf nodes at increasing depth to verify move generation.
     Expected values for the starting position are universally agreed upon;
     any deviation means a bug in move generation or apply/undo.

  2. Search sanity — verifies best_move returns a legal move and sane score.

  3. Mate in 1 — the engine must find checkmate in a trivial forced position.
     Tests that evaluation and search interact correctly at terminal nodes.
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search
using Printf

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

# ── Perft ──────────────────────────────────────────────────────
# These are the canonical perft values from the starting position.
# If any of these fail, there is a bug in move generation.
println("\n═══ Perft (move generation correctness) ═══\n")
println("  (Note: depth 4 triggers JIT compilation — first run may be slow.)\n")

b = new_board()
test("perft(1) = 20",     perft(b, 1) == 20)
test("perft(2) = 400",    perft(b, 2) == 400)
test("perft(3) = 8902",   perft(b, 3) == 8902)
test("perft(4) = 197281", perft(b, 4) == 197281)

# ── Search sanity ──────────────────────────────────────────────
println("\n═══ Search sanity ═══\n")

b2 = new_board()
m, score = best_move(b2; max_depth=3, verbose=true)
println()
test("best_move returns a Move",    m isa Move)
test("score is a Float64",          score isa Float64)
test("score is finite",             isfinite(score))
test("chosen move is legal",        m ∈ generate_moves(b2))

# ── Mate in 1 ──────────────────────────────────────────────────
# Position: White king e6, Black king e8, White queen d7.
# White to move. Qd8 is checkmate: queen covers the whole 8th rank,
# and White's king covers all escape squares (d7, e7, f7).
println("\n═══ Mate in 1 ═══\n")

b3 = new_board()
b3.grid .= 0.0
b3.grid[6, 5] = KING     # White king on e6
b3.grid[8, 5] = -KING    # Black king on e8
b3.grid[7, 4] = QUEEN    # White queen on d7  → Qd8#
b3.turn = WHITE
b3.castling = [false, false, false, false]
sync_board!(b3)

m3, s3 = best_move(b3; max_depth=3, verbose=true)
println()

# Verify the chosen move actually delivers checkmate (not just any move).
undo3 = apply_move!(b3, m3)
move_is_mate = is_checkmate(b3)
undo_move!(b3, m3, undo3)

test("Chosen move delivers checkmate", move_is_mate)
test("Score ≥ CHECKMATE_SCORE",        s3 >= 9000.0)

# ── Quiescence while in check ──────────────────────────────────
# At depth 0, qsearch must not "stand pat" in a checked position.
# This position has only quiet king escapes, so the old bug returned
# the static eval instead of searching evasions.
println("\n═══ Quiescence in Check ═══\n")

b4 = from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1")
field4 = compute_total_field(b4)
w4 = [
    Energy.W_MATERIAL,
    Energy.W_FIELD,
    Energy.W_KING_SAFETY,
    Energy.W_TENSION,
    Energy.W_MOBILITY,
]
stand_pat4 = Float64(b4.turn) * eval_w(b4, w4, field4)

test("Position is in check", is_in_check(b4, WHITE))
test("All legal evasions are quiet moves", all(m -> !m.is_en_passant &&
                                               piece_at(b4, m.to_rank, m.to_file) == 0.0,
                                               generate_moves(b4)))
q4 = Search.qsearch(b4, w4, -Search.INF, Search.INF, field4, 1, new_tt())
# In check, qsearch must search evasions rather than stand pat.
# If it stood pat, it would return -INF (since stand_pat is set to -INF
# when in_check is true). A finite score proves it searched evasions.
test("Qsearch does not stand pat while in check", q4 > -9000.0)

# Qsearch must also score a checkmated leaf as mate, not static eval.
b5 = from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
field5 = compute_total_field(b5)
q5 = Search.qsearch(b5, w4, -Search.INF, Search.INF, field5, 1, new_tt())
test("Checked side with no legal moves gets mate score", q5 <= -9000.0)

# ── Summary ────────────────────────────────────────────────────
println("\n═══════════════════════════════════════════")
println("  Results: $passed passed, $failed failed")
failed == 0 ? println("  All tests passed! ✓") :
              println("  SOME TESTS FAILED ✗")
println("═══════════════════════════════════════════\n")
failed == 0 || exit(1)
