#=
perft_suite.jl — Comprehensive move generation correctness tests.

Perft (performance test) counts all leaf nodes at a given depth from
a position. The expected values are universally agreed upon by the
chess programming community. Any deviation means a bug in:
  - pseudo-legal move generation
  - legality filtering (leaving king in check)
  - en passant handling
  - castling through/out-of/into check
  - promotion (all four piece types)
  - apply_move! / undo_move! state corruption

This suite tests six standard positions that together exercise every
tricky corner of the rules. If all of these pass, move generation
is almost certainly correct.

Run with:
    julia test/perft_suite.jl

References:
    https://www.chessprogramming.org/Perft_Results
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Search
using Printf

# ── Test harness ──────────────────────────────────────────────
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

# ── Helper: run perft and report ──────────────────────────────
# Runs perft at the given depth and checks against expected count.
# Prints the time taken so you can spot performance regressions.
function perft_test(label::String, fen::String, depth::Int, expected::Int)
    b = from_fen(fen)
    t = @elapsed result = perft(b, depth)
    ok = result == expected
    if ok
        @printf("  ✓ %-50s  depth=%d  nodes=%-10d  (%.3fs)\n",
                label, depth, result, t)
    else
        @printf("  ✗ %-50s  depth=%d  expected=%d  got=%d  ← FAILED\n",
                label, depth, expected, result)
    end
    global passed, failed
    ok ? (passed += 1) : (failed += 1)
    return ok
end


# ═══════════════════════════════════════════════════════════════
# Position 1: Starting Position
# ═══════════════════════════════════════════════════════════════
# The baseline. If these fail, nothing else matters.
# Tests basic piece movement, pawn double push, simple captures.

println("\n═══ Position 1: Starting Position ═══\n")

const POS1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

perft_test("Start — perft(1)",  POS1, 1, 20)
perft_test("Start — perft(2)",  POS1, 2, 400)
perft_test("Start — perft(3)",  POS1, 3, 8902)
perft_test("Start — perft(4)",  POS1, 4, 197281)


# ═══════════════════════════════════════════════════════════════
# Position 2: "Kiwipete"  (Peter McKenzie)
# ═══════════════════════════════════════════════════════════════
# The gold standard for tricky move generation. Contains:
#   - Castling rights on both sides
#   - Pinned pieces
#   - En passant possibilities
#   - Promotion threats
#   - Bishops on long diagonals
#   - Multiple knights creating fork potential
# If your engine passes Kiwipete perft, it handles most edge cases.

println("\n═══ Position 2: Kiwipete ═══\n")

const POS2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

perft_test("Kiwipete — perft(1)",  POS2, 1, 48)
perft_test("Kiwipete — perft(2)",  POS2, 2, 2039)
perft_test("Kiwipete — perft(3)",  POS2, 3, 97862)
perft_test("Kiwipete — perft(4)",  POS2, 4, 4085603)


# ═══════════════════════════════════════════════════════════════
# Position 3: En Passant + Edge Cases
# ═══════════════════════════════════════════════════════════════
# Sparse endgame with:
#   - En passant that would be illegal (exposes king to check)
#   - Rook vs pawn endgame tension
#   - King on the a-file (edge of board)
#   - Pawns that can promote soon
# This is where many engines trip on en passant legality.

println("\n═══ Position 3: En Passant Edge Cases ═══\n")

const POS3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"

perft_test("EP edges — perft(1)",  POS3, 1, 14)
perft_test("EP edges — perft(2)",  POS3, 2, 191)
perft_test("EP edges — perft(3)",  POS3, 3, 2812)
perft_test("EP edges — perft(4)",  POS3, 4, 43238)
perft_test("EP edges — perft(5)",  POS3, 5, 674624)


# ═══════════════════════════════════════════════════════════════
# Position 4: Castling + Promotion Stress
# ═══════════════════════════════════════════════════════════════
# A chaotic midgame position with:
#   - Black can still castle (kq) but White cannot
#   - White has a pawn on b7 about to promote (4 promotion choices)
#   - Black has a pawn on b2 about to promote
#   - Multiple pins and x-ray attacks
#   - Knights, bishops, queen all active
# Tests that castling rights are correctly revoked on rook
# captures and that all four promotion types are generated.

println("\n═══ Position 4: Castling + Promotion Stress ═══\n")

const POS4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"

perft_test("Castle+promo — perft(1)",  POS4, 1, 6)
perft_test("Castle+promo — perft(2)",  POS4, 2, 264)
perft_test("Castle+promo — perft(3)",  POS4, 3, 9467)
perft_test("Castle+promo — perft(4)",  POS4, 4, 422333)


# ═══════════════════════════════════════════════════════════════
# Position 5: Promotion-Heavy (No Castling for Black King)
# ═══════════════════════════════════════════════════════════════
# White has a pawn on d7 that can promote with check.
# Black has a knight on f2 attacking the White king.
# Tests:
#   - Promotion with check vs without check
#   - Interaction between promotion and castling rights
#   - Black king on f8 (moved, so no castling)
#   - White still has KQ castling rights

println("\n═══ Position 5: Promotion-Heavy ═══\n")

const POS5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"

perft_test("Promo-heavy — perft(1)",  POS5, 1, 44)
perft_test("Promo-heavy — perft(2)",  POS5, 2, 1486)
perft_test("Promo-heavy — perft(3)",  POS5, 3, 62379)
perft_test("Promo-heavy — perft(4)",  POS5, 4, 2103487)


# ═══════════════════════════════════════════════════════════════
# Position 6: Mirror Position (Symmetry Test)
# ═══════════════════════════════════════════════════════════════
# A perfectly symmetric position (after color swap). Both sides
# have identical piece structures. Tests:
#   - No castling rights (both kings have castled)
#   - Pin detection on both sides
#   - Bishop pair vs bishop pair
#   - Many tactical possibilities at every depth
# Good for catching color-dependent bugs in move generation.

println("\n═══ Position 6: Mirror Symmetry ═══\n")

const POS6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

perft_test("Mirror — perft(1)",  POS6, 1, 46)
perft_test("Mirror — perft(2)",  POS6, 2, 2079)
perft_test("Mirror — perft(3)",  POS6, 3, 89890)
perft_test("Mirror — perft(4)",  POS6, 4, 3894594)


# ═══════════════════════════════════════════════════════════════
# FEN Round-Trip Sanity Check
# ═══════════════════════════════════════════════════════════════
# Verify that from_fen produces a board that matches new_board()
# for the starting position. Catches FEN parser bugs.

println("\n═══ FEN Parser Sanity ═══\n")

b_new = new_board()
b_fen = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

test("FEN grid matches new_board()",         b_fen.grid == b_new.grid)
test("FEN turn matches new_board()",         b_fen.turn == b_new.turn)
test("FEN castling matches new_board()",     b_fen.castling == b_new.castling)
test("FEN en_passant matches new_board()",   b_fen.en_passant == b_new.en_passant)
test("FEN halfmove matches new_board()",     b_fen.halfmove == b_new.halfmove)
test("FEN fullmove matches new_board()",     b_fen.fullmove == b_new.fullmove)
test("FEN Zobrist matches new_board()",      b_fen.hash == b_new.hash)
test("FEN material matches new_board()",     b_fen.material == b_new.material)
test("FEN white_king matches new_board()",   b_fen.white_king == b_new.white_king)
test("FEN black_king matches new_board()",   b_fen.black_king == b_new.black_king)


# ═══════════════════════════════════════════════════════════════
# Apply/Undo Integrity Check
# ═══════════════════════════════════════════════════════════════
# For each perft position, apply every legal move and undo it,
# verifying the board returns to exactly the same state. This
# catches subtle bugs where apply_move!/undo_move! corrupt
# the grid, Zobrist hash, castling rights, or king coordinates.

println("\n═══ Apply/Undo Integrity ═══\n")

for (label, fen) in [("Start",        POS1),
                      ("Kiwipete",     POS2),
                      ("EP edges",     POS3),
                      ("Castle+promo", POS4),
                      ("Promo-heavy",  POS5),
                      ("Mirror",       POS6)]
    b = from_fen(fen)
    original_grid      = copy(b.grid)
    original_hash      = b.hash
    original_castling  = copy(b.castling)
    original_ep        = b.en_passant
    original_material  = b.material
    original_wk        = b.white_king
    original_bk        = b.black_king
    original_halfmove  = b.halfmove
    original_hist_len  = length(b.history)

    moves = generate_moves(b)
    all_ok = true
    for m in moves
        undo = apply_move!(b, m)
        undo_move!(b, m, undo)

        if b.grid != original_grid || b.hash != original_hash ||
           b.castling != original_castling || b.en_passant != original_ep ||
           b.material != original_material || b.white_king != original_wk ||
           b.black_king != original_bk || b.halfmove != original_halfmove ||
           length(b.history) != original_hist_len
            all_ok = false
            println("    ← $(move_to_string(m)) corrupted state")
            break
        end
    end
    test("$label: apply/undo preserves all state ($(length(moves)) moves)", all_ok)
end


# ═══════════════════════════════════════════════════════════════
# Repetition Detection
# ═══════════════════════════════════════════════════════════════
# Verify that is_repetition correctly detects when the engine
# has returned to a previously seen position.
#
# Test sequence: 1. Nf3 Nf6 2. Ng1 Ng8 — both knights return
# home, restoring the exact starting position. After move 4
# (Ng8), the hash should match the starting position hash stored
# in history, so is_repetition must return true.

println("\n═══ Repetition Detection ═══\n")

# Helper: find a legal move by UCI string, or error
function find_move(b::Board, uci::String)::Move
    for m in generate_moves(b)
        move_to_string(m) == uci && return m
    end
    error("Could not find legal move: $uci")
end

let
    b = new_board()
    test("Fresh board: no repetition", !is_repetition(b))

    # Play 1. Nf3 Nf6 2. Ng1 Ng8 — a knight dance back to the start
    apply_move!(b, find_move(b, "g1f3"))
    test("After 1. Nf3: no repetition", !is_repetition(b))

    apply_move!(b, find_move(b, "g8f6"))
    test("After 1... Nf6: no repetition", !is_repetition(b))

    apply_move!(b, find_move(b, "f3g1"))
    test("After 2. Ng1: no repetition", !is_repetition(b))

    apply_move!(b, find_move(b, "f6g8"))
    test("After 2... Ng8: REPETITION detected", is_repetition(b))

    # History should have exactly 4 entries (one per apply_move! call)
    test("History has 4 entries", length(b.history) == 4)

    # The current hash should match the starting position hash
    # (which is the first entry in history)
    test("Current hash matches starting position", b.hash == b.history[1])
end

# Verify that captures break the repetition window.
# After a capture, halfmove resets to 0, so is_repetition only
# looks at positions since the capture — earlier ones are unreachable.
let
    b = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # 1. e4 — pawn push resets halfmove to 0
    apply_move!(b, find_move(b, "e2e4"))
    test("After pawn push: halfmove is 0", b.halfmove == 0)
    test("After pawn push: start hash in history but not reachable",
         !is_repetition(b))
end

# Verify is_game_over includes repetition
let
    b = new_board()
    test("Fresh board: game not over", !is_game_over(b))

    # Play the knight dance again: 1. Nf3 Nf6 2. Ng1 Ng8
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]
        apply_move!(b, find_move(b, uci))
    end
    test("After repetition: is_game_over returns true", is_game_over(b))
end


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

println("\n═══════════════════════════════════════════════════════")
println("  Results: $passed passed, $failed failed")
if failed == 0
    println("  All tests passed! ✓")
    println("  Move generation and repetition detection verified.")
else
    println("  SOME TESTS FAILED ✗")
    println("  Run perft_divide on failing positions to isolate the bug.")
end
println("═══════════════════════════════════════════════════════\n")
failed == 0 || exit(1)
