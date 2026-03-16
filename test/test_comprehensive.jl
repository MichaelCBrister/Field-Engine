#=
test_comprehensive.jl — Deep correctness tests for search, state, and energy.

Covers:
  1. Perft validation against 10 FENs (Stockfish-verified node counts)
  2. Evaluation symmetry (eval(pos) + eval(flipped) ≈ 0 for color-swapped positions)
  3. Repetition detection (3-fold via forced move sequences, edge cases)
  4. Transposition table consistency (same position via different move orders)
  5. Move ordering sanity (no illegal moves in generated lists)
  6. Zobrist hash correctness (incremental vs full recompute)
  7. Search termination edge cases (mate detection, qsearch depth)
  8. Zero-alloc buffer management (buffer sizes don't grow unexpectedly)

Run with:
    julia test/test_comprehensive.jl
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search
using Printf

passed = 0
failed = 0
errors = String[]

function test(name, condition)
    global passed, failed, errors
    if condition
        passed += 1
        println("  ✓ $name")
    else
        failed += 1
        push!(errors, name)
        println("  ✗ $name  ← FAILED")
    end
end

# Helper: find a legal move by UCI string
function find_move(b::Board, uci::String)::Move
    for m in generate_moves(b)
        move_to_string(m) == uci && return m
    end
    error("Could not find legal move: $uci in position")
end

# Helper: flip a board (swap colors, mirror ranks)
# Creates the color-symmetric position: White↔Black, rank r → rank 9-r
function flip_board(b::Board)::Board
    fb = Board(
        zeros(Float64, 8, 8),
        -b.turn,
        [b.castling[3], b.castling[4], b.castling[1], b.castling[2]],
        b.en_passant == (0,0) ? (0,0) : (9 - b.en_passant[1], b.en_passant[2]),
        b.halfmove,
        b.fullmove,
        0.0, (0,0), (0,0), UInt64(0), UInt64[]
    )
    for r in 1:8, f in 1:8
        fb.grid[9-r, f] = -b.grid[r, f]
    end
    sync_board!(fb)
    return fb
end


# ═══════════════════════════════════════════════════════════════
# 1. PERFT VALIDATION — 10 FENs with Stockfish-verified counts
# ═══════════════════════════════════════════════════════════════
println("\n═══ 1. Perft Validation (10 positions) ═══\n")

# Standard perft suite from chessprogramming.org
const PERFT_SUITE = [
    # (label, FEN, depth, expected_nodes)

    # Position 1: Starting position
    ("Start d3",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     3, 8902),

    ("Start d4",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     4, 197281),

    # Position 2: Kiwipete — pins, en passant, castling
    ("Kiwipete d3",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     3, 97862),

    # Position 3: En passant edge cases
    ("EP edges d4",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     4, 43238),

    ("EP edges d5",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     5, 674624),

    # Position 4: Castling + promotion stress
    ("Castle+promo d3",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     3, 9467),

    # Position 5: Promotion-heavy
    ("Promo-heavy d3",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     3, 62379),

    # Position 6: Mirror symmetry
    ("Mirror d3",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     3, 89890),

    # Position 7: Italian Game — open center, pins, development
    ("Italian d3",
     "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     3, 37080),

    # Position 8: Pawn endgame — king activity, passed pawns
    ("Pawn endgame d4",
     "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
     4, 396),
]

for (label, fen, depth, expected) in PERFT_SUITE
    b = from_fen(fen)
    t = @elapsed result = perft(b, depth)
    ok = result == expected
    if ok
        @printf("  ✓ %-25s  depth=%d  nodes=%-10d  (%.3fs)\n", label, depth, result, t)
        global passed += 1
    else
        @printf("  ✗ %-25s  depth=%d  expected=%d  got=%d  ← FAILED\n",
                label, depth, expected, result)
        global failed += 1
        push!(errors, "Perft: $label")
    end
end


# ═══════════════════════════════════════════════════════════════
# 2. EVALUATION SYMMETRY — eval(pos) + eval(flipped) ≈ 0
# ═══════════════════════════════════════════════════════════════
println("\n═══ 2. Evaluation Symmetry ═══\n")

# Test positions covering diverse scenarios
const SYMMETRY_FENS = [
    # Starting position — perfectly symmetric
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # After 1. e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # After 1. e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    # Italian Game
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Sicilian Defense
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    # Queen's Gambit
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
    # Ruy Lopez
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # French Defense
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # King's Indian
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    # Caro-Kann
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # Endgame: R+K vs K
    "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
    # Endgame: Q vs R
    "4k3/8/8/8/8/8/r7/Q3K3 w - - 0 1",
    # Endgame: symmetric pawns
    "4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1",
    # Complex middlegame
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    # Opposite-side castling
    "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 4 7",
    # Pawn chains
    "r1bqkbnr/pp3ppp/2n1p3/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 5",
    # London System
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
    # Scandinavian
    "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
    # Mixed endgame with bishops
    "2b1k3/8/8/8/8/8/8/3BK3 w - - 0 1",
    # Rook endgame
    "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1",
    # Material imbalance: bishop pair vs rook
    "4k3/8/8/8/8/8/r7/2B1KB2 w - - 0 1",
    # Passed pawns
    "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
    # Doubled pawns
    "4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1",
    # Knight outpost
    "4k3/8/3N4/8/8/8/8/4K3 w - - 0 1",
    # Bishops of same color
    "4k3/8/5b2/8/8/2B5/8/4K3 w - - 0 1",
    # Double rook endgame
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    # Queen and knight
    "4k3/8/8/8/8/8/8/Q1N1K3 w - - 0 1",
    # Fianchetto position
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # Isolated queen pawn
    "rnbqkbnr/ppp2ppp/4p3/3P4/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    # Middlegame with tension
    "r1bqk2r/ppp2ppp/2n2n2/3pp3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq d6 0 5",
    # 50 positions: add more varied endgames
    "8/8/4k3/8/8/8/4K3/4Q3 w - - 0 1",
    "8/8/4k3/8/3N4/8/4K3/8 w - - 0 1",
    "8/3pk3/8/8/8/8/3PK3/8 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1",
    "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
    "2r2rk1/pp1bqppp/2n1pn2/3p4/3P4/1PN1PN2/PBQ2PPP/2R2RK1 w - - 0 12",
    "r1b1k2r/ppppqppp/2n2n2/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 6",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
]

sym_pass = 0
sym_fail = 0
sym_max_err = 0.0
sym_worst_fen = ""

for fen in SYMMETRY_FENS
    b = from_fen(fen)
    fb = flip_board(b)

    e1 = evaluate(b)
    e2 = evaluate(fb)

    # eval(pos) should be -eval(flipped_pos) for a symmetric evaluation
    err = abs(e1 + e2)
    if err > sym_max_err
        global sym_max_err = err
        global sym_worst_fen = fen
    end

    if err < 0.01
        global sym_pass += 1
    else
        global sym_fail += 1
        @printf("  ✗ Symmetry violation: |%.4f + %.4f| = %.4f  FEN: %s\n", e1, e2, err, fen)
    end
end

test("Symmetry: $sym_pass/$(length(SYMMETRY_FENS)) positions within tolerance",
     sym_fail == 0)
if sym_fail > 0
    @printf("  → Worst error: %.6f at FEN: %s\n", sym_max_err, sym_worst_fen)
end
@printf("  → Max symmetry error across all positions: %.6f\n", sym_max_err)


# ═══════════════════════════════════════════════════════════════
# 3. REPETITION DETECTION — comprehensive edge cases
# ═══════════════════════════════════════════════════════════════
println("\n═══ 3. Repetition Detection ═══\n")

# 3a. Basic 2-fold via knight dance (used by search as draw)
let
    b = new_board()
    test("Fresh board: no repetition", !is_repetition(b))

    # 1. Nf3 Nf6 2. Ng1 Ng8 → returns to starting position
    apply_move!(b, find_move(b, "g1f3"))
    test("After 1. Nf3: no repetition", !is_repetition(b))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))
    apply_move!(b, find_move(b, "f6g8"))
    test("After Ng8 (2-fold): repetition detected", is_repetition(b))
    test("Hash matches first position", b.hash == b.history[1])
end

# 3b. True 3-fold: need TWO round trips
let
    b = new_board()
    start_hash = b.hash

    # First cycle
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))
    apply_move!(b, find_move(b, "f6g8"))

    test("3-fold: after 1st cycle, hash matches start", b.hash == start_hash)

    # Second cycle
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))
    apply_move!(b, find_move(b, "f6g8"))

    test("3-fold: after 2nd cycle, hash still matches start", b.hash == start_hash)
    test("3-fold: repetition detected after 2nd cycle", is_repetition(b))

    # Count occurrences in history
    count = sum(h == start_hash for h in b.history)
    test("3-fold: start hash appears twice in history", count == 2)
end

# 3c. Repetition broken by capture
let
    b = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # 1. e4 — pawn push resets halfmove
    apply_move!(b, find_move(b, "e2e4"))
    test("Pawn push: halfmove is 0", b.halfmove == 0)

    # Even after a long knight dance, the pawn push break prevents
    # positions from BEFORE the push from repeating
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "f6g8"))
    apply_move!(b, find_move(b, "f3g1"))

    # This position differs from start (e4 pawn is out), so no repetition
    test("Post-pawn-push knight dance: no repetition", !is_repetition(b))
end

# 3d. Repetition with rook shuffling (different piece type)
let
    b = from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    start_hash = b.hash

    # Ra1-a2, Ke8-d8, Ra2-a1, Kd8-e8 → returns to start
    apply_move!(b, find_move(b, "a1a2"))
    apply_move!(b, find_move(b, "e8d8"))
    apply_move!(b, find_move(b, "a2a1"))
    apply_move!(b, find_move(b, "d8e8"))

    test("Rook shuffle: hash matches start", b.hash == start_hash)
    test("Rook shuffle: repetition detected", is_repetition(b))
end

# 3e. Repetition NOT detected when en passant differs
let
    b = from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
    ep_hash = b.hash

    # Black can play d4-d3, then positions look the same but EP differs
    b2 = from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1")
    no_ep_hash = b2.hash

    # EP changes the hash
    test("EP flag changes Zobrist hash", ep_hash != no_ep_hash)
end

# 3f. Repetition in game_over check (requires 3-fold under FIDE rules)
let
    b = new_board()
    test("Fresh: game not over", !is_game_over(b))

    # First cycle: 2-fold — game continues (search sees draw, but game is not over)
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]
        apply_move!(b, find_move(b, uci))
    end
    test("After 2-fold: is_game_over returns false (FIDE)", !is_game_over(b))

    # Second cycle: 3-fold — game is over
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]
        apply_move!(b, find_move(b, uci))
    end
    test("After 3-fold: is_game_over returns true", is_game_over(b))
end

# 3g. Undo restores repetition state correctly
let
    b = new_board()
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))

    m = find_move(b, "f6g8")
    undo = apply_move!(b, m)
    test("Before undo: repetition detected", is_repetition(b))

    undo_move!(b, m, undo)
    test("After undo: repetition no longer detected", !is_repetition(b))
    test("After undo: history length restored", length(b.history) == 3)
end


# ═══════════════════════════════════════════════════════════════
# 4. TRANSPOSITION TABLE CONSISTENCY
# ═══════════════════════════════════════════════════════════════
println("\n═══ 4. Transposition Table Consistency ═══\n")

# 4a. Same position via different move orders → same Zobrist hash
# NOTE: must avoid pawn double-pushes which set different en passant state
let
    # Path A: 1. Nf3 Nc6 2. Nc3
    ba = new_board()
    apply_move!(ba, find_move(ba, "g1f3"))
    apply_move!(ba, find_move(ba, "b8c6"))
    apply_move!(ba, find_move(ba, "b1c3"))

    # Path B: 1. Nc3 Nc6 2. Nf3
    bb = new_board()
    apply_move!(bb, find_move(bb, "b1c3"))
    apply_move!(bb, find_move(bb, "b8c6"))
    apply_move!(bb, find_move(bb, "g1f3"))

    test("Transposition: different move orders → same hash",
         ba.hash == bb.hash)
    test("Transposition: different move orders → same grid",
         ba.grid == bb.grid)
    test("Transposition: different move orders → same material",
         ba.material == bb.material)
end

# 4b. Same position via different move orders → same evaluation
# Use knight moves only to avoid en passant hash differences
let
    # Path A: 1. Nf3 Nf6 2. Nc3 Nc6
    ba = new_board()
    apply_move!(ba, find_move(ba, "g1f3"))
    apply_move!(ba, find_move(ba, "g8f6"))
    apply_move!(ba, find_move(ba, "b1c3"))
    apply_move!(ba, find_move(ba, "b8c6"))

    # Path B: 1. Nc3 Nc6 2. Nf3 Nf6
    bb = new_board()
    apply_move!(bb, find_move(bb, "b1c3"))
    apply_move!(bb, find_move(bb, "b8c6"))
    apply_move!(bb, find_move(bb, "g1f3"))
    apply_move!(bb, find_move(bb, "g8f6"))

    ea = evaluate(ba)
    eb = evaluate(bb)
    test("Transposition: same position → same eval",
         abs(ea - eb) < 0.001)
    test("Transposition: same hash via different knight order",
         ba.hash == bb.hash)
end

# 4c. TT stores and retrieves correctly
let
    tt = new_tt()
    b = new_board()

    # Store a value
    Search.tt_store!(tt, b.hash, 3, 1.5, Search.TT_EXACT)

    # Probe with matching depth should return the value
    result = Search.tt_probe(tt, b.hash, 3, -1000.0, 1000.0)
    test("TT: exact probe returns stored value", result !== nothing && abs(result - 1.5) < 0.01)

    # Probe with deeper depth requirement should miss
    result2 = Search.tt_probe(tt, b.hash, 4, -1000.0, 1000.0)
    test("TT: probe with deeper requirement returns nothing", result2 === nothing)

    # Probe with shallower depth should hit
    result3 = Search.tt_probe(tt, b.hash, 2, -1000.0, 1000.0)
    test("TT: probe with shallower requirement hits", result3 !== nothing)
end

# 4d. TT lower/upper bound flags work correctly
let
    tt = new_tt()
    h = UInt64(0x12345678ABCDEF01)

    # Store lower bound (fail-high)
    Search.tt_store!(tt, h, 4, 5.0, Search.TT_LOWER)
    # Lower bound of 5.0: if β ≤ 5.0, should return 5.0
    result = Search.tt_probe(tt, h, 4, 0.0, 4.0)
    test("TT: lower bound with β=4 < 5 → returns 5.0",
         result !== nothing && abs(result - 5.0) < 0.01)
    # Lower bound of 5.0: if β > 5.0, should not cutoff
    result2 = Search.tt_probe(tt, h, 4, 0.0, 6.0)
    test("TT: lower bound with β=6 > 5 → nothing", result2 === nothing)

    # Store upper bound (fail-low)
    h2 = UInt64(0xABCDEF0123456789)
    Search.tt_store!(tt, h2, 4, 2.0, Search.TT_UPPER)
    # Upper bound of 2.0: if α ≥ 2.0, should return 2.0
    result3 = Search.tt_probe(tt, h2, 4, 3.0, 6.0)
    test("TT: upper bound with α=3 > 2 → returns 2.0",
         result3 !== nothing && abs(result3 - 2.0) < 0.01)
    # Upper bound of 2.0: if α < 2.0, should not cutoff
    result4 = Search.tt_probe(tt, h2, 4, 1.0, 6.0)
    test("TT: upper bound with α=1 < 2 → nothing", result4 === nothing)
end

# 4e. Search produces consistent results from same position
let
    b1 = from_fen("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
    b2 = from_fen("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")

    m1, s1 = best_move(b1; max_depth=3, verbose=false)
    m2, s2 = best_move(b2; max_depth=3, verbose=false)

    test("Deterministic search: same position → same move",
         move_to_string(m1) == move_to_string(m2))
    test("Deterministic search: same position → same score",
         abs(s1 - s2) < 0.001)
end


# ═══════════════════════════════════════════════════════════════
# 5. MOVE ORDERING SANITY — no illegal moves in generated list
# ═══════════════════════════════════════════════════════════════
println("\n═══ 5. Move Ordering & Legality ═══\n")

const LEGALITY_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    # Positions where king is in check
    "rnbqkbnr/ppp1pppp/8/1B1p4/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
    # Positions with pins
    "4k3/8/4r3/8/8/4B3/8/4K3 w - - 0 1",
    # Positions with en passant that exposes king (tricky!)
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
    # Promotion position
    "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
]

legality_all_ok = true
for fen in LEGALITY_FENS
    b = from_fen(fen)
    moves = generate_moves(b)

    for m in moves
        # Verify move is from our piece
        piece = piece_at(b, m.from_rank, m.from_file)
        if sign(piece) != b.turn
            println("  ✗ Illegal: move from wrong color at $fen: $(move_to_string(m))")
            global legality_all_ok = false
            continue
        end

        # Verify move doesn't leave king in check
        undo = apply_move!(b, m)
        if is_in_check(b, -b.turn)  # -b.turn is the side that just moved
            println("  ✗ Illegal: $(move_to_string(m)) leaves king in check at $fen")
            global legality_all_ok = false
        end
        undo_move!(b, m, undo)
    end

    # Verify no moves target own pieces (except castling which is king→rook-like)
    for m in moves
        m.is_castling && continue
        target = piece_at(b, m.to_rank, m.to_file)
        if target != 0.0 && sign(target) == b.turn
            println("  ✗ Illegal: $(move_to_string(m)) captures own piece at $fen")
            global legality_all_ok = false
        end
    end
end
test("All moves in $(length(LEGALITY_FENS)) positions are legal", legality_all_ok)

# 5b. Verify no king captures exist in move lists
king_capture_ok = true
for fen in LEGALITY_FENS
    b = from_fen(fen)
    moves = generate_moves(b)
    for m in moves
        m.is_castling && continue
        target = piece_at(b, m.to_rank, m.to_file)
        if abs(target) == KING
            println("  ✗ King capture possible: $(move_to_string(m)) at $fen")
            global king_capture_ok = false
        end
    end
end
test("No king captures in any move list", king_capture_ok)

# 5c. MVV-LVA scores are consistent
let
    b = from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    moves = generate_moves(b)

    captures = filter(m -> piece_at(b, m.to_rank, m.to_file) != 0.0 || m.is_en_passant, moves)
    quiet    = filter(m -> piece_at(b, m.to_rank, m.to_file) == 0.0 && !m.is_en_passant, moves)

    # All captures should score ≥ 0 in MVV-LVA
    cap_ok = all(m -> Search.mvv_lva_score(b, m) >= 0, captures)
    test("MVV-LVA: all captures score ≥ 0", cap_ok)

    # All quiet moves should score -1
    quiet_ok = all(m -> Search.mvv_lva_score(b, m) == -1, quiet)
    test("MVV-LVA: all quiet moves score -1", quiet_ok)
end


# ═══════════════════════════════════════════════════════════════
# 6. ZOBRIST HASH CORRECTNESS
# ═══════════════════════════════════════════════════════════════
println("\n═══ 6. Zobrist Hash Correctness ═══\n")

# 6a. Incremental hash matches full recompute after every move
let
    hash_ok = true
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ]

    for fen in test_fens
        b = from_fen(fen)
        moves = generate_moves(b)
        for m in moves
            undo = apply_move!(b, m)

            # Recompute hash from scratch and compare
            recomputed = State._recompute_hash(b)
            if b.hash != recomputed
                println("  ✗ Hash mismatch after $(move_to_string(m)) in $fen")
                println("    incremental: $(b.hash), recomputed: $recomputed")
                hash_ok = false
            end

            undo_move!(b, m, undo)

            # Verify undo restores hash
            recomputed2 = State._recompute_hash(b)
            if b.hash != recomputed2
                println("  ✗ Hash mismatch after undo $(move_to_string(m)) in $fen")
                hash_ok = false
            end
        end
    end
    test("Incremental Zobrist hash matches full recompute (all moves in 4 positions)", hash_ok)
end

# 6b. Hash changes when any component changes
let
    b1 = new_board()
    h1 = b1.hash

    b2 = new_board()
    b2.turn = BLACK
    sync_board!(b2)

    test("Zobrist: different turn → different hash", h1 != b2.hash)

    b3 = new_board()
    b3.castling[1] = false  # remove WK castling
    sync_board!(b3)
    test("Zobrist: different castling → different hash", h1 != b3.hash)

    b4 = new_board()
    b4.en_passant = (3, 5)  # fake en passant on e3
    sync_board!(b4)
    test("Zobrist: different en passant → different hash", h1 != b4.hash)
end

# 6c. Hash uniqueness: different positions should have different hashes
let
    hashes = Set{UInt64}()
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/R3K3 b - - 0 1",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    ]
    for fen in fens
        b = from_fen(fen)
        push!(hashes, b.hash)
    end
    test("Zobrist: $(length(fens)) different positions → $(length(hashes)) unique hashes",
         length(hashes) == length(fens))
end


# ═══════════════════════════════════════════════════════════════
# 7. SEARCH TERMINATION EDGE CASES
# ═══════════════════════════════════════════════════════════════
println("\n═══ 7. Search Termination Edge Cases ═══\n")

# 7a. Mate in 1 — engine must find it
let
    b = new_board()
    b.grid .= 0.0
    b.grid[6, 5] = KING     # Ke6
    b.grid[8, 5] = -KING    # Ke8
    b.grid[7, 4] = QUEEN    # Qd7 → Qd8#
    b.turn = WHITE
    b.castling = [false, false, false, false]
    sync_board!(b)

    m, s = best_move(b; max_depth=3, verbose=false)
    undo = apply_move!(b, m)
    test("Mate-in-1: engine delivers checkmate", is_checkmate(b))
    undo_move!(b, m, undo)
    test("Mate-in-1: score ≥ 9000", s >= 9000.0)
end

# 7b. Stalemate detection — should evaluate to 0
let
    b = new_board()
    b.grid .= 0.0
    b.grid[1, 1] = KING     # Ka1
    b.grid[8, 8] = -KING    # Kh8
    b.grid[7, 6] = QUEEN    # Qf7 → stalemate
    b.turn = BLACK
    b.castling = [false, false, false, false]
    sync_board!(b)

    test("Stalemate: detected correctly", is_stalemate(b))
    score = evaluate(b)
    test("Stalemate: evaluates to 0.0", score == 0.0)
end

# 7c. 50-move rule detection
let
    b = from_fen("4k3/8/8/8/8/8/8/4K2R w - - 99 50")
    # One more non-pawn, non-capture move → 50-move draw

    m = find_move(b, "h1h2")
    undo = apply_move!(b, m)
    test("50-move: halfmove clock at 100", b.halfmove == 100)
    test("50-move: game is over", is_game_over(b))
    undo_move!(b, m, undo)
end

# 7d. Checkmate score includes ply bonus (prefer faster mates)
let
    b = new_board()
    b.grid .= 0.0
    b.grid[6, 5] = KING
    b.grid[8, 5] = -KING
    b.grid[7, 4] = QUEEN
    b.turn = WHITE
    b.castling = [false, false, false, false]
    sync_board!(b)

    m, s = best_move(b; max_depth=4, verbose=false)
    test("Mate score is positive for White mating", s > 0)

    # The engine should prefer mating in 1 over mating in 3
    # Score for mate-in-1 should be higher than mate-in-3
    test("Mate-in-1 score > 9990 (ply bonus works)", s > 9990.0)
end

# 7e. Qsearch handles checks correctly (doesn't stand pat)
let
    b = from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1")
    test("Qsearch-check position: White in check", is_in_check(b, WHITE))

    field = compute_total_field(b)
    w = Search.DEFAULT_WEIGHTS
    q = Search.qsearch(b, w, -Search.INF, Search.INF, field, 1, new_tt())
    test("Qsearch while in check returns finite score", isfinite(q) && q > -9000.0)
end

# 7f. Qsearch detects mate at leaf
let
    b = from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    field = compute_total_field(b)
    w = Search.DEFAULT_WEIGHTS
    q = Search.qsearch(b, w, -Search.INF, Search.INF, field, 1, new_tt())
    test("Qsearch detects mate at leaf (score ≤ -9000)", q <= -9000.0)
end

# 7g. Engine handles forced single move
let
    # Position with only one legal move
    b = from_fen("4k3/8/8/8/8/7r/8/K7 w - - 0 1")
    moves = generate_moves(b)
    if length(moves) > 0
        m, s = best_move(b; max_depth=3, verbose=false)
        test("Single/few moves: engine returns a valid move", m ∈ generate_moves(b))
    else
        test("Position should have legal moves", false)
    end
end


# ═══════════════════════════════════════════════════════════════
# 8. ZERO-ALLOC BUFFER MANAGEMENT
# ═══════════════════════════════════════════════════════════════
println("\n═══ 8. Buffer Management ═══\n")

# 8a. Buffers exist for thread 1
let
    tid = 1
    test("Buffer: LEGAL_BUFS[1] has ≥ MAX_PLY entries",
         length(Search.LEGAL_BUFS[tid]) >= Search.MAX_PLY)
    test("Buffer: PSEUDO_BUFS[1] has ≥ MAX_PLY entries",
         length(Search.PSEUDO_BUFS[tid]) >= Search.MAX_PLY)
    test("Buffer: FIELD_STACK[1] has ≥ MAX_PLY entries",
         length(Search.FIELD_STACK[tid]) >= Search.MAX_PLY)
    test("Buffer: FIELD_BUFS[1] is 8×8",
         size(Search.FIELD_BUFS[tid]) == (8, 8))
    test("Buffer: FROM_SEEN[1] is 8×8",
         size(Search.FROM_SEEN[tid]) == (8, 8))
end

# 8b. ensure_ply_buffers! grows correctly
let
    tid = 1
    old_len = length(Search.LEGAL_BUFS[tid])
    Search.ensure_ply_buffers!(tid, old_len + 5)
    test("Buffer: ensure_ply_buffers! grew LEGAL_BUFS",
         length(Search.LEGAL_BUFS[tid]) >= old_len + 5)
    test("Buffer: ensure_ply_buffers! grew FIELD_STACK to match",
         length(Search.FIELD_STACK[tid]) == length(Search.LEGAL_BUFS[tid]))
end

# 8c. In-place move generation doesn't grow buffers
let
    b = new_board()
    legal_buf = Move[]
    pseudo_buf = Move[]
    sizehint!(legal_buf, 50)
    sizehint!(pseudo_buf, 50)

    generate_moves!(legal_buf, b, pseudo_buf)
    test("In-place movegen: legal_buf has 20 moves", length(legal_buf) == 20)

    # Apply a move and regenerate — buffer should be reused (no new allocation)
    undo = apply_move!(b, legal_buf[1])
    generate_moves!(legal_buf, b, pseudo_buf)
    test("In-place movegen: buffers work after move", length(legal_buf) > 0)
    undo_move!(b, legal_buf[1], undo)
end


# ═══════════════════════════════════════════════════════════════
# 9. ADDITIONAL EDGE CASES
# ═══════════════════════════════════════════════════════════════
println("\n═══ 9. Additional Edge Cases ═══\n")

# 9a. Promotion generates all 4 piece types
let
    b = from_fen("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    promos = filter(m -> m.promotion != 0.0, moves)
    promo_types = Set(abs(m.promotion) for m in promos)
    test("Promotion: generates 4 piece types", promo_types == Set([QUEEN, ROOK, BISHOP, KNIGHT]))
end

# 9b. Castling blocked by check
let
    # White rook on e5 checks Black king — cannot castle
    b = from_fen("r3k2r/pppp1ppp/8/4R3/8/8/PPPPPPPP/4K3 b kq - 0 1")
    test("Castling check: Black is in check", is_in_check(b, BLACK))
    moves = generate_moves(b)
    castling_moves = filter(m -> m.is_castling, moves)
    test("Castling: blocked when in check", isempty(castling_moves))
end

# 9c. Castling blocked by square under attack
let
    # Bishop on h3 attacks f1 — White kingside castling blocked
    b = from_fen("4k3/8/8/8/8/7b/8/R3K2R w KQ - 0 1")
    moves = generate_moves(b)
    ks_castling = filter(m -> m.is_castling && m.to_file == 7, moves)
    test("Castling: blocked when path under attack", isempty(ks_castling))
end

# 9d. En passant that exposes king (illegal)
let
    # En passant would expose king to rook
    b = from_fen("8/8/8/8/k2Pp2R/8/8/4K3 b - d3 0 1")
    moves = generate_moves(b)
    ep_moves = filter(m -> m.is_en_passant, moves)
    # The en passant capture would remove the d4 pawn, exposing king to rook on h4
    # Actually need to check: king on a4, pawns on d4 and e4, rook on h4
    # After exd3 e.p., e4 pawn disappears, but d4 pawn was on d4...
    # Let me construct this more carefully:
    # After e4xd3 e.p., d4 pawn is removed. King on a4 would be exposed to Rh4.
    # So this EP should be illegal.
    for m in ep_moves
        undo = apply_move!(b, m)
        leaves_check = is_in_check(b, BLACK)
        undo_move!(b, m, undo)
        if leaves_check
            test("EP: illegal en passant correctly filtered out", false)
        end
    end
    # If EP moves exist and none leave king in check, that's fine too.
    # The key test is: no EP move in the legal list leaves king in check.
    test("EP: no illegal en passant in move list",
         all(m -> begin
             undo = apply_move!(b, m)
             ok = !is_in_check(b, BLACK)
             undo_move!(b, m, undo)
             ok
         end, ep_moves))
end

# 9e. Material tracking is correct across complex sequences
let
    b = new_board()
    test("Material: starting position = 0", b.material == 0.0)

    # Play a series of moves and verify material after each
    apply_move!(b, find_move(b, "e2e4"))
    test("Material: after e4 still 0 (no capture)", b.material == 0.0)

    apply_move!(b, find_move(b, "d7d5"))
    apply_move!(b, find_move(b, "e4d5"))  # capture pawn
    test("Material: after exd5 = +1.0 (captured black pawn)", b.material == 1.0)
end

# 9f. King position tracking is correct
let
    b = new_board()
    test("King pos: White king at e1", b.white_king == (1, 5))
    test("King pos: Black king at e8", b.black_king == (8, 5))

    apply_move!(b, find_move(b, "e2e4"))
    apply_move!(b, find_move(b, "e7e5"))
    apply_move!(b, find_move(b, "e1e2"))  # King move

    test("King pos: White king at e2 after Ke2", b.white_king == (2, 5))
    test("King pos: Black king still at e8", b.black_king == (8, 5))
end

# 9g. Null move pruning hash restoration
let
    b = from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    original_hash = b.hash
    original_turn = b.turn
    original_ep = b.en_passant

    # Simulate null move (as done in negamax)
    b.turn = -b.turn
    b.hash ⊻= State.ZOBRIST_SIDE
    old_ep = b.en_passant
    if b.en_passant != (0,0)
        b.hash ⊻= State.ZOBRIST_EP[b.en_passant[2]]
        b.en_passant = (0,0)
    end

    # Undo null move
    b.turn = -b.turn
    b.hash ⊻= State.ZOBRIST_SIDE
    b.en_passant = old_ep
    old_ep != (0,0) && (b.hash ⊻= State.ZOBRIST_EP[old_ep[2]])

    test("Null move: hash restored after null-move undo", b.hash == original_hash)
    test("Null move: turn restored", b.turn == original_turn)
    test("Null move: en passant restored", b.en_passant == original_ep)
end


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

println("\n" * "═"^60)
println("  Results: $passed passed, $failed failed")
if failed == 0
    println("  All tests passed! ✓")
else
    println("  SOME TESTS FAILED ✗")
    println("  Failed tests:")
    for e in errors
        println("    • $e")
    end
end
println("═"^60 * "\n")
failed == 0 || exit(1)
