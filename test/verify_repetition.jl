#=
verify_repetition.jl — Issue #1 verification tests (post-fix)

Verifies the dual-API repetition detection:
  • is_repetition(b, threshold=2): search pruning (2-fold default)
  • is_threefold_repetition(b): FIDE-compliant game termination (3-fold)
  • is_game_over(b): uses 3-fold for legal draw detection

FIDE rule: A game is drawn when the same position occurs THREE times
(with the same side to move, castling rights, and en passant square).
Standard engine practice: use 2-fold in search (conservative pruning)
but 3-fold for actual game termination.

Run:  julia test/verify_repetition.jl
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search

# ── Test harness ──────────────────────────────────────────────
passed = 0
failed = 0
errors = String[]

function test(name, condition)
    global passed, failed, errors
    if condition
        passed += 1
        println("  ✓ PASS: $name")
    else
        failed += 1
        push!(errors, name)
        println("  ✗ FAIL: $name")
    end
end

function find_move(b::Board, uci::String)::Move
    for m in generate_moves(b)
        move_to_string(m) == uci && return m
    end
    error("Could not find legal move: $uci in position")
end

# Helper: count how many times the current hash appears in history
function count_repetitions(b::Board)::Int
    h = b.hash
    count = 0
    for i in 1:length(b.history)
        b.history[i] == h && (count += 1)
    end
    return count
end

# Helper: play a knight-dance cycle (Nf3 Nf6 Ng1 Ng8) returning to start
function play_knight_cycle!(b::Board)
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))
    apply_move!(b, find_move(b, "f6g8"))
end

println("═══════════════════════════════════════════════════════════════")
println(" Issue #1: Verify Repetition Detection (Post-Fix)")
println("═══════════════════════════════════════════════════════════════")

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 1: is_repetition(b) defaults to 2-fold (search pruning)
# ══════════════════════════════════════════════════════════════════
println("\n── 1. is_repetition(b) threshold (default=2, for search) ──\n")

let
    b = new_board()
    start_hash = b.hash

    test("Fresh board: no repetition", !is_repetition(b))
    test("Fresh board: 0 occurrences in history", count_repetitions(b) == 0)

    # First cycle: position seen for the 2nd time total
    play_knight_cycle!(b)
    reps_after_1_cycle = count_repetitions(b)
    test("After 1 cycle: hash in history exactly 1 time", reps_after_1_cycle == 1)
    test("After 1 cycle: current position = starting position", b.hash == start_hash)

    fires_on_2fold = is_repetition(b)
    test("is_repetition(b) fires on 2-fold (default threshold=2)", fires_on_2fold)

    # Explicit threshold=2 behaves the same as default
    test("is_repetition(b, 2) also fires on 2-fold", is_repetition(b, 2))

    # Second cycle: position seen for the 3rd time total
    play_knight_cycle!(b)
    reps_after_2_cycles = count_repetitions(b)
    test("After 2 cycles: hash in history exactly 2 times", reps_after_2_cycles == 2)
    test("After 2 cycles: is_repetition(b) still true", is_repetition(b))
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 2: is_threefold_repetition(b) requires 3-fold
# ══════════════════════════════════════════════════════════════════
println("\n── 2. is_threefold_repetition(b) threshold (FIDE 3-fold) ──\n")

let
    b = new_board()

    test("Fresh board: no threefold", !is_threefold_repetition(b))

    # After 1 cycle: 2-fold — NOT enough for threefold
    play_knight_cycle!(b)
    test("After 1 cycle (2-fold): is_threefold_repetition is FALSE",
         !is_threefold_repetition(b))
    test("After 1 cycle (2-fold): is_repetition(b, 2) is TRUE",
         is_repetition(b, 2))

    # After 2 cycles: 3-fold — NOW threefold fires
    play_knight_cycle!(b)
    test("After 2 cycles (3-fold): is_threefold_repetition is TRUE",
         is_threefold_repetition(b))
    test("After 2 cycles: is_repetition(b, 3) also TRUE",
         is_repetition(b, 3))
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 3: is_game_over() now uses 3-fold (not 2-fold)
# ══════════════════════════════════════════════════════════════════
println("\n── 3. is_game_over() uses 3-fold (FIDE-compliant) ──\n")

let
    # 3a. At 2-fold: game should NOT be over (just a search draw)
    b = new_board()
    play_knight_cycle!(b)

    test("At 2-fold: is_repetition(b) is true (search sees draw)",
         is_repetition(b))
    test("At 2-fold: is_game_over() is FALSE (game continues)",
         !is_game_over(b))
    test("At 2-fold: is_threefold_repetition is false",
         !is_threefold_repetition(b))

    # 3b. At 3-fold: game IS over
    play_knight_cycle!(b)
    test("At 3-fold: is_game_over() is TRUE", is_game_over(b))
    test("At 3-fold: game_result() returns 0 (draw)", game_result(b) == 0)
    test("At 3-fold: not checkmate", !is_checkmate(b))
    test("At 3-fold: not stalemate", !is_stalemate(b))
    test("At 3-fold: halfmove < 100", b.halfmove < 100)
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 4: Dual API exists
# ══════════════════════════════════════════════════════════════════
println("\n── 4. Verify dual API exists in State module ──\n")

let
    test("is_repetition is defined in State", isdefined(State, :is_repetition))
    test("is_threefold_repetition is defined in State",
         isdefined(State, :is_threefold_repetition))

    # is_repetition now accepts an optional threshold parameter
    rep_methods = methods(is_repetition)
    test("is_repetition has at least 1 method", length(rep_methods) >= 1)

    # is_threefold_repetition exists as a convenience wrapper
    tf_methods = methods(is_threefold_repetition)
    test("is_threefold_repetition has at least 1 method", length(tf_methods) >= 1)
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 5: Search (negamax) still uses 2-fold
# ══════════════════════════════════════════════════════════════════
println("\n── 5. Search (negamax) uses 2-fold correctly ──\n")

let
    b = new_board()
    play_knight_cycle!(b)
    test("Pre-condition: 2-fold repetition", is_repetition(b))
    test("Pre-condition: NOT threefold", !is_threefold_repetition(b))

    field = zeros(Float64, 8, 8)
    Fields.compute_total_field!(field, b)
    w = Float64[Energy.W_MATERIAL, Energy.W_FIELD, Energy.W_KING_SAFETY,
                Energy.W_TENSION, Energy.W_MOBILITY]

    tid = Threads.threadid()
    Search.ensure_ply_buffers!(tid, 2)
    tt = Search.new_tt()
    score = Search.negamax(b, w, 1, -10000.0, 10000.0, field, 1, tt)
    test("negamax returns 0.0 on 2-fold (search draw)", score == 0.0)
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 6: 2-fold vs 3-fold distinction in game context
# ══════════════════════════════════════════════════════════════════
println("\n── 6. 2-fold vs 3-fold: correct separation ──\n")

let
    b = new_board()

    test("Start: 0 prior occurrences", count_repetitions(b) == 0)

    # After 1 cycle: 2-fold
    play_knight_cycle!(b)
    test("After 1 cycle: 1 prior occurrence", count_repetitions(b) == 1)
    test("2-fold: is_repetition TRUE", is_repetition(b))
    test("2-fold: is_threefold FALSE", !is_threefold_repetition(b))
    test("2-fold: is_game_over FALSE", !is_game_over(b))

    # After 2 cycles: 3-fold
    play_knight_cycle!(b)
    test("After 2 cycles: 2 prior occurrences", count_repetitions(b) == 2)
    test("3-fold: is_repetition TRUE", is_repetition(b))
    test("3-fold: is_threefold TRUE", is_threefold_repetition(b))
    test("3-fold: is_game_over TRUE", is_game_over(b))
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 7: Edge cases — irreversible moves reset repetition
# ══════════════════════════════════════════════════════════════════
println("\n── 7. Irreversible move resets repetition window ──\n")

let
    b = new_board()

    # Do a knight cycle to create 2-fold
    play_knight_cycle!(b)
    test("2-fold after knight cycle", is_repetition(b))

    # Now play a pawn move (irreversible) — resets halfmove counter
    apply_move!(b, find_move(b, "e2e4"))
    test("Pawn push resets halfmove to 0", b.halfmove == 0)
    test("After pawn push: no repetition (different position)", !is_repetition(b))

    # The post-e4 position has an en passant square set, so a knight
    # cycle won't return to the SAME hash (ep square disappears after
    # the next move). Instead, we need to "burn off" the ep square first
    # and then repeat THAT position.
    apply_move!(b, find_move(b, "g8f6"))  # ep square gone after this
    apply_move!(b, find_move(b, "g1f3"))
    post_cycle_hash = b.hash

    # Now complete the cycle and do another one to get back here
    apply_move!(b, find_move(b, "f6g8"))
    apply_move!(b, find_move(b, "f3g1"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "g1f3"))
    test("After cycle post-e4: repetition detected", is_repetition(b))
    test("Hash matches mid-cycle position", b.hash == post_cycle_hash)
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 8: Rook shuffle — 2-fold does NOT end game, 3-fold does
# ══════════════════════════════════════════════════════════════════
println("\n── 8. Rook shuffle repetition (KR vs K endgame) ──\n")

let
    b = from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    start_hash = b.hash

    # First cycle: Ra1-a2, Ke8-d8, Ra2-a1, Kd8-e8 → 2-fold
    apply_move!(b, find_move(b, "a1a2"))
    apply_move!(b, find_move(b, "e8d8"))
    apply_move!(b, find_move(b, "a2a1"))
    apply_move!(b, find_move(b, "d8e8"))

    test("Rook shuffle returns to start hash", b.hash == start_hash)
    test("2-fold detected after rook shuffle", is_repetition(b))
    test("is_game_over does NOT fire at 2-fold (FIDE correct)", !is_game_over(b))

    # Second cycle → 3-fold
    apply_move!(b, find_move(b, "a1a2"))
    apply_move!(b, find_move(b, "e8d8"))
    apply_move!(b, find_move(b, "a2a1"))
    apply_move!(b, find_move(b, "d8e8"))

    test("3-fold after second rook shuffle", is_threefold_repetition(b))
    test("is_game_over fires at 3-fold", is_game_over(b))
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 9: Castling rights affect repetition correctly
# ══════════════════════════════════════════════════════════════════
println("\n── 9. Castling rights difference prevents false repetition ──\n")

let
    # Position with castling rights
    b1 = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    # Same position, no castling rights
    b2 = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1")

    test("Different castling rights produce different hashes", b1.hash != b2.hash)
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 10: Undo correctly restores repetition state
# ══════════════════════════════════════════════════════════════════
println("\n── 10. Undo correctly restores repetition state ──\n")

let
    b = new_board()

    # Set up near-repetition
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))

    # The completing move
    m = find_move(b, "f6g8")
    undo = apply_move!(b, m)
    test("After completing cycle: 2-fold repetition detected", is_repetition(b))
    test("After completing cycle (2-fold): game NOT over", !is_game_over(b))

    history_len_before = length(b.history)
    undo_move!(b, m, undo)
    test("After undo: no repetition", !is_repetition(b))
    test("After undo: game not over", !is_game_over(b))
    test("After undo: history length decremented", length(b.history) == history_len_before - 1)
end

# ══════════════════════════════════════════════════════════════════
# TEST GROUP 11: Threshold parameter edge cases
# ══════════════════════════════════════════════════════════════════
println("\n── 11. Threshold parameter edge cases ──\n")

let
    b = new_board()

    # threshold=1 would fire even with zero prior occurrences
    # (current position counts as 1st occurrence via the hash itself)
    # Actually no — is_repetition only checks history, not current.
    # So threshold=1 fires on ANY single match in history.
    play_knight_cycle!(b)  # 2-fold
    test("is_repetition(b, 1) true at 2-fold", is_repetition(b, 1))
    test("is_repetition(b, 2) true at 2-fold", is_repetition(b, 2))
    test("is_repetition(b, 3) false at 2-fold", !is_repetition(b, 3))

    play_knight_cycle!(b)  # 3-fold
    test("is_repetition(b, 1) true at 3-fold", is_repetition(b, 1))
    test("is_repetition(b, 2) true at 3-fold", is_repetition(b, 2))
    test("is_repetition(b, 3) true at 3-fold", is_repetition(b, 3))
    test("is_repetition(b, 4) false at 3-fold", !is_repetition(b, 4))
end

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
println("\n═══════════════════════════════════════════════════════════════")
println(" RESULTS: $passed passed, $failed failed")
println("═══════════════════════════════════════════════════════════════")

if !isempty(errors)
    println("\n  Failed tests:")
    for e in errors
        println("    ✗ $e")
    end
end

println("\n═══════════════════════════════════════════════════════════════")
println(" VERIFICATION SUMMARY (Post-Fix)")
println("═══════════════════════════════════════════════════════════════")
println()
println("  Dual API correctly implemented:")
println("    ✓ is_repetition(b)      → 2-fold (search pruning)")
println("    ✓ is_repetition(b, N)   → N-fold (configurable threshold)")
println("    ✓ is_threefold_repetition(b) → 3-fold (FIDE game termination)")
println()
println("  Call site separation verified:")
println("    ✓ negamax uses is_repetition(b) → 2-fold (correct for search)")
println("    ✓ is_game_over uses is_threefold_repetition → 3-fold (FIDE)")
println("    ✓ Game continues at 2-fold, only ends at 3-fold")
println("    ✓ KR vs K no longer falsely drawn after one shuffle cycle")
println("═══════════════════════════════════════════════════════════════")

exit(failed > 0 ? 1 : 0)
