#=
verify_repetition.jl — Verify correct two-fold (search) vs three-fold (game) repetition behavior.

This test suite validates the fix for Issue #1:
  https://github.com/MichaelCBrister/Field-Engine/issues/1

Tests cover:
  1.  is_repetition(b, threshold=2) — default 2-fold for search pruning
  2.  is_threefold_repetition(b)    — 3-fold for FIDE game termination
  3.  is_game_over uses 3-fold (not 2-fold)
  4.  threshold semantics: n-th total occurrence (including current)
  5.  Edge cases: captures reset window, undo/redo, halfmove window

Expected: 40/40 tests pass.
=#

# ── Bootstrap: load the engine from the project root ──────────────────
let project_root = joinpath(@__DIR__, "..")
    pushfirst!(LOAD_PATH, project_root)
    include(joinpath(project_root, "src", "FieldEngine.jl"))
end

using .FieldEngine.State

# ── Test harness ───────────────────────────────────────────────────────
passed = 0
failed = 0

function test(name::String, result::Bool)
    global passed, failed
    if result
        passed += 1
        println("  ✓ $name")
    else
        failed += 1
        println("  ✗ FAIL: $name")
    end
end

function find_move(b::Board, uci::String)::Move
    for m in generate_moves(b)
        move_to_string(m) == uci && return m
    end
    error("Move not found: $uci")
end

println("\n═══ verify_repetition.jl ═══\n")

# ── Knight-dance helper: plays 1. Nf3 Nf6 2. Ng1 Ng8 (one full cycle) ──
function knight_dance!(b::Board)
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]
        apply_move!(b, find_move(b, uci))
    end
end

# ════════════════════════════════════════════════════════════════════════
# Section 1: is_repetition with default threshold=2 (search pruning)
# ════════════════════════════════════════════════════════════════════════
println("─── Section 1: is_repetition (2-fold default) ───\n")

# Test 1: Fresh board — no repetition
let b = new_board()
    test("1. Fresh board: is_repetition(b) == false", !is_repetition(b))
end

# Test 2: After 1. Nf3 — no repetition yet
let b = new_board()
    apply_move!(b, find_move(b, "g1f3"))
    test("2. After 1.Nf3: is_repetition(b) == false", !is_repetition(b))
end

# Test 3: After 1. Nf3 Nf6 — no repetition yet
let b = new_board()
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    test("3. After 1.Nf3 Nf6: is_repetition(b) == false", !is_repetition(b))
end

# Test 4: After 1. Nf3 Nf6 2. Ng1 — no repetition yet (position is new)
let b = new_board()
    apply_move!(b, find_move(b, "g1f3"))
    apply_move!(b, find_move(b, "g8f6"))
    apply_move!(b, find_move(b, "f3g1"))
    test("4. After 2.Ng1: is_repetition(b) == false", !is_repetition(b))
end

# Test 5: After 1. Nf3 Nf6 2. Ng1 Ng8 — 2-fold: is_repetition returns true
let b = new_board()
    knight_dance!(b)
    test("5. After 2-fold: is_repetition(b) == true", is_repetition(b))
end

# Test 6: is_repetition with explicit threshold=2 matches default
let b = new_board()
    knight_dance!(b)
    test("6. is_repetition(b,2) == is_repetition(b)", is_repetition(b, 2) == is_repetition(b))
end

# Test 7: is_repetition(b, 1) — trivially true once any prior position equals current
let b = new_board()
    knight_dance!(b)
    test("7. is_repetition(b,1) == true after any repetition", is_repetition(b, 1))
end

# Test 8: is_repetition(b, 2) == true after first repetition (2 total occurrences)
let b = new_board()
    knight_dance!(b)
    test("8. is_repetition(b,2) == true after 2 total occurrences", is_repetition(b, 2))
end

# Test 9: is_repetition(b, 3) == false after only 2 occurrences
let b = new_board()
    knight_dance!(b)
    test("9. is_repetition(b,3) == false after only 2 occurrences", !is_repetition(b, 3))
end

# Test 10: Pawn push resets halfmove window — prior positions outside window
let b = new_board()
    apply_move!(b, find_move(b, "e2e4"))   # pawn push → halfmove resets to 0
    test("10. After pawn push: halfmove == 0, no repetition", b.halfmove == 0 && !is_repetition(b))
end

# ════════════════════════════════════════════════════════════════════════
# Section 2: is_threefold_repetition (FIDE game termination)
# ════════════════════════════════════════════════════════════════════════
println("\n─── Section 2: is_threefold_repetition (3-fold FIDE) ───\n")

# Test 11: Fresh board — no threefold
let b = new_board()
    test("11. Fresh board: is_threefold_repetition(b) == false", !is_threefold_repetition(b))
end

# Test 12: After 2-fold only — threefold NOT triggered
let b = new_board()
    knight_dance!(b)
    test("12. After 2-fold: is_threefold_repetition == false", !is_threefold_repetition(b))
end

# Test 13: After 3-fold — threefold IS triggered
let b = new_board()
    knight_dance!(b)   # 2nd occurrence
    knight_dance!(b)   # 3rd occurrence
    test("13. After 3-fold: is_threefold_repetition == true", is_threefold_repetition(b))
end

# Test 14: is_threefold_repetition delegates to is_repetition(b, 3)
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    test("14. is_threefold_repetition(b) == is_repetition(b,3)", is_threefold_repetition(b) == is_repetition(b, 3))
end

# Test 15: After only first dance (2-fold) is_repetition(b,3)==false
let b = new_board()
    knight_dance!(b)
    test("15. is_repetition(b,3) == false after 2-fold", !is_repetition(b, 3))
end

# Test 16: History length after 2 full knight dances = 8
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    test("16. History length == 8 after two knight dances", length(b.history) == 8)
end

# Test 17: Current hash matches starting position hash after 3-fold
let b = new_board()
    start_hash = b.hash
    knight_dance!(b)
    knight_dance!(b)
    test("17. After two dances, current hash matches start hash", b.hash == start_hash)
end

# Test 18: Pawn push between two dances breaks repetition window
let b = new_board()
    knight_dance!(b)                        # 2nd occurrence of start pos
    apply_move!(b, find_move(b, "e2e4"))    # pawn push: halfmove resets to 0
    test("18. After pawn push, repetition window reset: no threefold", !is_threefold_repetition(b))
end

# Test 19: After 4-fold: is_threefold_repetition still true
let b = new_board()
    knight_dance!(b)   # 2nd occurrence
    knight_dance!(b)   # 3rd occurrence
    knight_dance!(b)   # 4th occurrence
    test("19. After 4-fold: is_threefold_repetition still true", is_threefold_repetition(b))
end

# Test 20: is_repetition(b,3) equals is_threefold_repetition(b)
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    test("20. is_repetition(b,3) == is_threefold_repetition(b)", is_repetition(b, 3) == is_threefold_repetition(b))
end

# ════════════════════════════════════════════════════════════════════════
# Section 3: is_game_over uses 3-fold (FIDE compliance)
# ════════════════════════════════════════════════════════════════════════
println("\n─── Section 3: is_game_over uses 3-fold ───\n")

# Test 21: Fresh board — game not over
let b = new_board()
    test("21. Fresh board: is_game_over == false", !is_game_over(b))
end

# Test 22: After 2-fold: game NOT over (old bug was true here — the key FIDE fix)
let b = new_board()
    knight_dance!(b)
    test("22. After 2-fold only: is_game_over == false (FIDE fix)", !is_game_over(b))
end

# Test 23: After 3-fold: game IS over
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    test("23. After 3-fold: is_game_over == true", is_game_over(b))
end

# Test 24: is_game_over and is_threefold_repetition agree at 2-fold boundary
let b = new_board()
    knight_dance!(b)
    test("24. At 2-fold: is_game_over==false and is_threefold_repetition==false", !is_game_over(b) && !is_threefold_repetition(b))
end

# Test 25: is_game_over and is_threefold_repetition agree at 3-fold boundary
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    test("25. At 3-fold: is_game_over==true and is_threefold_repetition==true", is_game_over(b) && is_threefold_repetition(b))
end

# ════════════════════════════════════════════════════════════════════════
# Section 4: Threshold separation — search vs game contexts
# ════════════════════════════════════════════════════════════════════════
println("\n─── Section 4: Threshold separation (search vs game) ───\n")

# Test 26: After 2-fold: search prunes (is_repetition true), game continues (threefold false)
let b = new_board()
    knight_dance!(b)
    search_prunes = is_repetition(b)           # 2-fold → true (prune in search)
    game_ends     = is_threefold_repetition(b) # 3-fold → false (game continues)
    test("26. 2-fold: search prunes=true, game-over=false", search_prunes && !game_ends)
end

# Test 27: After 3-fold: both search prunes AND game ends
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    search_prunes = is_repetition(b)
    game_ends     = is_threefold_repetition(b)
    test("27. 3-fold: search prunes=true AND game-over=true", search_prunes && game_ends)
end

# Test 28: is_repetition(b,2) and is_repetition(b,3) differ at 2-fold
let b = new_board()
    knight_dance!(b)
    test("28. 2-fold: threshold=2 true, threshold=3 false", is_repetition(b, 2) && !is_repetition(b, 3))
end

# Test 29: is_repetition(b,2) and is_repetition(b,3) both true at 3-fold
let b = new_board()
    knight_dance!(b)
    knight_dance!(b)
    test("29. 3-fold: both threshold=2 and threshold=3 true", is_repetition(b, 2) && is_repetition(b, 3))
end

# Test 30: halfmove == 4 after one knight dance (no captures/pawn moves)
let b = new_board()
    knight_dance!(b)
    test("30. halfmove == 4 after one knight dance", b.halfmove == 4)
end

# ════════════════════════════════════════════════════════════════════════
# Section 5: Undo/redo consistency
# ════════════════════════════════════════════════════════════════════════
println("\n─── Section 5: Undo/redo consistency ───\n")

# Tests 31-32: Undo after 2-fold restores non-repetition state
let b = new_board()
    for uci in ["g1f3", "g8f6", "f3g1"]
        apply_move!(b, find_move(b, uci))
    end
    m    = find_move(b, "f6g8")
    undo = apply_move!(b, m)                   # 2-fold reached
    test("31. 2-fold reached before undo", is_repetition(b))
    undo_move!(b, m, undo)
    test("32. After undo: is_repetition(b) == false", !is_repetition(b))
end

# Test 33: History length correct after apply and undo
let b = new_board()
    m    = find_move(b, "g1f3")
    undo = apply_move!(b, m)
    len_after = length(b.history)
    undo_move!(b, m, undo)
    len_restored = length(b.history)
    test("33. apply_move! pushes history; undo_move! pops it", len_after == 1 && len_restored == 0)
end

# Tests 34-35: Undo last move of 3-fold sequence restores 2-fold state
let b = new_board()
    knight_dance!(b)   # 2nd occurrence
    for uci in ["g1f3", "g8f6", "f3g1"]
        apply_move!(b, find_move(b, uci))
    end
    m_last    = find_move(b, "f6g8")
    undo_last = apply_move!(b, m_last)         # 3rd occurrence
    test("34. Before undo: is_threefold_repetition == true", is_threefold_repetition(b))
    undo_move!(b, m_last, undo_last)
    test("35. After undo last: is_threefold_repetition == false", !is_threefold_repetition(b))
end

# ════════════════════════════════════════════════════════════════════════
# Section 6: Edge cases and additional coverage
# ════════════════════════════════════════════════════════════════════════
println("\n─── Section 6: Edge cases ───\n")

# Test 36: Empty history — no repetition for default threshold
let b = new_board()
    test("36. Empty history: is_repetition(b,2) == false", !is_repetition(b, 2))
end

# Test 37: Threefold requires >= 3 total, not just 2 — boundary one move before
let b = new_board()
    knight_dance!(b)   # 2nd occurrence
    for uci in ["g1f3", "g8f6", "f3g1"]
        apply_move!(b, find_move(b, uci))
    end
    # 3 moves into 3rd cycle — position NOT yet repeated a 3rd time
    test("37. 3 moves into 3rd cycle: is_threefold_repetition == false", !is_threefold_repetition(b))
end

# Tests 38-39: is_repetition with threshold=4 requires 4 occurrences
let b = new_board()
    knight_dance!(b)   # 2nd occurrence
    knight_dance!(b)   # 3rd occurrence
    test("38. After 3-fold: is_repetition(b,4) == false", !is_repetition(b, 4))
    knight_dance!(b)   # 4th occurrence
    test("39. After 4-fold: is_repetition(b,4) == true", is_repetition(b, 4))
end

# Test 40: Zobrist hash after two knight dances equals starting hash
let b = new_board()
    start_hash = b.hash
    knight_dance!(b)
    knight_dance!(b)
    test("40. Hash after 2 dances equals starting hash", b.hash == start_hash)
end

# ════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════
println("\n═══════════════════════════════════════════════════════")
println("  Results: $passed passed, $failed failed out of $(passed + failed)")
if failed == 0
    println("  All tests passed! ✓")
    println("  Two-fold/three-fold repetition API separation verified.")
else
    println("  SOME TESTS FAILED ✗")
end
println("═══════════════════════════════════════════════════════\n")
failed == 0 || exit(1)
