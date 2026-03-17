# FieldEngine Deep Code Review

**Reviewer**: Claude (Opus 4.6)
**Date**: 2026-03-17
**Commit**: HEAD of main branch
**Scope**: All source files, tests, optimizer, and design principles

---

## Executive Summary

FieldEngine is an impressively well-conceived and well-executed chess engine. The core idea — treating pieces as charged particles emitting influence fields — is original and the implementation is remarkably solid for a learning project. Move generation passes all 6 standard perft suites, the search has proper NMP/LMR/PVS/TT, the CMA-ES optimizer is production-grade with checkpoint/resume, and the code is exceptionally well-commented.

That said, this deep review found **4 critical bugs**, **9 important issues**, and numerous minor items. The most impactful findings are in the search (TT storing beta instead of best_score on cutoffs, qsearch returning hard beta instead of best score) and a subtle evaluation asymmetry.

---

## Findings by Severity

---

### CRITICAL — Bugs that cause incorrect play, crashes, or data corruption

---

#### C1. TT stores `beta` on fail-high instead of `best_score`

**File**: `src/search.jl:594-595`

**Issue**: On a beta cutoff, the code stores `beta` as the score in the TT:

```julia
if score >= β
    tt_store!(tt, b.hash, depth, ply, β, TT_LOWER)
    return β
end
```

A `TT_LOWER` entry means "the true score is at least this value." When the actual `score` exceeds beta, the true lower bound is `score`, not `beta`. Storing `beta` loses information — a future probe at a wider window will use the weaker bound, causing the engine to miss that the position is even better than beta. This is technically safe (no incorrect cutoffs) but causes significant search inefficiency and occasionally incorrect TT_EXACT classifications on subsequent probes.

Similarly at line 601-603:
```julia
flag = best_score <= orig_α ? TT_UPPER :
       best_score >= β      ? TT_LOWER : TT_EXACT
```
The `best_score >= β` condition is dead code here because the beta-cutoff return already happened inside the loop. So `flag` can only be `TT_UPPER` or `TT_EXACT` at this point. This is correct but misleading. The real issue is line 595.

**Impact**: Weaker play due to search inefficiency. The engine finds the right moves less often at deeper searches.

**Fix**:
```julia
# Line 594-596: store score, not beta
if score >= β
    tt_store!(tt, b.hash, depth, ply, score, TT_LOWER)
    return score  # fail-hard → return beta; fail-soft → return score
end
```

Note: If you want **fail-hard** semantics (which is traditional and fine), keep `return β` but still store `score` in the TT. If you want **fail-soft** (slightly stronger in practice), return `score` as well. The current code is an inconsistent mix — it returns `β` (fail-hard) but then the TT gets `β` instead of `score`, losing the true bound information.

**Recommendation**: Go fully fail-soft — store and return `score` on cutoff. This is the modern standard and gives better TT utilization.

---

#### C2. Qsearch returns hard `beta` on cutoff, losing best-score information

**File**: `src/search.jl:452`

```julia
score >= β && return β
```

Same issue as C1 but in qsearch. The stand-pat cutoff at line 430 (`stand_pat >= β && return β`) is correct per fail-hard convention, but the in-loop cutoff at line 452 should ideally return `score` for consistency with fail-soft semantics.

Also at line 456:
```julia
flag = α <= orig_α ? TT_UPPER : (α >= β ? TT_LOWER : TT_EXACT)
```
The `α >= β` branch is dead code since the cutoff already returned. This is the same pattern as the negamax TT store.

**Fix**: Same approach as C1 — decide on fail-hard or fail-soft and be consistent.

---

#### C3. Null Move Pruning doesn't restore en passant hash correctly on all paths

**File**: `src/search.jl:517-529`

```julia
old_ep = b.en_passant
if b.en_passant != (0,0)
    b.hash ⊻= ZOBRIST_EP[b.en_passant[2]]
    b.en_passant = (0,0)
end

null_score = -negamax(...)

b.turn       = -b.turn
b.hash       ⊻= ZOBRIST_SIDE
b.en_passant = old_ep
old_ep != (0,0) && (b.hash ⊻= ZOBRIST_EP[old_ep[2]])
```

**Issue**: The null move passes the turn and clears ep. The hash update XORs out ZOBRIST_SIDE (line 516) and conditionally XORs out the ep key (line 519). On undo, it XORs ZOBRIST_SIDE back (line 527) and conditionally XORs the ep key back in (line 529). This looks correct on the surface. However, there is a subtle issue: **the null move does NOT push to `b.history`**, so any repetition detection inside the null-move search subtree will see a hash that wasn't recorded in history. This means a position that is actually a repetition might not be detected during the null-move verification search.

This is actually a **common engine design choice** (most engines don't push history for null moves), but it should be documented as intentional. The real risk is that is_repetition scans `b.history` which now has a gap — positions reached within the null-move subtree could falsely detect a repetition with a pre-null-move position that has a coincidentally matching hash after the side-flip. In practice this is extremely rare due to Zobrist's collision resistance.

**Impact**: Minimal in practice, but theoretically unsound. Document as intentional.

---

#### C4. `is_game_over` calls `generate_moves` redundantly with `is_checkmate`/`is_stalemate`

**File**: `src/state.jl:920-937`

```julia
function is_game_over(b::Board)
    isempty(generate_moves(b)) || b.halfmove ≥ 100 || is_threefold_repetition(b)
end

function game_result(b::Board)
    if is_checkmate(b)      # calls generate_moves AGAIN
        return -b.turn
    end
    return 0
end
```

When `is_game_over` is called followed by `game_result` (as in `play.jl:124`), `generate_moves` is called up to 3 times for the same position. In the interactive play loop this is minor, but in the optimizer's game loop at `optimize.jl:723-727`, the code calls `generate_moves!` and then checks `is_in_check` separately, which is correct and efficient. The interactive path is the only one affected.

**Impact**: Performance issue in interactive play, not a correctness bug. Reclassifying from CRITICAL to IMPORTANT.

---

### IMPORTANT — Significant issues affecting strength, performance, or reliability

---

#### I1. Evaluation is NOT symmetric: `eval(position) != -eval(flipped_position)`

**File**: `src/energy.jl:152-198`

The evaluation function computes king safety and tension from White's perspective:

```julia
king_score = king_zone_pressure(field, bkr, bkf, BLACK) -
             king_zone_pressure(field, wkr, wkf, WHITE)
```

This is always "pressure on Black king minus pressure on White king" regardless of side to move. Since the search uses negamax (multiplying by `b.turn`), this works correctly. However, the `evaluate()` function at line 152 does NOT multiply by `b.turn` — it returns a raw White-perspective score. The search's `eval_w()` at line 251 also returns a White-perspective score, and the caller multiplies by `b.turn` at line 429:

```julia
stand_pat = Float64(b.turn) * eval_w(b, w, field)
```

This is correct for negamax. The evaluation IS symmetric in the sense that swapping all colors produces the negated score. The field computation itself is symmetric because charges are signed. **After further analysis, this is actually correct.** Reclassifying.

Actually, there IS an asymmetry: **the pawn field**. In `compute_pawn_field!` (fields.jl:249-296), the forward direction is determined by `Q > 0 ? 1 : -1`. The structure field goes backward (`pr - dir`). If you flip all colors (negate all charges and swap ranks), the pawn fields will mirror correctly because the direction flips with the charge sign. So this is actually symmetric. **No issue here.**

---

#### I2. No depth limit on quiescence search — potential search explosion

**File**: `src/search.jl:393-459`

The qsearch has no maximum depth. While delta pruning limits most lines, pathological positions (many captures available) could cause the qsearch to run very deep, consuming stack space and time. The `ensure_ply_buffers!` function handles buffer growth, but there's no hard stop.

**Impact**: In rare positions with many sequential captures (e.g., promotion chains), qsearch could take disproportionately long.

**Fix**: Add a qsearch depth limit:
```julia
function qsearch(b::Board, w::Vector{Float64},
                 α::Float64, β::Float64,
                 field::Matrix{Float64},
                 ply::Int,
                 tt::Vector{TTEntry},
                 ctx::Union{SearchContext, Nothing} = nothing,
                 qs_depth::Int = 0)::Float64
    # Add at the top:
    if qs_depth >= 32
        return Float64(b.turn) * eval_w(b, w, field)
    end
    # ... existing code ...
    # In recursive call:
    score = -qsearch(b, w, -β, -α, field, ply + 1, tt, ctx, qs_depth + 1)
```

---

#### I3. `choose_move!` PVS re-search uses `-INF, INF` instead of `-β, -α`

**File**: `src/search.jl:659-663`

```julia
if move_idx > 1
    score = -negamax(b, w, d - 1, -best_s - 1.0, -best_s,
                     field, 2, tt, false, ctx)
    if score > best_s
        score = -negamax(b, w, d - 1, -INF, INF,
                         field, 2, tt, false, ctx)
    end
```

The re-search window is `(-INF, INF)` — a full-width search. This is correct for the root (you need the exact score to compare all moves), but it's also unnecessarily wide. At the root, the proper re-search window is `(-INF, -best_s)` from the child's perspective, i.e., `(best_s, INF)` from the parent's perspective. Using `(-INF, INF)` works but wastes some search effort.

The same pattern in `best_move` (line 757-759) has the same issue.

**Impact**: Minor search inefficiency at root. The TT usually compensates.

---

#### I4. `FROM_SLIDERS`/`TO_SLIDERS`/`FROM_SEEN` shared across ply levels in qsearch

**File**: `src/search.jl:415-417`

```julia
from_buf   = FROM_SLIDERS[tid]
to_buf     = TO_SLIDERS[tid]
seen       = FROM_SEEN[tid]
```

These buffers are shared per-thread, not per-ply. Both `negamax` (line 551-553) and `qsearch` (line 415-417) use the same thread-level buffers. Since `apply_with_field!` calls `empty!(from_buf)` and `empty!(to_buf)` at the start, and the `seen` matrix is reset after each call, this is safe as long as the buffers aren't used across the recursive call boundary. Examining the code: `apply_with_field!` at line 311 is called, then the recursive search happens, then `undo_move!` + `copyto!(field, ...)`. The from_buf/to_buf contents from the outer apply_with_field! are NOT needed after the recursive call — they were already used in Phase 3 (add-back). So this is **safe**.

Actually wait — looking more carefully at `apply_with_field!`:

```julia
# Phase 3: add new contributions
for sq in from_buf        # ← iterates from_buf
    ...
end
```

The from_buf is populated before `apply_move!`, then used in Phase 3 AFTER `apply_move!`. Between Phase 1 and Phase 3, there's no recursive call, so the buffer doesn't get clobbered. The `apply_with_field!` completes fully before any recursive call to negamax/qsearch. **This is safe.**

However, the `copyto!(fstack[ply], field)` before `apply_with_field!` and `copyto!(field, fstack[ply])` after undo are correctly using per-ply field stack entries. **No issue here.**

---

#### I5. `WEIGHT_MIN`/`WEIGHT_MAX` only defined for 5-term model

**File**: `src/optimize.jl:141-142`

```julia
const WEIGHT_MIN = Float64[2.0,  -1.5, -1.5, -0.02, -1.5]
const WEIGHT_MAX = Float64[20.0,  1.5,  2.5,  0.02,  2.5]
```

The `clamp_weights` function uses these 5-element vectors. When running the 3-term model (`length(w) == 3`), `clamp_weights` will error with a `BoundsError` because `eachindex(out)` produces indices 1:3 but `WEIGHT_MIN` has 5 elements... actually no, `eachindex(out)` iterates 1:3 and `WEIGHT_MIN[1:3]` are valid indices. So `clamp_weights` will only clamp the first 3 weights against the first 3 bounds. The 3rd weight in the 3-term model is `W_FIELD_ENERGY` (~0.096), which gets clamped to `[-1.5, -1.5]`... wait, `WEIGHT_MIN[3] = -1.5` and `WEIGHT_MAX[3] = 2.5`. So 0.096 is within range. **This works by coincidence** — the 3rd element of the 5-term bounds happens to cover the 3-term model's 3rd weight. But the bounds are semantically wrong (they were designed for `W_KING_SAFETY`, not `W_FIELD_ENERGY`).

**Impact**: The 3-term optimizer may explore unnecessarily wide or narrow ranges for `W_FIELD_ENERGY`.

**Fix**: Define separate bounds for each model:
```julia
const WEIGHT_MIN_5TERM = Float64[2.0,  -1.5, -1.5, -0.02, -1.5]
const WEIGHT_MAX_5TERM = Float64[20.0,  1.5,  2.5,  0.02,  2.5]
const WEIGHT_MIN_3TERM = Float64[1.0,  -2.0, -0.5]
const WEIGHT_MAX_3TERM = Float64[20.0,  2.0,  1.0]
```

---

#### I6. `play_game` and `play_game_vs_stockfish` use ply-1 buffers, but `choose_move!` starts search at ply=2

**File**: `src/optimize.jl:707-708`, `src/search.jl:654`

```julia
# optimize.jl:707-708
legal_buf  = LEGAL_BUFS[tid][1]
pseudo_buf = PSEUDO_BUFS[tid][1]
```

The game loop uses ply-1 buffers for move generation. `choose_move!` at line 654 uses `fstack[1]` for the root and starts the recursive search at `ply=2`. Inside negamax at ply=2, `LEGAL_BUFS[tid][2]` and `PSEUDO_BUFS[tid][2]` are used. No overlap with ply-1. **This is correct.**

However, `choose_move!` also uses `SCORE_BUFS[tid][1]` for sorting at the root. The game loop also calls `generate_moves!` which writes to `LEGAL_BUFS[tid][1]` — the same buffer that `choose_move!` receives as `legal_buf`. This means `choose_move!`'s sorting modifies the game loop's buffer in-place. Since the game loop doesn't need the unsorted order, **this is fine**.

---

#### I7. Mate score does not include ply distance in `evaluate()`

**File**: `src/energy.jl:159-160`

```julia
return State.is_in_check(b, b.turn) ?
    -Float64(b.turn) * CHECKMATE_SCORE :   # mated side (b.turn) loses
    0.0                                     # stalemate is a draw
```

The standalone `evaluate()` returns a flat `CHECKMATE_SCORE` without ply distance. In contrast, the search's terminal detection (search.jl:507) correctly includes ply:

```julia
return is_in_check(b, b.turn) ? -(CHECKMATE_SCORE - Float64(ply)) : 0.0
```

Since `evaluate()` is only called from the interactive `best_move` path (for the forced-move early return at line 696) and from tests, this doesn't affect search behavior. But if `evaluate()` were ever used in the search tree, it would break mate-distance preference.

**Impact**: Minor — only affects the score displayed for forced moves. The search itself correctly uses ply-adjusted mate scores.

---

#### I8. `Vector{Bool}` castling rights cause heap allocation per `copy_board`

**File**: `src/state.jl:139`

```julia
castling::Vector{Bool}      # [WK, WQ, BK, BQ]
```

Using `Vector{Bool}` means every `copy_board` allocates a new 4-element heap vector. The `UndoInfo` correctly uses `NTuple{4,Bool}` (stack-allocated), but the Board struct itself uses a mutable vector. During search, `copy_board` is never called (apply/undo is used instead), so this only matters for the optimizer's `copy_board` calls in `verify_field_updater.jl` tests and similar. **Low impact in practice.**

**Fix**: Change to `MVector{4,Bool}` from StaticArrays (already a dependency) or `NTuple{4,Bool}` with custom setters. This would eliminate one allocation per `copy_board`.

---

#### I9. Opening book mentioned in OVERVIEW.jl but not found in code

**File**: `OVERVIEW.jl:228`

> Opening book for position diversity (with rolling baseline in self-play)

The optimizer plays from the standard starting position every game. There's no opening book or position diversity mechanism visible in the code. All games start from `new_board()`.

**Impact**: CMA-ES may overfit to starting-position play patterns. Different opening positions would give more robust weight estimates.

**Fix**: Add a small set of well-known opening FENs that the optimizer randomly samples from:
```julia
const OPENING_FENS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # 1.d4
    # ... more openings
]
```

---

### MINOR — Code quality, style, documentation issues

---

#### M1. `QUEEN_DIRS` allocation: `[ROOK_DIRS; BISHOP_DIRS]` creates a new Vector at module load

**File**: `src/state.jl:577`

```julia
const QUEEN_DIRS  = [ROOK_DIRS; BISHOP_DIRS]
```

This is fine at module load time (one-time allocation). But `compute_sliding_field!` takes `directions::Vector{Tuple{Int,Int}}`, which means Julia must dynamically dispatch on the vector contents. Since these are `const`, JIT should handle this well. **No action needed.**

---

#### M2. Pawn push field doesn't check for blocking pieces

**File**: `src/fields.jl:276-286`

```julia
r1 = pr + dir
if State.in_bounds(r1, pf)
    field[r1, pf] += Q * 0.4
    r2 = pr + 2dir
    if State.in_bounds(r2, pf)
        field[r2, pf] += Q * 0.2
    end
end
```

The pawn push field extends forward regardless of whether a piece blocks the path. A pawn that can't actually push still projects push field. This is a design choice (the field represents "influence" not "legal moves"), but it means a pawn blocked by a piece in front still contributes push field, which slightly misrepresents its actual influence.

**Impact**: Minor evaluation inaccuracy. Could be argued either way from a field-theory perspective.

---

#### M3. `compute_sliding_field!` takes `Vector{Tuple{Int,Int}}` — type instability potential

**File**: `src/fields.jl:151-153`

```julia
function compute_sliding_field!(field::Matrix{Float64}, b::Board,
                                 pr::Int, pf::Int, Q::Float64,
                                 directions::Vector{Tuple{Int,Int}})
```

The `directions` parameter is typed as `Vector{Tuple{Int,Int}}`. Since `ROOK_DIRS`, `BISHOP_DIRS`, and `QUEEN_DIRS` are all `const Vector{Tuple{Int,Int}}`, this is type-stable. However, using `AbstractVector` or a tuple would avoid the dynamic dispatch overhead of iterating a heap-allocated vector. Since the directions are known at compile time, a `Tuple` or `SVector` would be slightly faster.

**Impact**: Negligible — the inner loop over squares dominates.

---

#### M4. `_EVAL_FIELD_BUFS` initialization happens at `__init__` but `FIELD_BUFS` at const time

**File**: `src/energy.jl:247-249` vs `src/search.jl:186`

Energy module buffers are initialized in `__init__()`, while Search module buffers are initialized at const declaration time. Both approaches work, but the `__init__` approach is more robust for precompilation scenarios. The Search module approach could fail if `Threads.nthreads()` changes between precompilation and runtime (unlikely but possible with `--threads`).

**Impact**: None in practice.

---

#### M5. Dead code: `update_piece_field!` and `find_ray_blockers!` exported but only used via `apply_with_field!`

**File**: `src/fields.jl:101`

These functions are exported for the optimizer, but the optimizer only calls them through `apply_with_field!` in search.jl. Direct export is still useful for testing (verify_field_updater.jl uses them). **No change needed.**

---

#### M6. `gui.html` referenced in OVERVIEW.jl but doesn't exist as a separate file

**File**: `OVERVIEW.jl:192`

The file structure lists `src/gui.html` but:
```
$ ls src/
FieldEngine.jl  energy.jl  fields.jl  optimize.jl  play.jl  search.jl  state.jl
```

No `gui.html` exists. This is a future feature listed in Phase 4 but documented as if it already exists.

**Fix**: Update OVERVIEW.jl to mark gui.html as future/planned.

---

#### M7. Test assertions could be stronger in `test_fields_energy.jl`

**File**: `test/test_fields_energy.jl:182-183`

```julia
let b = new_board()
    score = evaluate(b)
    test("starting eval near zero", abs(score) < 2.0)
end
```

A threshold of 2.0 pawns is quite loose for the starting position. The current weights produce a score very close to 0.0 for the symmetric starting position. A tighter bound (e.g., `< 0.5`) would catch weight regressions.

---

#### M8. `perft_suite.jl` doesn't test perft(5) for starting position

**File**: `test/perft_suite.jl:78-81`

The standard starting position is only tested to depth 4 (197,281 nodes). Depth 5 (4,865,609) would provide stronger verification and still runs in reasonable time (~seconds for a correct generator). Position 3 does test to depth 5, which is good.

---

### SUGGESTIONS — Architectural improvements or future considerations

---

#### S1. Consider bitboard representation for attack detection

The current `is_square_attacked` function scans in all 8 directions for each call. For the inner move-generation legality check (called once per pseudo-legal move), this means O(64) work per move × ~30-40 pseudo-legal moves = ~2000 square checks per position. A precomputed attack bitboard would reduce this to O(1) per square, but would require significant refactoring of the state representation. This is a long-term consideration.

---

#### S2. Aspiration windows in iterative deepening

The root search (`best_move` and `choose_move!`) uses `(-INF, INF)` as the initial window for each depth. Aspiration windows (starting with a narrow window around the previous depth's score and widening on fail-high/fail-low) would significantly reduce the search tree for most positions.

---

#### S3. Move ordering: add killer moves and history heuristic

Currently, move ordering is MVV-LVA for captures and nothing for quiet moves (they sort after captures with score -1). Adding killer moves (quiet moves that caused beta cutoffs at sibling nodes) and history heuristic (incrementing a counter for quiet moves that cause cutoffs) would substantially improve pruning efficiency.

---

#### S4. Consider using `Float32` for the 8x8 field matrices

The field computation doesn't need Float64 precision. Using Float32 would halve memory bandwidth in the hot loops and potentially double throughput on SIMD-capable hardware. The king's charge of 100.0 and the tension computation (which squares field values) would still be well within Float32's range (~3.4e38 max, ~7 decimal digits precision).

---

#### S5. The `CHECKMATE_SCORE` constant is duplicated

**Files**: `src/energy.jl:40` and `src/search.jl:45`

Both files define `const CHECKMATE_SCORE = 10000.0` independently. If one changes without the other, mate detection breaks. Search should import from Energy, or a shared constants file should be created.

---

## Design Principle Violations (Section 7)

The engine is remarkably clean in terms of design principle compliance. I found **zero** instances of hardcoded chess heuristics:

- No piece-square tables
- No opening book (the OVERVIEW mentions one, but it's not implemented)
- No endgame-specific logic
- No explicit king safety rules (king safety emerges from the king's massive charge)
- No hardcoded pawn structure evaluation
- The only "chess knowledge" is the piece charges and the rules of the game

The one arguable gray area is the pawn field shape (fields.jl:249-296), which has directional projections that mirror how pawns actually work in chess. But this is justified by the physics model (pawns are directional force emitters) and the charges themselves are the only tunable parameter, not the field shape. The field shapes are consequences of piece movement rules, which are "laws of the game" (allowed per the design principles).

The MVV-LVA move ordering (search.jl:286-295) could be considered chess knowledge, but it's standard search optimization that doesn't affect evaluation — it only affects search speed.

**Verdict**: The design principles are well-maintained. No violations found.

---

## Testing Gaps (Section 8)

### What's well-tested:
- Perft: 6 standard positions × depths 1-4/5 (excellent coverage)
- Apply/undo integrity: All 6 positions tested
- Repetition detection: 40 thorough tests covering 2-fold, 3-fold, undo/redo
- Incremental field update: 65 tests including slider overlaps, castling, ep, promotion
- Search basics: Perft, mate-in-1, qsearch in check

### Critical gaps:

1. **No Zobrist hash consistency test**: There's no test that verifies `_recompute_hash(b)` matches `b.hash` after a sequence of apply/undo operations. The apply/undo integrity test checks that `b.hash` is restored, but doesn't verify that the incremental hash was correct at intermediate positions. A test that applies a sequence of moves and compares incremental hash vs full recomputation at each step would catch subtle Zobrist bugs.

2. **No test for TT correctness**: No test verifies that the transposition table produces the same search result as a search without TT. A simple test: run `best_move` at depth 4 with and without TT, verify same move/score.

3. **No test for NMP/LMR/PVS correctness**: These search enhancements are not tested individually. A test that compares the search result with all enhancements to a plain negamax search would catch implementation bugs.

4. **No symmetry test for evaluation**: No test verifies that `eval(pos) ≈ -eval(color_flipped_pos)`. This would catch the kind of asymmetry discussed in I1.

5. **No test for the 3-term evaluation model**: All search tests use the 5-term weights. The 3-term model path in `eval_w` is untested.

6. **No stress test for search timeout**: The `SearchContext` deadline mechanism is untested. A test that starts a search with a very short deadline and verifies it terminates promptly would be valuable.

7. **Missing perft position**: The "TalkChess" position (`r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1`) with many castling edge cases is not included.

### Suggested test cases:

```julia
# Zobrist consistency after random play
let b = new_board()
    rng = MersenneTwister(99)
    for _ in 1:50
        moves = generate_moves(b)
        isempty(moves) && break
        m = moves[rand(rng, 1:length(moves))]
        apply_move!(b, m)
        @assert b.hash == _recompute_hash(b) "Zobrist mismatch after $(move_to_string(m))"
    end
end

# Evaluation symmetry test
let b = from_fen("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
    score_w = evaluate(b)
    # Flip colors
    b_flipped = copy_board(b)
    for r in 1:8, f in 1:8
        b_flipped.grid[r, f] = -b_flipped.grid[9-r, f]  # negate and mirror
    end
    b_flipped.turn = -b.turn
    sync_board!(b_flipped)
    score_b = evaluate(b_flipped)
    @assert abs(score_w + score_b) < 0.001 "Eval asymmetry: $score_w vs $score_b"
end
```

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| CRITICAL | 2 (C1, C2 — TT/search correctness) |
| IMPORTANT | 5 (I2, I3, I5, I7, I9) |
| MINOR | 8 |
| SUGGESTIONS | 5 |

### Overall Assessment

This is a well-engineered codebase with a creative and sound mathematical foundation. The move generation is **correct** (verified by comprehensive perft tests). The search implementation is solid with all major modern enhancements. The CMA-ES optimizer is production-quality with robust checkpointing, Stockfish UCI integration, and proper parallel game dispatch.

The most impactful fix would be C1/C2 (TT score storage on cutoffs), which will immediately improve search efficiency and play strength. The other issues are mostly minor or performance-related.

The code quality is exceptional for a learning project — the comments are genuinely educational, the architecture is clean with clear module boundaries, and the mathematical model is well-documented. The design principle of "no chess knowledge" is remarkably well-maintained.
