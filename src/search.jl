#=
search.jl — Unified field-guided game tree search

This is the SINGLE search implementation used by both interactive play
(play.jl, best_move) and the weight optimizer (optimize.jl, choose_move!).

"Tune what you play": the optimizer tunes weights against the exact same
search algorithm that plays games. No more divergence between the two paths.

Search features (all from the optimizer's battle-tested code):
    • Negamax alpha-beta with iterative deepening
    • Null Move Pruning (NMP) — skip a turn; if still winning, prune
    • Late Move Reductions (LMR) — search bad moves at reduced depth
    • Principal Variation Search (PVS) — zero-window for non-PV moves
    • Quiescence search with delta pruning — no horizon effect
    • MVV-LVA move ordering — best captures first for maximum cutoffs
    • Packed transposition table — per-game, Vector-based, zero-alloc
    • Incremental field maintenance — field updated per move, not recomputed
    • Pre-allocated per-thread buffers — zero heap allocation in hot path
    • Repetition detection — 2-fold in search treated as draw

Score convention: all internal scores are from the CURRENT PLAYER's
perspective (negamax). best_move() converts to White's perspective.
=#

module Search

using Printf
using ..State
using ..Fields
using ..Energy

# ── Public API (interactive play, tests) ──────────────────────
export best_move, perft, perft_divide

# ── Optimizer API (used by optimize.jl game loops) ────────────
export eval_w, apply_with_field!, choose_move!, new_tt
export ensure_ply_buffers!
export FIELD_BUFS, FIELD_STACK, LEGAL_BUFS, PSEUDO_BUFS
export FROM_SLIDERS, TO_SLIDERS, FROM_SEEN
export TT_EMPTY, TT_SIZE

# ── Constants ─────────────────────────────────────────────────
const INF             = 1.0e9
const CHECKMATE_SCORE = 10000.0   # must match Energy.CHECKMATE_SCORE
const MAX_PLY         = 64        # initial pre-allocation depth; grows if needed

# Total buffer slots: one per Julia thread (default + interactive pools).
const _NBUFS = Threads.nthreads() + Threads.nthreads(:interactive)

# ── Search statistics ─────────────────────────────────────────
# Per-thread counters for verbose output in best_move.
# The optimizer ignores these — the overhead is one increment per node.
mutable struct SearchStats
    nodes::Int
    qnodes::Int
end
const _STATS = [SearchStats(0, 0) for _ in 1:_NBUFS]

# ── Transposition table ──────────────────────────────────────
# Packed, fixed-size, per-game TT.  Each game/search gets its own
# TT so parallel games on different threads never interfere.
#
# Size: 2^16 = 65536 entries × 16 bytes ≈ 1 MB per game.
# Flag values:
#   TT_EXACT = 0 — score is exact (alpha < score < beta)
#   TT_LOWER = 1 — fail-high (score >= beta): lower bound
#   TT_UPPER = 2 — fail-low  (score <= alpha): upper bound
const TT_SIZE  = 1 << 16   # default for interactive play
const TT_EXACT = UInt8(0)
const TT_LOWER = UInt8(1)
const TT_UPPER = UInt8(2)

struct TTEntry
    hash ::UInt64
    score::Float32
    depth::Int8
    flag ::UInt8
end
const TT_EMPTY = TTEntry(UInt64(0), Float32(0), Int8(-1), TT_EXACT)

"""
    new_tt(size = TT_SIZE)

Create a fresh per-game transposition table. `size` must be a power of 2.
The optimizer uses a larger table (1<<18) for better hit rates at depth 5+.
"""
new_tt(size::Int = TT_SIZE) = fill(TT_EMPTY, size)

@inline function tt_probe(tt::Vector{TTEntry}, hash::UInt64,
                           depth::Int, α::Float64, β::Float64)
    mask = UInt64(length(tt) - 1)
    e = @inbounds tt[(hash & mask) + 1]
    e.hash != hash     && return nothing
    Int(e.depth) < depth && return nothing
    s = Float64(e.score)
    e.flag == TT_EXACT && return s
    e.flag == TT_LOWER && s >= β && return s
    e.flag == TT_UPPER && s <= α && return s
    return nothing
end

@inline function tt_store!(tt::Vector{TTEntry}, hash::UInt64,
                            depth::Int, score::Float64, flag::UInt8)
    mask = UInt64(length(tt) - 1)
    idx = (hash & mask) + 1
    @inbounds tt[idx] = TTEntry(hash, Float32(score), Int8(depth), flag)
end

# ── Per-thread pre-allocated buffers ─────────────────────────
#
# THREE sources of allocation eliminated:
#   1. generate_moves  → new Vector{Move} at every node
#   2. find_ray_blockers → small Vector per slider refresh
#   3. field stack save/restore → 8×8 matrix copies per ply
#
# We pre-allocate one set per thread × ply depth so every
# recursive level has its own non-overlapping scratch space.

# Legal move buffers: LEGAL_BUFS[thread][ply]
const LEGAL_BUFS  = [[Move[] for _ in 1:MAX_PLY] for _ in 1:_NBUFS]
# Pseudo-legal scratch: PSEUDO_BUFS[thread][ply]
const PSEUDO_BUFS = [[Move[] for _ in 1:MAX_PLY] for _ in 1:_NBUFS]
# Field buffers: one current-field per thread
const FIELD_BUFS  = [zeros(Float64, 8, 8) for _ in 1:_NBUFS]
# Field save-stack: FIELD_STACK[thread][ply]
const FIELD_STACK = [[zeros(Float64, 8, 8) for _ in 1:MAX_PLY] for _ in 1:_NBUFS]
# Slider work buffers: two per thread (from-square + to-square sliders)
const FROM_SLIDERS = [Tuple{Int,Int}[] for _ in 1:_NBUFS]
const TO_SLIDERS   = [Tuple{Int,Int}[] for _ in 1:_NBUFS]
# O(1) dedup for apply_with_field! (replaces sq ∈ from_buf linear scan)
const FROM_SEEN    = [zeros(Bool, 8, 8) for _ in 1:_NBUFS]
# Score scratch for move ordering — avoids closure allocation in sort!
const SCORE_BUFS   = [[Int[] for _ in 1:MAX_PLY] for _ in 1:_NBUFS]

# Pre-size slider buffers (max 8 per square — one per ray direction)
for _tid in 1:_NBUFS
    sizehint!(FROM_SLIDERS[_tid], 8)
    sizehint!(TO_SLIDERS[_tid],   8)
end

"""
Ensure a thread's ply-indexed buffers extend to at least `ply`.
Rare deep qsearch lines can exceed the initial MAX_PLY preallocation.
"""
@inline function ensure_ply_buffers!(tid::Int, ply::Int)
    if ply <= length(LEGAL_BUFS[tid])
        return
    end
    while length(LEGAL_BUFS[tid]) < ply
        push!(LEGAL_BUFS[tid], Move[])
        push!(PSEUDO_BUFS[tid], Move[])
        push!(FIELD_STACK[tid], zeros(Float64, 8, 8))
        push!(SCORE_BUFS[tid], Int[])
    end
end

# ── Default evaluation weights ───────────────────────────────
# Pulled from Energy module constants. When you update weights in
# energy.jl, interactive play (best_move) uses them automatically.
# The optimizer passes its own candidate weights via eval_w.
const DEFAULT_WEIGHTS = Float64[
    Energy.W_MATERIAL,
    Energy.W_FIELD,
    Energy.W_KING_SAFETY,
    Energy.W_TENSION,
    Energy.W_MOBILITY,
]

# ── Parametric evaluation ────────────────────────────────────
# Evaluates the position using the given weight vector and
# pre-maintained field matrix.  All expensive scans eliminated:
#   b.material   → was sum(b.grid), O(64) → O(1)
#   b.white_king → was find_king scan, O(64) → O(1)
#   b.black_king → same
#   field        → maintained incrementally, not recomputed
function eval_w(b::Board, w::Vector{Float64}, field::Matrix{Float64})::Float64
    material   = b.material
    field_ctrl = sum(field)

    wkr, wkf = b.white_king
    bkr, bkf = b.black_king

    king_score    = Energy.king_zone_pressure(field, bkr, bkf, BLACK) -
                    Energy.king_zone_pressure(field, wkr, wkf, WHITE)
    tension_score = Energy.tension_near_king(field, bkr, bkf) -
                    Energy.tension_near_king(field, wkr, wkf)
    mob_score     = Energy.total_mobility(b, WHITE) - Energy.total_mobility(b, BLACK)

    return (w[1] * material      +
            w[2] * field_ctrl    +
            w[3] * king_score    +
            w[4] * tension_score +
            w[5] * mob_score)
end

# ── MVV-LVA move ordering ───────────────────────────────────
# Most Valuable Victim / Least Valuable Attacker.
# Captures sorted so queen×pawn (89) >> pawn×queen (1).
# Non-captures return -1 to sort after all captures.
#
# This is the MOST important alpha-beta optimization: good ordering
# reduces the effective branching factor from ~30 to ~8-10 at depth 5.
@inline function mvv_lva_score(b::Board, m::Move)::Int
    if m.is_en_passant
        return 10 * 1 - 1   # pawn captures pawn
    end
    captured_val = b.grid[m.to_rank, m.to_file]
    captured_val == 0.0 && return -1
    victim   = Int(round(abs(captured_val)))
    attacker = Int(round(abs(b.grid[m.from_rank, m.from_file])))
    return 10 * victim - attacker
end

# ── Incremental field update ─────────────────────────────────
# Apply a move to the board AND update the field matrix incrementally.
#
# The update has three phases:
#   1. BEFORE apply_move!: subtract contributions that will change
#      — moving piece (leaving from_sq)
#      — captured piece (disappearing)
#      — sliders blocked by from_sq (their rays will extend)
#      — sliders through to_sq (their rays will be clipped)
#   2. apply_move! — modifies b.grid
#   3. AFTER apply_move!: add new contributions using updated board
#
# Special moves (castling, en passant) fall back to full recompute
# (~50 per game, negligible).
@inline function apply_with_field!(field::Matrix{Float64}, b::Board, m::Move,
                                    from_buf::Vector{Tuple{Int,Int}},
                                    to_buf::Vector{Tuple{Int,Int}},
                                    seen::Matrix{Bool},
                                    debug::Bool = false)::State.UndoInfo
    fr, ff = m.from_rank, m.from_file
    tr, tf = m.to_rank, m.to_file

    if debug
        println("apply_with_field! DEBUG")
        println("  move: $(move_to_string(m))  from=($(fr),$(ff)) to=($(tr),$(tf))")
        println("  is_castling=$(m.is_castling)  is_en_passant=$(m.is_en_passant)")
        println("  piece at from: $(b.grid[fr, ff])  piece at to: $(b.grid[tr, tf])")
        println("  en_passant square: $(b.en_passant)")
        println("  field_sum before: $(sum(field))")
    end

    # Special moves: full recompute is simpler and they're rare
    if m.is_castling || m.is_en_passant
        undo = apply_move!(b, m)
        compute_total_field!(field, b)
        if debug
            println("  [special move path: full recompute]")
            println("  field_sum after full recompute: $(sum(field))")
        end
        return undo
    end

    # Normal move: incremental update
    captured = b.grid[tr, tf]

    # Phase 1: subtract old contributions
    update_piece_field!(field, b, fr, ff, -1)
    captured == 0.0 || update_piece_field!(field, b, tr, tf, -1)

    # Sliders through from_sq will change (ray may extend past it)
    empty!(from_buf)
    find_ray_blockers!(from_buf, b, fr, ff)
    for sq in from_buf
        update_piece_field!(field, b, sq[1], sq[2], -1)
        seen[sq[1], sq[2]] = true
    end

    # Sliders through to_sq change only if to_sq was empty (new blocker)
    empty!(to_buf)
    if captured == 0.0
        find_ray_blockers!(to_buf, b, tr, tf)
        for sq in to_buf
            sq == (fr, ff)    && continue
            seen[sq[1], sq[2]] && continue
            update_piece_field!(field, b, sq[1], sq[2], -1)
        end
    end

    if debug
        println("Phase 1 (subtract):")
        println("  from_buf: $(from_buf)")
        println("  to_buf: $(to_buf)")
        overlap = [sq for sq in from_buf if sq in to_buf]
        println("  overlap (in both): $(overlap)")
        println("  captured at to_sq: $(captured)")
        field_sum = sum(field)
        println("  field after subtract: $(field_sum)")
    end

    # Phase 2: apply the move
    undo = apply_move!(b, m)

    # Phase 3: add new contributions using updated board
    update_piece_field!(field, b, tr, tf, 1)
    for sq in from_buf
        update_piece_field!(field, b, sq[1], sq[2], 1)
        # Keep seen=true so to_buf loop below skips pieces already added here
    end

    if debug
        println("Phase 3a (add from_buf):")
        println("  field after adding from_buf: $(sum(field))")
    end

    if captured == 0.0
        for sq in to_buf
            sq == (fr, ff)    && continue
            seen[sq[1], sq[2]] && continue  # skip if already added via from_buf
            update_piece_field!(field, b, sq[1], sq[2], 1)
        end
    end

    if debug
        println("Phase 3b (add to_buf):")
        println("  field after adding to_buf: $(sum(field))")
        println("Final incremental field: $(sum(field))")
    end

    # Reset seen flags now that both add loops are done
    for sq in from_buf
        seen[sq[1], sq[2]] = false
    end

    return undo
end

# Convenience: looks up thread-local slider buffers by thread id.
@inline function apply_with_field!(field::Matrix{Float64}, b::Board, m::Move,
                                    tid::Int)::State.UndoInfo
    return apply_with_field!(field, b, m,
                             FROM_SLIDERS[tid], TO_SLIDERS[tid], FROM_SEEN[tid])
end

# ── Quiescence search ────────────────────────────────────────
# At depth=0, keep searching captures until the position is "quiet."
# Prevents the horizon effect: stopping where a piece is en prise
# produces wildly inaccurate scores.
#
# Stand pat: if the static eval already beats beta, prune.
# Delta pruning: if even the best possible capture won't reach alpha.
# If the side to move is in check, stand-pat is illegal: qsearch must
# search all evasions and detect mate/stalemate directly at the leaf.
function qsearch(b::Board, w::Vector{Float64},
                 α::Float64, β::Float64,
                 field::Matrix{Float64},
                 ply::Int,
                 tt::Vector{TTEntry})::Float64
    tid = Threads.threadid()
    _STATS[tid].qnodes += 1

    b.halfmove >= 100 && return 0.0

    hit = tt_probe(tt, b.hash, 0, α, β)
    hit !== nothing && return hit

    ensure_ply_buffers!(tid, ply)
    legal_buf  = LEGAL_BUFS[tid][ply]
    pseudo_buf = PSEUDO_BUFS[tid][ply]
    from_buf   = FROM_SLIDERS[tid]
    to_buf     = TO_SLIDERS[tid]
    seen       = FROM_SEEN[tid]
    fstack     = FIELD_STACK[tid]

    State.generate_moves!(legal_buf, b, pseudo_buf)
    in_check = is_in_check(b, b.turn)

    if isempty(legal_buf)
        return in_check ? -(CHECKMATE_SCORE - Float64(ply)) : 0.0
    end

    stand_pat = -INF
    if !in_check
        stand_pat = Float64(b.turn) * eval_w(b, w, field)
        stand_pat >= β && return β
        α = max(α, stand_pat)
    end

    orig_α = α
    for m in legal_buf
        if !in_check
            # Only captures and en passant in normal qsearch.
            (!m.is_en_passant && b.grid[m.to_rank, m.to_file] == 0.0) && continue

            # Delta pruning is only sound when stand-pat is legal.
            cap_val   = abs(b.grid[m.to_rank, m.to_file])
            promo_val = m.promotion != 0.0 ? QUEEN : 0.0
            stand_pat + cap_val + promo_val + QUEEN < α && continue
        end

        copyto!(fstack[ply], field)
        undo  = apply_with_field!(field, b, m, from_buf, to_buf, seen)
        score = -qsearch(b, w, -β, -α, field, ply + 1, tt)
        undo_move!(b, m, undo)
        copyto!(field, fstack[ply])

        score >= β && return β
        α = max(α, score)
    end

    flag = α <= orig_α ? TT_UPPER : (α >= β ? TT_LOWER : TT_EXACT)
    tt_store!(tt, b.hash, 0, α, flag)
    return α
end

# ── Negamax with alpha-beta ──────────────────────────────────
# Full search with all optimizations:
#   1. Transposition table — skip already-searched positions
#   2. Null move pruning — prune if even "passing" beats beta
#   3. Late Move Reductions — search bad moves at reduced depth
#   4. Principal Variation Search — zero-window for non-PV moves
#   5. MVV-LVA move ordering — best captures first
#   6. Quiescence at depth=0 — no horizon effect
#   7. Zero allocation — all buffers pre-allocated per thread × ply
#   8. Repetition detection — 2-fold draw
#
# `is_null` prevents two consecutive null moves (unsound).
function negamax(b::Board, w::Vector{Float64},
                 depth::Int, α::Float64, β::Float64,
                 field::Matrix{Float64},
                 ply::Int,
                 tt::Vector{TTEntry},
                 is_null::Bool = false)::Float64
    tid = Threads.threadid()
    _STATS[tid].nodes += 1

    # ── Draw detection (before TT) ───────────────────────────────
    b.halfmove >= 100 && return 0.0
    is_repetition(b) && return 0.0

    # ── Transposition table probe ────────────────────────────────
    orig_α = α
    hit = tt_probe(tt, b.hash, depth, α, β)
    hit !== nothing && return hit

    # ── Terminal: enter quiescence ───────────────────────────────
    depth == 0 && return qsearch(b, w, α, β, field, ply, tt)

    ensure_ply_buffers!(tid, ply)
    legal_buf  = LEGAL_BUFS[tid][ply]
    pseudo_buf = PSEUDO_BUFS[tid][ply]

    State.generate_moves!(legal_buf, b, pseudo_buf)

    if isempty(legal_buf)
        return is_in_check(b, b.turn) ? -(CHECKMATE_SCORE - Float64(ply)) : 0.0
    end

    in_check = is_in_check(b, b.turn)

    # ── Null move pruning ────────────────────────────────────────
    if !in_check && !is_null && depth >= 3 && ply > 1
        R = depth >= 6 ? 3 : 2
        b.turn = -b.turn
        b.hash ⊻= ZOBRIST_SIDE
        old_ep = b.en_passant
        if b.en_passant != (0,0)
            b.hash ⊻= ZOBRIST_EP[b.en_passant[2]]
            b.en_passant = (0,0)
        end

        null_score = -negamax(b, w, depth - 1 - R, -β, -β + 1.0,
                              field, ply + 1, tt, true)

        b.turn       = -b.turn
        b.hash       ⊻= ZOBRIST_SIDE
        b.en_passant = old_ep
        old_ep != (0,0) && (b.hash ⊻= ZOBRIST_EP[old_ep[2]])

        null_score >= β && return β
    end

    # ── MVV-LVA move ordering ────────────────────────────────────
    score_buf = SCORE_BUFS[tid][ply]
    n_moves   = length(legal_buf)
    resize!(score_buf, n_moves)
    for i in 1:n_moves
        score_buf[i] = mvv_lva_score(b, legal_buf[i])
    end
    for i in 2:n_moves
        key_s = score_buf[i]; key_m = legal_buf[i]
        j = i - 1
        while j >= 1 && score_buf[j] < key_s
            score_buf[j+1] = score_buf[j]; legal_buf[j+1] = legal_buf[j]
            j -= 1
        end
        score_buf[j+1] = key_s; legal_buf[j+1] = key_m
    end

    from_buf = FROM_SLIDERS[tid]
    to_buf   = TO_SLIDERS[tid]
    seen     = FROM_SEEN[tid]
    fstack   = FIELD_STACK[tid]

    best_score = -INF
    for move_idx in 1:n_moves
        m = legal_buf[move_idx]

        # Must check before apply — grid changes after
        is_quiet = (!m.is_en_passant && b.grid[m.to_rank, m.to_file] == 0.0 &&
                    m.promotion == 0.0)

        copyto!(fstack[ply], field)
        undo = apply_with_field!(field, b, m, from_buf, to_buf, seen)

        # ── LMR + PVS ───────────────────────────────────────────
        local score::Float64

        if move_idx > 4 && depth >= 3 && is_quiet && !in_check
            score = -negamax(b, w, depth - 2, -α - 1.0, -α,
                             field, ply + 1, tt, false)
            if score > α
                score = -negamax(b, w, depth - 1, -α - 1.0, -α,
                                 field, ply + 1, tt, false)
            end
        elseif move_idx > 1
            score = -negamax(b, w, depth - 1, -α - 1.0, -α,
                             field, ply + 1, tt, false)
        else
            score = -negamax(b, w, depth - 1, -β, -α,
                             field, ply + 1, tt, false)
        end

        if move_idx > 1 && score > α && score < β
            score = -negamax(b, w, depth - 1, -β, -α,
                             field, ply + 1, tt, false)
        end

        undo_move!(b, m, undo)
        copyto!(field, fstack[ply])

        best_score = max(best_score, score)
        if score >= β
            tt_store!(tt, b.hash, depth, β, TT_LOWER)
            return β
        end
        α = max(α, score)
    end

    flag = best_score <= orig_α ? TT_UPPER :
           best_score >= β      ? TT_LOWER : TT_EXACT
    tt_store!(tt, b.hash, depth, best_score, flag)
    return best_score
end

# ── Root search (optimizer) ──────────────────────────────────
# Iterative deepening 1 → depth.  Each depth seeds TT entries that
# improve ordering at the next depth.  Returns the best move.
# `legal_buf` must already contain legal moves for the side to move.
function choose_move!(b::Board,
                      w::Vector{Float64},
                      depth::Int,
                      field::Matrix{Float64},
                      tt::Vector{TTEntry},
                      tid::Int,
                      legal_buf::Vector{Move})::Move
    fstack    = FIELD_STACK[tid]
    from_buf  = FROM_SLIDERS[tid]
    to_buf    = TO_SLIDERS[tid]
    seen      = FROM_SEEN[tid]
    score_buf = SCORE_BUFS[tid][1]

    length(legal_buf) == 1 && return legal_buf[1]

    best_m = legal_buf[1]

    for d in 1:depth
        n_moves = length(legal_buf)
        resize!(score_buf, n_moves)
        for i in 1:n_moves
            score_buf[i] = legal_buf[i] == best_m ? 2_000_000 :
                           mvv_lva_score(b, legal_buf[i])
        end
        for i in 2:n_moves
            ks = score_buf[i]; km = legal_buf[i]; j = i - 1
            while j >= 1 && score_buf[j] < ks
                score_buf[j+1] = score_buf[j]
                legal_buf[j+1] = legal_buf[j]
                j -= 1
            end
            score_buf[j+1] = ks
            legal_buf[j+1] = km
        end

        best_s = -INF
        for (move_idx, m) in enumerate(legal_buf)
            copyto!(fstack[1], field)
            undo = apply_with_field!(field, b, m, from_buf, to_buf, seen)

            local score::Float64
            if move_idx > 1
                score = -negamax(b, w, d - 1, -best_s - 1.0, -best_s,
                                 field, 2, tt, false)
                if score > best_s
                    score = -negamax(b, w, d - 1, -INF, INF,
                                     field, 2, tt, false)
                end
            else
                score = -negamax(b, w, d - 1, -INF, INF,
                                 field, 2, tt, false)
            end

            undo_move!(b, m, undo)
            copyto!(field, fstack[1])

            if score > best_s
                best_s = score
                best_m = m
            end
        end
    end

    return best_m
end

# ── Root search (interactive play) ───────────────────────────
# Convenience API: takes just a board and depth.  Creates its own
# TT, computes the initial field, uses default weights from energy.jl.
# Returns (best_move, score) where score is from White's perspective.
function best_move(b::Board; max_depth::Int = 4,
                   verbose::Bool = false)::Tuple{Move, Float64}
    sync_board!(b)
    moves = generate_moves(b)
    isempty(moves) && error("No legal moves — position is terminal")

    if length(moves) == 1
        verbose && println("  (forced move: $(move_to_string(moves[1])))")
        return (moves[1], Float64(b.turn) * Energy.evaluate(b))
    end

    tid = Threads.threadid()
    ensure_ply_buffers!(tid, 2)

    w     = DEFAULT_WEIGHTS
    tt    = new_tt()
    field = FIELD_BUFS[tid]
    compute_total_field!(field, b)

    fstack    = FIELD_STACK[tid]
    from_buf  = FROM_SLIDERS[tid]
    to_buf    = TO_SLIDERS[tid]
    seen      = FROM_SEEN[tid]
    score_buf = SCORE_BUFS[tid][1]

    stats = _STATS[tid]
    stats.nodes  = 0
    stats.qnodes = 0

    best_m = moves[1]
    best_s = 0.0

    for depth in 1:max_depth
        n_moves = length(moves)
        resize!(score_buf, n_moves)
        for i in 1:n_moves
            score_buf[i] = moves[i] == best_m ? 2_000_000 :
                           mvv_lva_score(b, moves[i])
        end
        for i in 2:n_moves
            ks = score_buf[i]; km = moves[i]; j = i - 1
            while j >= 1 && score_buf[j] < ks
                score_buf[j+1] = score_buf[j]
                moves[j+1] = moves[j]
                j -= 1
            end
            score_buf[j+1] = ks
            moves[j+1] = km
        end

        depth_best_m = moves[1]
        depth_best_s = -INF

        for (move_idx, m) in enumerate(moves)
            copyto!(fstack[1], field)
            undo = apply_with_field!(field, b, m, from_buf, to_buf, seen)

            local score::Float64
            if move_idx > 1
                score = -negamax(b, w, depth - 1, -depth_best_s - 1.0, -depth_best_s,
                                 field, 2, tt, false)
                if score > depth_best_s
                    score = -negamax(b, w, depth - 1, -INF, INF,
                                     field, 2, tt, false)
                end
            else
                score = -negamax(b, w, depth - 1, -INF, INF,
                                 field, 2, tt, false)
            end

            undo_move!(b, m, undo)
            copyto!(field, fstack[1])

            if score > depth_best_s
                depth_best_s = score
                depth_best_m = m
            end
        end

        best_m = depth_best_m
        best_s = Float64(b.turn) * depth_best_s

        if verbose
            @printf("  depth %d:  %-8s  score %+.3f  nodes=%d  qnodes=%d\n",
                    depth, move_to_string(best_m), best_s,
                    stats.nodes, stats.qnodes)
        end
    end

    return best_m, best_s
end

# ── Perft ────────────────────────────────────────────────────
function perft(b::Board, depth::Int)::Int
    depth == 0 && return 1
    n = 0
    for m in generate_moves(b)
        undo = apply_move!(b, m)
        n   += perft(b, depth - 1)
        undo_move!(b, m, undo)
    end
    return n
end

function perft_divide(b::Board, depth::Int)
    total = 0
    moves = sort(generate_moves(b), by = move_to_string)
    println("\nPerft($depth) divide:")
    for m in moves
        undo = apply_move!(b, m)
        n    = perft(b, depth - 1)
        undo_move!(b, m, undo)
        @printf("  %-6s  %d\n", move_to_string(m), n)
        total += n
    end
    @printf("  Total: %d\n\n", total)
    return total
end

end # module Search
