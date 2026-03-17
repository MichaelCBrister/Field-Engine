# energy.jl — Position evaluation via field energy
#
# Converts the field matrices computed in fields.jl into a single scalar score.
# All scores are from White's perspective:
#   Positive → White is better
#   Negative → Black is better
#   Scale ≈ pawn units (1.0 ≈ one pawn advantage)

module Energy

using Printf
using Base.Threads
using ..State
using ..Fields

export evaluate, evaluate_verbose, field_energy

# ── Evaluation weights ─────────────────────────────────────────
# W_MATERIAL dominates by design — a pawn is a pawn.
# The other weights are fractions of a pawn, tunable through play.
#
# W_MATERIAL and W_FIELD are sourced from the completed 3-term CMA-ES run
# (200-gen vs Stockfish skill-5, best fitness -0.5563, early-stopped gen 21).
# W_KING_SAFETY and W_MOBILITY are hand-tuned baselines pending the 5-term run.
# W_TENSION converged to ~0 in both 3-term and early 5-term runs; kept at 0.
const W_MATERIAL    = 3.1238     # Raw material balance (charge sum)
const W_FIELD       = -0.092832  # Net board influence (territorial control)
const W_KING_SAFETY = 0.4380     # Enemy field penetrating the king's shelter
const W_TENSION     = 0.0000     # Tactical tension concentrated near the king
const W_MOBILITY    = 1.2847     # Piece activity (total reachable squares)

# ── 3-term model weight ──────────────────────────────────────
# Field energy — sum of squared field values over all squares.
# In the 3-term model, E = w1*material + w2*sum(Phi) + w3*sum(Phi^2),
# this is the starting weight for the sum(Phi^2) term.
# Sourced from the same completed 3-term CMA-ES run as W_MATERIAL/W_FIELD above.
const W_FIELD_ENERGY = 0.095950

# Score returned for checkmate — large enough to dominate all positional terms.
const CHECKMATE_SCORE = 10000.0

# Thread-local field buffers avoid allocating an 8×8 matrix in every evaluate() call.
# Use a lock to grow the buffer pool on demand if new threads appear after module load.
const _EVAL_FIELD_BUFS = Matrix{Float64}[]
const _EVAL_FIELD_LOCK = ReentrantLock()

function _init_eval_bufs!()
    n = Threads.nthreads() + Threads.nthreads(:interactive)
    resize!(_EVAL_FIELD_BUFS, 0)
    for _ in 1:n
        push!(_EVAL_FIELD_BUFS, zeros(Float64, 8, 8))
    end
end

function _get_eval_buf()::Matrix{Float64}
    tid = Threads.threadid()
    if tid <= length(_EVAL_FIELD_BUFS)
        return @inbounds _EVAL_FIELD_BUFS[tid]
    end
    lock(_EVAL_FIELD_LOCK) do
        while length(_EVAL_FIELD_BUFS) < tid
            push!(_EVAL_FIELD_BUFS, zeros(Float64, 8, 8))
        end
    end
    return @inbounds _EVAL_FIELD_BUFS[tid]
end

# ── Field energy: signed quadratic field observable ──────────────
#
# Computes sum(v * |v|) over all 64 squares, which equals sum(sign(v) * v²).
#
# WHY NOT sum(v²)?
#   Plain sum(v²) is always positive — both kings contribute 100² = 10,000
#   regardless of color. The symmetric starting position would evaluate to
#   ~64,000 instead of 0. CMA-ES cannot optimize a term that doesn't
#   distinguish White from Black.
#
# WHY sum(v * |v|)?
#   - Positive where White's field is stronger (net positive influence)
#   - Negative where Black's field is stronger (net negative influence)
#   - ~0 for symmetric positions (starting position, equal material)
#   - Amplifies strong fields quadratically (a field of 2 contributes 4,
#     not 2) so concentrated attacks/king threats register more than
#     diffuse influence
#   - One O(64) pass, no branches, no spatial filters — pure math
#
# In physics terms: this is the signed energy density of the field,
# integrated over the board lattice.
@inline function field_energy(field::Matrix{Float64})::Float64
    s = 0.0
    @inbounds for v in field
        s += v * abs(v)
    end
    return s
end

# Sum the enemy-colored field magnitude in the 3×3 zone around the king at (kr, kf),
# excluding the king's own square (dominated by its 100-charge; uninformative).
# Takes pre-located king coordinates so the caller can reuse them across helpers.
# Returns 0 when the king is well-sheltered, higher values when exposed.
# The expression max(0, -v * color) extracts enemy contributions without branching:
#   color=+1 (White): enemy field is negative, so -v*color > 0 for enemy squares.
#   color=-1 (Black): enemy field is positive, so -v*color > 0 for enemy squares.
function king_zone_pressure(field::Matrix{Float64}, kr::Int, kf::Int, color::Int)::Float64
    pressure = 0.0
    for dr in -1:1, df in -1:1
        (dr == 0 && df == 0) && continue          # skip king's own square
        r, f = kr + dr, kf + df
        State.in_bounds(r, f) || continue
        pressure += max(0.0, -field[r, f] * color)
    end
    return pressure
end

# Sum the field gradient magnitude in a 3×3 zone around the king at (kr, kf).
# High gradient means competing fields are meeting sharply — tactical pressure.
# Takes pre-located king coordinates so the caller can reuse them.
# Checks all 4 axis-aligned neighbors per square. Interior edges are counted
# twice, but this is intentional: both king zones (White at rank 1, Black at
# rank 8) have equal edge counts this way. Forward-only would give White's king
# zone 12 edges vs Black's 9 — a structural asymmetry that skews the score.
function tension_near_king(field::Matrix{Float64}, kr::Int, kf::Int)::Float64
    total = 0.0
    for dr in -1:1, df in -1:1
        r, f = kr + dr, kf + df
        State.in_bounds(r, f) || continue
        for (δr, δf) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            nr, nf = r + δr, f + δf
            State.in_bounds(nr, nf) || continue
            d = field[r, f] - field[nr, nf]
            total += d * d
        end
    end
    return total
end

# Total squares reachable by all pieces of `color`, using the mobility field.
# Wraps compute_mobility_field with a sum — more squares = more active pieces.
function total_mobility(b::Board, color::Int)::Float64
    return compute_mobility_count(b, color)
end

# Main evaluation function. Returns a score from White's perspective.
# Terminal positions are detected first; field computation only runs on live positions.
#
# ALLOCATION NOTE: The terminal check calls generate_moves, which allocates a
# Vector{Move}. This is acceptable for interactive play (play.jl / search.jl)
# where evaluate() is only called at moderate rates. The optimizer uses eval_w()
# directly which skips this terminal check entirely — negamax_w handles terminals
# via the isempty(legal_buf) check in the search loop, so no allocation occurs
# in the hot path.
function evaluate(b::Board)::Float64
    # 50-move rule is a free check — do it before the expensive generate_moves call.
    b.halfmove >= 100 && return 0.0

    # Terminal position: no legal moves means checkmate or stalemate.
    moves = State.generate_moves(b)
    if isempty(moves)
        return State.is_in_check(b, b.turn) ?
            -Float64(b.turn) * CHECKMATE_SCORE :   # mated side (b.turn) loses
            0.0                                     # stalemate is a draw
    end

    field = _get_eval_buf()
    compute_total_field!(field, b)

    # Material: sum of all signed charges. Kings always cancel (+100 − 100 = 0),
    # so this equals raw piece material in pawn units.
    material = b.material

    # Field control: net influence across the entire board.
    # Captures activity, coordination, and space beyond raw material.
    field_ctrl = sum(field)

    # King positions are stored in Board struct (O(1) lookup, no scan needed).
    wkr, wkf = b.white_king
    bkr, bkf = b.black_king

    # King safety: enemy pressure on Black's king minus enemy pressure on White's.
    # Positive = we're pressing their king more than they're pressing ours.
    king_score = king_zone_pressure(field, bkr, bkf, BLACK) -
                 king_zone_pressure(field, wkr, wkf, WHITE)

    # Attack tension: gradient sharpness near Black's king versus near ours.
    # Rewards positions where our pieces create competing field vectors around
    # the enemy king — a physical analogue of building up an attack.
    tension_score = tension_near_king(field, bkr, bkf) -
                    tension_near_king(field, wkr, wkf)

    # Mobility: how many squares our pieces can reach versus theirs.
    mob_score = total_mobility(b, WHITE) - total_mobility(b, BLACK)

    return (W_MATERIAL    * material     +
            W_FIELD       * field_ctrl   +
            W_KING_SAFETY * king_score   +
            W_TENSION     * tension_score +
            W_MOBILITY    * mob_score)
end

# Print a labeled breakdown of every evaluation component.
# Intended for debugging — lets you see what the engine actually values.
# Computes the total directly from already-computed components rather than
# calling evaluate(b) again, which would recompute everything from scratch.
function evaluate_verbose(b::Board)
    # Verbose mode is used for diagnostics; resync first so printed components
    # reflect the current grid even if the position was hand-constructed.
    sync_board!(b)
    field = compute_total_field(b)

    material  = b.material
    field_ctrl = sum(field)

    wkr, wkf = State.find_king(b, WHITE)
    bkr, bkf = State.find_king(b, BLACK)

    king_score    = king_zone_pressure(field, bkr, bkf, BLACK) -
                    king_zone_pressure(field, wkr, wkf, WHITE)
    tension_score = tension_near_king(field, bkr, bkf) -
                    tension_near_king(field, wkr, wkf)
    mob_w         = total_mobility(b, WHITE)
    mob_b         = total_mobility(b, BLACK)
    mob_score     = mob_w - mob_b

    total = (W_MATERIAL    * material     +
             W_FIELD       * field_ctrl   +
             W_KING_SAFETY * king_score   +
             W_TENSION     * tension_score +
             W_MOBILITY    * mob_score)

    println("\n  ═══ Evaluation Breakdown ═══")
    @printf("  %-22s %+8.2f  × %.2f  =  %+.3f\n",
            "Material:",       material,      W_MATERIAL,    W_MATERIAL    * material)
    @printf("  %-22s %+8.2f  × %.2f  =  %+.3f\n",
            "Field control:",  field_ctrl,    W_FIELD,       W_FIELD       * field_ctrl)
    @printf("  %-22s %+8.2f  × %.2f  =  %+.3f\n",
            "King safety:",    king_score,    W_KING_SAFETY, W_KING_SAFETY * king_score)
    @printf("  %-22s %+8.2f  × %.2f  =  %+.3f\n",
            "Attack tension:", tension_score, W_TENSION,     W_TENSION     * tension_score)
    @printf("  %-22s  W=%.0f B=%.0f → %+.2f  × %.2f  =  %+.3f\n",
            "Mobility:",       mob_w, mob_b,  mob_score,     W_MOBILITY,    W_MOBILITY    * mob_score)
    println("  " * "─"^48)
    @printf("  %-22s                    =  %+.4f\n", "TOTAL:", total)
    side = b.turn == WHITE ? "White" : "Black"
    println("  (+ = White better,  − = Black better,  $side to move)\n")
end

function __init__()
    _init_eval_bufs!()
end

end # module Energy
