#=
debug_ep_case.jl — Debug the failing en passant field update case

Failing case from verify_field_updater test suite:
  FEN:  rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3
  move: e5f6  (described as white pawn on f5 captures black pawn on e5, moving to f6)
  Expected max_diff: ~0
  Actual max_diff:   1.00 (HUGE)

This script:
  1. Loads the position from the FEN
  2. Finds and applies the en passant move with debug=true
  3. Also computes the full field recomputation
  4. Prints both side-by-side and shows the difference

Run with:
    julia test/debug_ep_case.jl
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search
using Printf

# ── Helpers ────────────────────────────────────────────────────────────────

function print_field(label::String, f::Matrix{Float64})
    println("  $label:")
    for rank in 8:-1:1
        print("    rank $rank: ")
        for file in 1:8
            @printf("%7.3f ", f[rank, file])
        end
        println()
    end
end

function print_diff(f_inc::Matrix{Float64}, f_ref::Matrix{Float64})
    println("  Difference (incremental - full_recompute):")
    max_diff = 0.0
    max_sq   = (0, 0)
    for rank in 8:-1:1
        print("    rank $rank: ")
        for file in 1:8
            d = f_inc[rank, file] - f_ref[rank, file]
            d != 0.0 && (abs(d) > abs(max_diff)) && (max_diff = d; max_sq = (rank, file))
            @printf("%7.3f ", d)
        end
        println()
    end
    files = ["a","b","c","d","e","f","g","h"]
    sq_label = max_sq == (0,0) ? "none" : "$(files[max_sq[2]])$(max_sq[1])"
    println("  max |diff| = $(abs(max_diff))  at square $sq_label")
    return abs(max_diff)
end

# ── Main debug section ──────────────────────────────────────────────────────

const TARGET_FEN = "rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3"

println("="^70)
println("DEBUG: En passant field update case")
println("="^70)
println("FEN: $TARGET_FEN")
println()

b = from_fen(TARGET_FEN)
println("Board position (grid values):")
for rank in 8:-1:1
    print("  rank $rank: ")
    for file in 1:8
        @printf("%7.2f ", b.grid[rank, file])
    end
    println()
end
println()
println("Turn: $(b.turn == State.WHITE ? "WHITE" : "BLACK")")
println("En passant square: $(b.en_passant)")
println()

# Find all legal moves
moves = generate_moves(b)
println("All legal moves ($(length(moves)) total):")
for m in moves
    ep_flag = m.is_en_passant ? " [EN PASSANT]" : ""
    cast_flag = m.is_castling ? " [CASTLING]" : ""
    @printf("  %s%s%s\n", move_to_string(m), ep_flag, cast_flag)
end
println()

# Find the e5f6 move specifically
target_move = nothing
for m in moves
    if m.from_rank == 5 && m.from_file == 5 && m.to_rank == 6 && m.to_file == 6
        global target_move = m
        break
    end
end

# Also find any en passant moves
ep_moves = [m for m in moves if m.is_en_passant]

println("="^70)
println("Looking for move e5f6 (from=(5,5) to=(6,6)):")
if target_move !== nothing
    println("  FOUND: $(move_to_string(target_move))  is_en_passant=$(target_move.is_en_passant)")
else
    println("  NOT FOUND in legal move list")
end
println()
println("En passant moves found: $(length(ep_moves))")
for m in ep_moves
    println("  $(move_to_string(m))  from=($(m.from_rank),$(m.from_file)) to=($(m.to_rank),$(m.to_file))")
end
println()

# Pick the move to debug: prefer e5f6 if found, else first ep move
debug_move = target_move !== nothing ? target_move : (!isempty(ep_moves) ? first(ep_moves) : nothing)

if debug_move === nothing
    println("No suitable move found to debug. Trying all ep moves from nearby positions...")
    # Try alternate FENs with proper ep square set
    alt_fens = [
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "4k3/8/8/R2pP3/8/8/8/4K3 w - d6 0 1",
    ]
    for fen in alt_fens
        b2 = from_fen(fen)
        mvs = generate_moves(b2)
        ep = [m for m in mvs if m.is_en_passant]
        if !isempty(ep)
            println("Found ep move in alt FEN: $fen")
            global debug_move = first(ep)
            global b = b2
            break
        end
    end
end

if debug_move === nothing
    println("ERROR: Could not find any en passant move to debug.")
    exit(1)
end

println("="^70)
println("Debugging move: $(move_to_string(debug_move))")
println("  is_en_passant = $(debug_move.is_en_passant)")
println("  from = (rank=$(debug_move.from_rank), file=$(debug_move.from_file))")
println("  to   = (rank=$(debug_move.to_rank),   file=$(debug_move.to_file))")
println()

# Compute reference full field BEFORE the move
f_before = zeros(Float64, 8, 8)
compute_total_field!(f_before, b)
println("field_sum before move: $(sum(f_before))")
println()

# ── Path A: Incremental update with debug=true ─────────────────────────────
println("="^70)
println("PATH A: Incremental update (apply_with_field! with debug=true)")
println("="^70)

b_inc = copy_board(b)
f_inc = copy(f_before)
from_buf = FROM_SLIDERS[1]
to_buf   = TO_SLIDERS[1]
seen     = FROM_SEEN[1]

undo = apply_with_field!(f_inc, b_inc, debug_move, from_buf, to_buf, seen, true)
println()
println("Incremental field_sum after move: $(sum(f_inc))")

# ── Path B: Full recomputation ─────────────────────────────────────────────
println()
println("="^70)
println("PATH B: Full recomputation after apply_move!")
println("="^70)

b_ref = copy_board(b)
apply_move!(b_ref, debug_move)
f_ref = zeros(Float64, 8, 8)
compute_total_field!(f_ref, b_ref)
println("Full recompute field_sum after move: $(sum(f_ref))")

# ── Side-by-side comparison ────────────────────────────────────────────────
println()
println("="^70)
println("COMPARISON: incremental vs full recompute")
println("="^70)
println()
print_field("Incremental field", f_inc)
println()
print_field("Full recompute field", f_ref)
println()
max_diff = print_diff(f_inc, f_ref)
println()

if max_diff == 0.0
    println("✓ Fields match exactly — no divergence for this move.")
else
    println("✗ DIVERGENCE DETECTED: max_diff = $max_diff")
    println()
    println("Diagnosis:")
    println("  is_en_passant flag = $(debug_move.is_en_passant)")
    if debug_move.is_en_passant
        println("  → Move IS flagged as en passant → takes the full-recompute path")
        println("    The full-recompute path should always be correct.")
        println("    If there's still a divergence, check compute_total_field! or apply_move!.")
    else
        println("  → Move is NOT flagged as en passant → takes the normal incremental path")
        println("    The captured pawn is NOT at to_sq — it's at (from_rank, to_file) for white")
        println("    or (from_rank, to_file) for black. The normal path checks b.grid[to_sq]")
        println("    which is EMPTY for en passant, so the captured pawn's contribution")
        println("    is never subtracted. This explains the 1.00 divergence (pawn charge = 1.0).")
    end
end

println()
println("="^70)
