#=
profile_search.jl — Profile the search hot path and identify allocation bottlenecks.

Run with:
    julia test/profile_search.jl
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search
using Profile
using Printf

# ═══════════════════════════════════════════════════════════════
# BASELINE MEASUREMENT
# ═══════════════════════════════════════════════════════════════
println("\n═══ Baseline Profiling ═══\n")

# Use Kiwipete — complex middlegame with lots of tactical possibilities
const BENCH_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# Warm up JIT
b_warmup = from_fen(BENCH_FEN)
best_move(b_warmup; max_depth=4, verbose=false)

# ── @time and @allocated on depth-5 search ──
println("─── Depth 5 search (Kiwipete) ───\n")

b = from_fen(BENCH_FEN)
# First run after warmup — measures steady-state
alloc_before = @allocated begin
    m, s = best_move(b; max_depth=5, verbose=true)
end
println()
@printf("  Allocations: %d bytes (%.2f MB)\n", alloc_before, alloc_before / 1e6)

# Repeat for timing stability
println("\n─── Timing (3 runs, depth 5) ───\n")
times = Float64[]
allocs = Int[]
for i in 1:3
    b = from_fen(BENCH_FEN)
    t = @elapsed begin
        a = @allocated best_move(b; max_depth=5, verbose=false)
    end
    push!(times, t)
    push!(allocs, a)
    @printf("  Run %d: %.3fs, %.2f MB allocated\n", i, t, a/1e6)
end
avg_time = sum(times) / length(times)
avg_alloc = sum(allocs) / length(allocs)
@printf("\n  Average: %.3fs, %.2f MB\n", avg_time, avg_alloc/1e6)

# ═══════════════════════════════════════════════════════════════
# SUBSECTION ALLOCATION PROFILING
# ═══════════════════════════════════════════════════════════════
println("\n─── Allocation by subsection ───\n")

b = from_fen(BENCH_FEN)
sync_board!(b)
tid = 1
Search.ensure_ply_buffers!(tid, 2)

w = Search.DEFAULT_WEIGHTS
field = Search.FIELD_BUFS[tid]
compute_total_field!(field, b)

# Profile eval_w (called at every qsearch leaf)
a_eval = @allocated for _ in 1:10000
    eval_w(b, w, field)
end
@printf("  eval_w × 10k:              %8d bytes (%.1f B/call)\n", a_eval, a_eval/10000)

# Profile compute_mobility_count (called twice per eval_w)
a_mob = @allocated for _ in 1:10000
    compute_mobility_count(b, WHITE)
    compute_mobility_count(b, BLACK)
end
@printf("  mobility_count × 10k:      %8d bytes (%.1f B/call)\n", a_mob, a_mob/10000)

# Profile compute_total_field! (called on castling/ep special moves)
a_field = @allocated for _ in 1:10000
    compute_total_field!(field, b)
end
@printf("  compute_total_field! × 10k:%8d bytes (%.1f B/call)\n", a_field, a_field/10000)

# Profile generate_moves! (called at every node)
legal_buf = Move[]
pseudo_buf = Move[]
a_movegen = @allocated for _ in 1:10000
    generate_moves!(legal_buf, b, pseudo_buf)
end
@printf("  generate_moves! × 10k:     %8d bytes (%.1f B/call)\n", a_movegen, a_movegen/10000)

# Profile apply_move!/undo_move! cycle
moves = generate_moves(b)
m0 = moves[1]
a_apply = @allocated for _ in 1:10000
    undo = apply_move!(b, m0)
    undo_move!(b, m0, undo)
end
@printf("  apply/undo × 10k:          %8d bytes (%.1f B/call)\n", a_apply, a_apply/10000)

# Profile update_piece_field!
a_update = @allocated for _ in 1:10000
    update_piece_field!(field, b, 1, 1, 1)
    update_piece_field!(field, b, 1, 1, -1)
end
@printf("  update_piece_field! × 10k: %8d bytes (%.1f B/call)\n", a_update, a_update/10000)

# Profile find_ray_blockers!
slider_buf = Tuple{Int,Int}[]
sizehint!(slider_buf, 8)
a_blockers = @allocated for _ in 1:10000
    empty!(slider_buf)
    find_ray_blockers!(slider_buf, b, 4, 4)
end
@printf("  find_ray_blockers! × 10k:  %8d bytes (%.1f B/call)\n", a_blockers, a_blockers/10000)

# Profile is_in_check
a_check = @allocated for _ in 1:10000
    is_in_check(b, WHITE)
end
@printf("  is_in_check × 10k:         %8d bytes (%.1f B/call)\n", a_check, a_check/10000)

# Profile king_zone_pressure
wkr, wkf = b.white_king
a_kzp = @allocated for _ in 1:10000
    Energy.king_zone_pressure(field, wkr, wkf, WHITE)
end
@printf("  king_zone_pressure × 10k:  %8d bytes (%.1f B/call)\n", a_kzp, a_kzp/10000)

# Profile tension_near_king
a_tnk = @allocated for _ in 1:10000
    Energy.tension_near_king(field, wkr, wkf)
end
@printf("  tension_near_king × 10k:   %8d bytes (%.1f B/call)\n", a_tnk, a_tnk/10000)

# Profile sum(field)
a_sum = @allocated for _ in 1:10000
    sum(field)
end
@printf("  sum(field) × 10k:          %8d bytes (%.1f B/call)\n", a_sum, a_sum/10000)

# Profile TT operations
tt = new_tt()
a_tt = @allocated for _ in 1:10000
    Search.tt_store!(tt, b.hash, 3, 1.5, Search.TT_EXACT)
    Search.tt_probe(tt, b.hash, 3, -1000.0, 1000.0)
end
@printf("  TT store+probe × 10k:      %8d bytes (%.1f B/call)\n", a_tt, a_tt/10000)

# Profile apply_with_field! (incremental field update)
from_buf = Search.FROM_SLIDERS[tid]
to_buf = Search.TO_SLIDERS[tid]
seen = Search.FROM_SEEN[tid]
fstack = Search.FIELD_STACK[tid]
a_awf = @allocated for _ in 1:1000
    copyto!(fstack[1], field)
    undo = apply_with_field!(field, b, m0, from_buf, to_buf, seen)
    undo_move!(b, m0, undo)
    copyto!(field, fstack[1])
end
@printf("  apply_with_field! × 1k:    %8d bytes (%.1f B/call)\n", a_awf, a_awf/1000)

# Profile copyto! (field save/restore)
a_copy = @allocated for _ in 1:10000
    copyto!(fstack[1], field)
    copyto!(field, fstack[1])
end
@printf("  copyto! (field) × 10k:     %8d bytes (%.1f B/call)\n", a_copy, a_copy/10000)

# ═══════════════════════════════════════════════════════════════
# CPU PROFILE (flame graph data)
# ═══════════════════════════════════════════════════════════════
println("\n─── CPU Profile (depth 5 search) ───\n")

Profile.clear()
b = from_fen(BENCH_FEN)
Profile.@profile best_move(b; max_depth=5, verbose=false)

# Print flat profile — top functions by time
Profile.print(; format=:flat, sortedby=:count, mincount=10, noisefloor=2)

println("\n─── Profile tree (top callers) ───\n")
Profile.print(; format=:tree, mincount=50, noisefloor=2)

println("\n═══ Profiling complete ═══\n")
