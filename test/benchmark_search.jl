#=
benchmark_search.jl — Before/after benchmark for search hot-path optimizations.

Measures:
  1. Depth-5 search timing (wall clock, 5 runs)
  2. Allocation bytes per search
  3. Per-subsection allocation (eval_w, mobility, field ops)
  4. Profile flat output (top functions by time)

Run:  julia test/benchmark_search.jl
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search
using Profile
using Printf

# ═══════════════════════════════════════════════════════════════
# Fixed benchmark position: Kiwipete (complex middlegame)
# ═══════════════════════════════════════════════════════════════
const BENCH_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

function run_benchmark()
    println("═══════════════════════════════════════════════════")
    println("  Search Hot-Path Benchmark (Kiwipete, depth 5)")
    println("═══════════════════════════════════════════════════\n")

    # ── JIT warmup ────────────────────────────────────────────────
    print("Warming up JIT... ")
    b_warmup = from_fen(BENCH_FEN)
    best_move(b_warmup; max_depth=5, verbose=false)
    println("done.\n")

    # ── Timing benchmark (5 runs) ────────────────────────────────
    println("─── Timing (5 runs, depth 5) ───\n")
    times = Float64[]
    allocs = Int[]
    for i in 1:5
        b = from_fen(BENCH_FEN)
        a = Ref(0)
        t = @elapsed begin
            a[] = @allocated best_move(b; max_depth=5, verbose=false)
        end
        push!(times, t)
        push!(allocs, a[])
        @printf("  Run %d: %.4fs, %.2f KB allocated\n", i, t, a[]/1024)
    end
    avg_time  = sum(times) / length(times)
    avg_alloc = sum(allocs) / length(allocs)
    min_time  = minimum(times)
    @printf("\n  Average : %.4fs, %.2f KB\n", avg_time, avg_alloc/1024)
    @printf("  Best    : %.4fs\n\n", min_time)

    # ── Verbose single run to show node counts ────────────────────
    println("─── Search output (depth 5) ───\n")
    b = from_fen(BENCH_FEN)
    m, s = best_move(b; max_depth=5, verbose=true)
    @printf("\n  Best move: %s, score: %+.3f\n\n", move_to_string(m), s)

    # ── Per-subsection allocation profiling ───────────────────────
    println("─── Per-subsection allocations ───\n")

    b = from_fen(BENCH_FEN)
    sync_board!(b)
    tid = 1
    Search.ensure_ply_buffers!(tid, 2)

    w = Search.DEFAULT_WEIGHTS
    field = Search.FIELD_BUFS[tid]
    compute_total_field!(field, b)

    # eval_w (called at every qsearch leaf)
    a_eval = @allocated for _ in 1:100_000
        eval_w(b, w, field)
    end
    @printf("  eval_w × 100k:              %8d bytes (%.2f B/call)\n", a_eval, a_eval/100_000)

    # compute_mobility_count (called 2× per eval_w)
    a_mob = @allocated for _ in 1:100_000
        compute_mobility_count(b, WHITE)
        compute_mobility_count(b, BLACK)
    end
    @printf("  mobility_count × 100k:      %8d bytes (%.2f B/call)\n", a_mob, a_mob/100_000)

    # generate_moves!
    legal_buf = Move[]
    pseudo_buf = Move[]
    a_movegen = @allocated for _ in 1:100_000
        generate_moves!(legal_buf, b, pseudo_buf)
    end
    @printf("  generate_moves! × 100k:     %8d bytes (%.2f B/call)\n", a_movegen, a_movegen/100_000)

    # apply_move!/undo_move! cycle
    moves = generate_moves(b)
    m0 = moves[1]
    a_apply = @allocated for _ in 1:100_000
        undo = apply_move!(b, m0)
        undo_move!(b, m0, undo)
    end
    @printf("  apply/undo × 100k:          %8d bytes (%.2f B/call)\n", a_apply, a_apply/100_000)

    # is_square_attacked / is_in_check
    a_check = @allocated for _ in 1:100_000
        is_in_check(b, WHITE)
    end
    @printf("  is_in_check × 100k:         %8d bytes (%.2f B/call)\n", a_check, a_check/100_000)

    # ── CPU Profile ───────────────────────────────────────────────
    println("\n─── CPU Profile (depth 5 search, flat top-20) ───\n")

    Profile.clear()
    b = from_fen(BENCH_FEN)
    Profile.@profile best_move(b; max_depth=5, verbose=false)
    Profile.print(; format=:flat, sortedby=:count, mincount=10, noisefloor=2)

    println("\n═══ Benchmark complete ═══")
end

run_benchmark()
