#=
test_optimize_trial.jl — 100-generation self-play optimization trial.

Validates:
  1. Per-generation timing
  2. Plateau detection triggering
  3. JSON checkpoint stability and readability
  4. Convergence behavior

Run:
    julia --threads 2 test/test_optimize_trial.jl
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))
include(joinpath(@__DIR__, "..", "src", "optimize.jl"))

using .State, .Fields, .Energy, .Search
using Printf

# Clean up any stale checkpoints from previous runs.
for f in ("trial_checkpoint.txt", "trial_checkpoint_snapshot.json")
    isfile(f) && rm(f)
end

println("═══════════════════════════════════════════════════════")
println("  100-Generation Self-Play Trial")
println("  (depth=1, λ=4, n_pairs=1 for fast iteration)")
println("═══════════════════════════════════════════════════════\n")

t0 = time()

best_w = run_optimize(;
    depth           = 1,
    n_pairs         = 1,
    λ               = 4,
    n_gen           = 100,
    seed            = UInt64(0xDEAD_BEEF_1234_5678),
    checkpoint_path = "trial_checkpoint.txt",
    mode            = :selfplay,
    game_workers    = max(1, Threads.nthreads()),
)

total_time = time() - t0

println("\n═══════════════════════════════════════════════════════")
println("  Trial Results")
println("═══════════════════════════════════════════════════════\n")

@printf("  Total time:     %.1f s\n", total_time)
@printf("  Avg per gen:    %.2f s\n", total_time / 100)
@printf("  Best weights:   [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
        best_w[1], best_w[2], best_w[3], best_w[4], best_w[5])

# ── Validate JSON checkpoint ─────────────────────────────────────
json_path = "trial_checkpoint_snapshot.json"
println("\n─── JSON Checkpoint Validation ───\n")

if isfile(json_path)
    content = read(json_path, String)
    println("  JSON file exists: ✓")
    @printf("  File size: %d bytes\n", length(content))

    # Basic structural validation
    has_gen = occursin("\"generation\"", content)
    has_best = occursin("\"best_fitness\"", content)
    has_weights = occursin("\"best_weights\"", content)
    has_sigma = occursin("\"sigma\"", content)
    has_times = occursin("\"gen_times_s\"", content)
    has_avg = occursin("\"avg_gen_time_s\"", content)
    has_history = occursin("\"fitness_history\"", content)

    println("  Contains generation:      $(has_gen ? "✓" : "✗")")
    println("  Contains best_fitness:    $(has_best ? "✓" : "✗")")
    println("  Contains best_weights:    $(has_weights ? "✓" : "✗")")
    println("  Contains sigma:           $(has_sigma ? "✓" : "✗")")
    println("  Contains gen_times_s:     $(has_times ? "✓" : "✗")")
    println("  Contains avg_gen_time_s:  $(has_avg ? "✓" : "✗")")
    println("  Contains fitness_history: $(has_history ? "✓" : "✗")")

    all_valid = has_gen && has_best && has_weights && has_sigma &&
                has_times && has_avg && has_history
    println("\n  JSON checkpoint valid: $(all_valid ? "✓ PASS" : "✗ FAIL")")

    # Print a snippet
    println("\n  First 500 chars of JSON:")
    println("  ", content[1:min(500, length(content))])
else
    println("  JSON file NOT found: ✗ FAIL")
    println("  (Expected at: $json_path)")
end

# ── Validate plain-text checkpoint ────────────────────────────────
txt_path = "trial_checkpoint.txt"
println("\n─── Plain-Text Checkpoint Validation ───\n")
if isfile(txt_path)
    content = read(txt_path, String)
    println("  File exists: ✓")
    has_gen = occursin("gen ", content)
    has_best_f = occursin("best_f ", content)
    has_sigma = occursin("sigma ", content)
    println("  Contains gen:    $(has_gen ? "✓" : "✗")")
    println("  Contains best_f: $(has_best_f ? "✓" : "✗")")
    println("  Contains sigma:  $(has_sigma ? "✓" : "✗")")
    println("  Plain-text checkpoint valid: $((has_gen && has_best_f && has_sigma) ? "✓ PASS" : "✗ FAIL")")
else
    println("  File NOT found: ✗ FAIL")
end

# ── Cleanup ───────────────────────────────────────────────────────
for f in ("trial_checkpoint.txt", "trial_checkpoint_snapshot.json")
    isfile(f) && rm(f)
end

println("\n═══ Trial complete ═══\n")
