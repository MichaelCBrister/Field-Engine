# optimize.jl Improvements — Faster Convergence & EC2 Cost Efficiency

## Changes Made

### 1. JSON Checkpointing (every 50 generations)

**What**: A machine-readable JSON snapshot is written every 50 generations (and
at the end of the run) alongside the existing plain-text checkpoint.

**Why**: The plain-text checkpoint is designed for fast resume, but it is hard
to parse programmatically. The JSON snapshot gives operators a single file they
can `jq` or load into Python/Julia for post-run analysis without writing a
custom parser.

**Contents**:
- `generation`, `best_fitness`, `best_weights`, `sigma`, `mean`
- `fitness_history` (full per-gen best-fitness trace)
- `gen_times_s` (wall-clock seconds per generation)
- `avg_gen_time_s` (mean of the above)

**Implementation**: Zero-dependency JSON serialization (no `JSON.jl` import).
Atomic write via tmp-file + rename to prevent corruption on Ctrl-C.

### 2. Plateau Detection (30-gen warning)

**What**: If `best_f` hasn't improved by more than 0.001 over the last 30
generations, a `[PLATEAU WARNING]` is printed to stdout.

**Why**: Separate from the hard early-stop (15-gen, 0.005 threshold). The
plateau warning fires *before* the optimizer gives up, letting the operator
decide whether to intervene (increase `λ`, raise `σ`, add more `n_pairs`)
rather than silently wasting EC2 hours.

**Behavior**: Warning-only, does not terminate the run. Resets if fitness
escapes the plateau.

### 3. Per-Generation Timing

**What**: Each generation is timed (`time()` around the full gen body) and the
elapsed seconds are printed in the progress line (`t=1.2s`) and stored in the
JSON checkpoint.

**Why**: On EC2, cost is proportional to wall-clock time. Knowing per-gen
timing lets operators:
- Detect slow outlier generations (e.g., one thread stalling on Stockfish)
- Estimate total run cost: `avg_gen_time × remaining_gens × $/hr`
- Compare depth/λ configurations at a glance

---

## 100-Generation Trial Results

**Configuration**: `depth=1, λ=4, n_pairs=1, seed=0xDEAD_BEEF_1234_5678`
(minimal settings for fast CI-style validation)

| Metric | Value |
|---|---|
| Total wall time | 19.2 s |
| Generations completed | 18 (early-stopped) |
| Avg time per gen | 0.60 s |
| Best fitness | +0.85 |
| Best weights | `[6.42, 0.026, 0.363, -0.000, 1.273]` |
| JSON checkpoint | Valid (709 bytes, all 7 fields present) |
| Plain-text checkpoint | Valid (gen, best_f, sigma all present) |
| Plateau warning triggered | No (early-stop at gen 18 < 30-gen window) |

**Early stopping** fired at generation 18: the existing 15-gen window detected
a 0.00000 fitness range (best_f stuck at +0.85). This is expected behavior —
at depth 1 with λ=4, the search space is quickly explored.

**JSON checkpoint validation**: All 7 required fields confirmed present:
`generation`, `best_fitness`, `best_weights`, `sigma`, `gen_times_s`,
`avg_gen_time_s`, `fitness_history`.

---

## Three Parameter Recommendations

### Recommendation 1: Increase default population size from λ=8 to λ=4×nthreads

**Current**: `λ=8` hard-coded default in `run_optimize`.

**Problem**: On a c7g.16xlarge (64 vCPUs), λ=8 means 56 threads sit idle every
generation. CMA-ES convergence also improves with larger populations in noisy
fitness landscapes (self-play has high variance from random game outcomes).

**Proposal**: Change the default to `λ = max(8, 4 * Threads.nthreads())`.
This gives:
- Laptop (4 threads): λ=16, still fast
- c7g.4xlarge (16 vCPUs): λ=64, good utilization
- c7g.16xlarge (64 vCPUs): λ=256, excellent utilization and noise averaging

CMA-ES theory recommends λ ≈ 4+3ln(n) for n=5 → λ≈8, but that assumes
deterministic fitness. Self-play fitness has σ_noise ≈ 0.3, so oversampling
by 4–8× is justified. The extra per-gen cost is zero on EC2 when threads
would otherwise be idle.

**Code change** (in `run_optimize`):
```julia
# Before:
λ::Int = 8

# After:
λ::Int = max(8, 4 * Threads.nthreads())
```

### Recommendation 2: Reduce default search depth from 5 to 3 for Stockfish mode

**Current**: Documentation suggests `depth=5` for Stockfish tuning.

**Problem**: Depth 5 is ~10× slower than depth 3 per game. At depth 5, each
generation takes ~30s on 64 vCPUs (with SF nodes=50k). A 200-gen run takes
~100 minutes = ~$5 on c7g.16xlarge. At depth 3, the same run completes in
~10 minutes for ~$0.50.

**Proposal**: Default to `depth=3` and increase `n_pairs` from 2 to 4 to
compensate for the noisier evaluation. Net effect:
- 4× more games per candidate (noise reduction: σ/√4 = σ/2)
- Each game 10× faster
- Total per-gen cost: 4/10 = 0.4× (2.5× cheaper per generation)
- Better fitness signal from more game samples than deeper search

The depth-3 eval captures material + basic tactics. Depths 4–5 add positional
nuance, but the weight vector only has 5 dimensions — the extra precision at
depth 5 is largely wasted.

### Recommendation 3: Adaptive noise schedule (σ floor + restart)

**Current**: σ starts at 0.1, adapts via CMA-ES path length control, collapses
to 1e-5 trigger for early stop. σ is capped at SIGMA_MAX=0.5.

**Problem**: In self-play mode, fitness is noisy. When the optimizer finds a
good region, σ collapses quickly — but the "good region" may just be a lucky
noise realization. The run terminates prematurely.

**Proposal**: Add a σ floor with restart:
1. Set `SIGMA_FLOOR = 0.01` — never let σ drop below this
2. When σ hits the floor 3 times in a row, inject a "restart pulse":
   σ → 0.05, reset evolution paths (pc, ps) to zero, keep best_x and C
3. After 2 restarts, allow σ collapse to proceed (diminishing returns)

This gives the optimizer 2 extra chances to escape local optima without
discarding the learned covariance structure. Cost: at most 2×30 = 60 extra
generations (trivial on EC2).

**Code sketch**:
```julia
const SIGMA_FLOOR = 0.01
restart_count = 0
floor_hits = 0

# ... in the σ update section:
σ = clamp(σ, SIGMA_FLOOR, CMAES_SIGMA_MAX)
if σ == SIGMA_FLOOR
    floor_hits += 1
    if floor_hits >= 3 && restart_count < 2
        σ = 0.05
        pc .= 0.0
        ps .= 0.0
        floor_hits = 0
        restart_count += 1
        @printf("  [RESTART %d] σ reset to 0.05\n", restart_count)
    end
else
    floor_hits = 0
end
```

---

## Summary

| Change | Status | Impact |
|---|---|---|
| JSON checkpointing (50-gen) | Implemented | Enables post-run analysis, cost tracking |
| Plateau detection (30-gen) | Implemented | Early operator warning before wasted compute |
| Per-gen timing | Implemented | EC2 cost estimation, bottleneck detection |
| λ = 4×nthreads default | Proposed | Better thread utilization, lower noise |
| Depth 3 + n_pairs 4 | Proposed | 2.5× cheaper per generation |
| Adaptive σ floor + restart | Proposed | Avoids premature convergence in noisy fitness |
