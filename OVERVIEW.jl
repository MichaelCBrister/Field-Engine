#=
═══════════════════════════════════════════════════════════════════
FIELDENGINE — PROJECT OVERVIEW
═══════════════════════════════════════════════════════════════════

This file is the single source of truth for the FieldEngine project.
It explains what we're building, why, and how — in enough detail that
any developer or AI assistant can understand the full vision and
contribute meaningfully. Read this before touching any code.

═══════════════════════════════════════════════════════════════════
1. WHAT IS FIELDENGINE?
═══════════════════════════════════════════════════════════════════

FieldEngine is a chess engine that knows NOTHING about chess strategy.
It doesn't know what a fork is, what castling means strategically,
or that rooks belong on open files. Instead, it treats chess as a
physics problem:

    • Pieces are charged particles that emit influence fields
    • The board is a potential field landscape
    • Good moves maximize your field energy and disrupt the opponent's
    • Tactics, strategy, and positional understanding EMERGE from math

The engine never thinks "I should control the center" or "that knight
is on a good square." It thinks "the net field energy is higher if
I put this charge here." The fact that this produces recognizable
chess strategy is the whole point — we're discovering chess from
first principles using physics.

═══════════════════════════════════════════════════════════════════
2. WHY JULIA?
═══════════════════════════════════════════════════════════════════

The engine's core computation is:
    - Compute 8×8 field matrices for every piece (linear algebra)
    - Sum them via superposition (matrix addition)
    - Compute gradients/tension (numerical differentiation)
    - Search through thousands of positions per second (tight loops)

In Python, the inner loops would be painfully slow without dropping
into C extensions. In Julia, we write clean, readable math and it
compiles to near-C speed. Julia's type system and multiple dispatch
also make the code very natural for this kind of scientific computing.

The project creator (Chase) is learning Julia through this project.
Code should be heavily commented and educational.

═══════════════════════════════════════════════════════════════════
3. THE MATHEMATICAL MODEL
═══════════════════════════════════════════════════════════════════

3.1 STATE REPRESENTATION (state.jl)
───────────────────────────────────
The board is an 8×8 matrix S of signed real numbers:

    S[rank, file] > 0  →  White piece (magnitude = charge)
    S[rank, file] < 0  →  Black piece (magnitude = charge)
    S[rank, file] = 0  →  Empty square

Piece charges (magnitudes):
    Pawn=1.0  Knight=3.0  Bishop=3.25  Rook=5.0  Queen=9.0  King=100.0

The king's charge is intentionally massive (100). This means it
dominates all field calculations — the engine naturally cares about
king safety and king attacks without being told to. The math does it.

Signed encoding means we can sum the matrix to get material balance,
compute field superposition with simple addition, and the linear
algebra naturally handles friend-vs-foe interactions.

Coordinates: rank 1 = White's back rank, file 1 = a-file.
So S[1,5] = +100.0 is White's king on e1.

3.2 POTENTIAL FIELDS (fields.jl)
────────────────────────────────
Every piece emits an influence field across the board, like a
charged particle emitting an electric/gravitational field:

    Φ(r, c) = Q / (1 + d²)

Where:
    Q = the piece's charge (signed, from the board matrix)
    d = piece-specific distance to (r, c)

The distance metric d varies by piece type:
    • Rook:   distance along rank/file only, blocked by pieces
    • Bishop: distance along diagonals only, blocked by pieces
    • Queen:  combination of rook + bishop
    • Knight: binary — 0.8×Q at L-jump squares, 0 elsewhere
    • Pawn:   directional — forward attack diagonals + push
    • King:   Chebyshev distance, short range (2 squares)

Blocking is critical: a rook behind pawns has almost no field
beyond those pawns, just like in real chess.

The TOTAL field is the superposition (sum) of all individual fields:

    Φ_total(r,f) = Σ_all_pieces Φ_piece(r,f)

This produces an 8×8 matrix where:
    Positive = White-controlled territory
    Negative = Black-controlled territory
    Near zero = contested ground

3.3 FIELD TENSION (fields.jl)
─────────────────────────────
Tension = the gradient magnitude of the field:

    Tension(r,f) = |Φ(r,f) - Φ(r+1,f)|² + |Φ(r,f) - Φ(r,f+1)|²

High tension means the field changes rapidly between adjacent
squares — both sides have strong competing influence. In chess
terms, high-tension squares are where tactics happen (captures,
forks, pins). The engine searches these positions more deeply.

3.4 MOBILITY FIELD (fields.jl)
──────────────────────────────
A separate field counting how many of your pieces can reach each
square. This measures "piece activity" — the flux of the field.
Active pieces = high mobility = high flux.

3.5 ENERGY FUNCTION (energy.jl)
───────────────────────────────
The evaluation combines all field properties into a single score:

    E(S) = w₁·Material + w₂·FieldControl + w₃·KingSafety
         + w₄·Tension + w₅·Mobility

Where:
    Material     = sum(board.grid) — raw piece charge balance
    FieldControl = sum(total_field) — who controls more territory
    KingSafety   = enemy field pressure near opponent's king minus
                   enemy field pressure near your king
    Tension      = field gradient near opponent's king minus near yours
                   (rewards building attacking tension)
    Mobility     = your reachable squares minus opponent's

The weights w₁..w₅ are the ONLY tunable parameters. Currently
hand-tuned. Later they'll be optimized via CMA-ES.

All scores are from White's perspective: positive = White better.

3.6 SEARCH (search.jl)
─────────────────────────────────────
Alpha-beta minimax search, guided by the field:

    • Move ordering: sort by field disruption (biggest energy
      change first → better pruning)
    • Quiescence: keep searching in high-tension positions
      (captures, checks) until the position is "quiet"
    • Depth: 4-6 plies initially, deeper with optimizations

Field-guided move ordering is the key innovation — instead of
generic heuristics, we use the physics to predict which moves
are most important to search.

3.7 OPTIMIZATION (optimize.jl — TO BE BUILT)
─────────────────────────────────────────────
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to
optimize the energy weights w₁..w₅ by self-play:

    1. Generate population of weight vectors
    2. Each plays games against others
    3. Select winners, update distribution
    4. Repeat

CMA-ES is derivative-free — perfect because we can't differentiate
through the game tree. No neural networks, no gradients. Pure
evolutionary optimization.

═══════════════════════════════════════════════════════════════════
4. FILE STRUCTURE
═══════════════════════════════════════════════════════════════════

    field_engine/
    ├── OVERVIEW.jl           ← THIS FILE (read first)
    ├── Project.toml          ← Julia dependencies
    ├── scripts/
    │   ├── setup_stockfish.sh   ← Download pre-compiled Stockfish binary
    │   └── mock_stockfish.sh    ← UCI stub for offline testing
    ├── bin/
    │   └── stockfish         ← Downloaded by setup_stockfish.sh (gitignored)
    ├── src/
    │   ├── FieldEngine.jl    ← Main module (glue file)
    │   ├── state.jl          ← Board as 8×8 matrix, move generation
    │   ├── fields.jl         ← Potential field computation
    │   ├── energy.jl         ← Evaluation (field → score)
    │   ├── search.jl         ← Alpha-beta search with TT, qsearch, IDA
    │   ├── optimize.jl       ← CMA-ES weight tuning vs Stockfish ✅
    │   ├── play.jl           ← Interactive terminal play
    │   └── gui.html          ← Browser-based board visualizer
    └── test/
        ├── test_state.jl     ← Board/move correctness + perft
        ├── test_search.jl    ← Search correctness tests
        └── demo_fields.jl    ← Field visualization demo

═══════════════════════════════════════════════════════════════════
5. PHASE ROADMAP
═══════════════════════════════════════════════════════════════════

PHASE 1: Mathematical Foundation ✅ COMPLETE
    ✅ Board as 8×8 signed matrix (state.jl)
    ✅ Legal move generation with apply/undo
    ✅ Potential field computation for all piece types (fields.jl)
    ✅ Field tension (gradient) computation
    ✅ Mobility field
    ✅ Energy function combining all field properties (energy.jl)
    ✅ Tests and field visualization demo

PHASE 2: Search ✅ COMPLETE
    ✅ Alpha-beta minimax search, negamax formulation (search.jl)
    ✅ Field-guided move ordering (TT move → promotions → MVV-LVA → centrality)
    ✅ Quiescence search (captures only, stand-pat pruning)
    ✅ Zobrist hashing + transposition table (:exact/:lower/:upper flags)
    ✅ Iterative deepening (TT seeds ordering each depth)
    ✅ Perft verification: 20 / 400 / 8902 / 197281 all pass
    ✅ Interactive terminal play (play.jl)
    → Milestone: plays legal chess, beats random players

PHASE 3: Optimization ✅ COMPLETE
    ✅ CMA-ES implementation (optimize.jl)
    ✅ Self-play tournament framework
    ✅ Stockfish (UCI) opponent integration — per-thread process pool,
       poll-based timeout, auto-recovery on session crash
    ✅ Opening book for position diversity (with rolling baseline in self-play)
    ✅ Iterative deepening in root search (seeds TT across depths)
    ✅ Soft fitness signal: tanh(eval) at move limit — no more hard-zero draws
    ✅ Automated weight optimization (run_optimize entry point)
    □ Elo estimation vs fixed Stockfish skill levels
    → Milestone: optimized weights, measurably stronger vs Stockfish skill 5

PHASE 4: Advanced Fields
    □ Time-dependent fields (how fields evolve over sequences)
    □ Piece interaction fields (coordinated influence)
    □ Pawn structure as background potential
    □ Opening field patterns (emergent from optimization)
    □ Web-based GUI for visualization and play
    → Milestone: 1000-1500 Elo, visually interesting play

═══════════════════════════════════════════════════════════════════
6. DESIGN PRINCIPLES
═══════════════════════════════════════════════════════════════════

1. NO CHESS KNOWLEDGE: The engine should never contain hardcoded
   chess heuristics (piece-square tables, opening books, endgame
   rules). Everything must emerge from the field math.

2. PURE MATH, NO ML: No neural networks, no training data, no
   gradients. Only equations, physics, and evolutionary optimization.
   The engine improves through CMA-ES, not backpropagation.

3. HEAVILY COMMENTED: Chase (the project creator) is learning Julia
   through this project. Every function should explain WHAT it does,
   WHY it does it, and HOW the math works. Err on the side of too
   many comments. Include the mathematical equations in comments.

4. CORRECT CHESS FIRST: The engine must play strictly legal chess.
   Move generation, check detection, castling, en passant, promotion
   — all must be correct. The math rides on top of correct rules.

5. PERFORMANCE VIA JULIA: The inner loops (field computation, search)
   must be fast. Use Julia's type system, avoid allocations in hot
   paths, use mutating functions (!) for in-place operations.

6. EXPLAINABLE: Every decision the engine makes should be traceable
   back to the field math. We should be able to say "it played Nf3
   because that increased field tension near Black's king by 12.4."

═══════════════════════════════════════════════════════════════════
7. CURRENT STATE AND IMMEDIATE NEXT STEPS
═══════════════════════════════════════════════════════════════════

As of now, we have:
    • A working board representation and legal move generator
    • Field computation for all piece types with blocking
    • An energy function with 5 tunable weights
    • Alpha-beta minimax search with TT, quiescence, iterative deepening
    • Interactive terminal play interface (play.jl)
    • Full CMA-ES optimizer running vs Stockfish (optimize.jl)
      - Per-thread Stockfish processes (no bottleneck on 32+ cores)
      - Poll-based timeout + auto-recovery if Stockfish hangs
      - Soft tanh fitness: draws still produce useful signal
      - Iterative deepening at root: TT compounds across depths
      - Opening book for position diversity

To run the optimizer on EC2:
    julia --threads auto src/optimize.jl 5 32 100 \
      --stockfish /usr/local/bin/stockfish \
      --sf-nodes 50000 --sf-skill 5 --n-pairs 4

IMMEDIATE NEXT STEPS:
    1. Elo estimation: play optimized weights vs Stockfish at skill 1, 3, 5, 8
       in a longer match (~200 games) and compute Elo delta vs baseline
    2. Phase 4: Time-dependent fields, pawn structure potential,
       piece coordination fields

═══════════════════════════════════════════════════════════════════
8. TECHNICAL NOTES
═══════════════════════════════════════════════════════════════════

• Julia modules: files are included via include() in FieldEngine.jl
  and use ..State / ..Fields syntax for cross-module references.
  Printf must be imported at module level, not inside functions.

• Performance: Board uses mutable struct with in-place apply/undo
  to avoid allocations during search. Fields are computed as fresh
  8×8 matrices (acceptable cost for evaluation, which happens at
  leaf nodes, not every node).

• The engine runs on macOS and Windows. Chase develops on both.
  Commands use `julia test/test_state.jl` style invocation.

• The companion project "BerserkerBot" is a separate ML-based chess
  engine in Python using PyTorch and AlphaZero methods. FieldEngine
  is the opposite approach: pure math, no ML, Julia.

═══════════════════════════════════════════════════════════════════
=#
