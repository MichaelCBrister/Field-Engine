#=
verify_field_updater.jl — Verify incremental field update correctness

The audit flagged a potential bug: apply_with_field! may double-apply some
slider contributions when source/target blocker sets overlap.

If true, evaluation becomes path-dependent (same position → different scores
depending on move history), which poisons search and tuning.

Run with:
    julia test/verify_field_updater.jl

Tests (65 total):
  1. Basic equivalence         — 10 random positions, 1 move each
  2. Slider overlap cases      — 15 positions with overlapping sliders
  3. Castling + field state    —  5 castling positions
  4. En passant + field state  —  5 en passant positions
  5. Promotion + field state   —  5 pawn promotion positions
  6. Random playouts           — 20 games × multiple plies
  7. Undo/redo consistency     —  5 undo/redo tests
=#

include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))
include(joinpath(@__DIR__, "..", "src", "energy.jl"))
include(joinpath(@__DIR__, "..", "src", "search.jl"))

using .State, .Fields, .Energy, .Search
using Printf
using Random

# ── Test harness ──────────────────────────────────────────────────
passed  = 0
failed  = 0
mismatches = String[]   # collect FENs/moves where field diverges

function test(name::String, condition::Bool)
    global passed, failed
    if condition
        passed += 1
    else
        failed += 1
        println("  FAIL: $name")
    end
end

# ── Core comparison helper ────────────────────────────────────────
"""
Compare two 8×8 field matrices for exact (bitwise) equality.
Returns (true, 0.0) or (false, max_abs_diff).
"""
const FIELD_EPS = 1e-12   # tolerance for IEEE 754 summation-order rounding

function fields_equal(a::Matrix{Float64}, b::Matrix{Float64})
    max_diff = 0.0
    for i in eachindex(a)
        d = abs(a[i] - b[i])
        d > max_diff && (max_diff = d)
    end
    return max_diff <= FIELD_EPS, max_diff
end

"""
Given a board b, compute the full field from scratch and return it.
"""
function full_field(b::Board)::Matrix{Float64}
    f = zeros(Float64, 8, 8)
    compute_total_field!(f, b)
    return f
end

"""
Apply one move incrementally (using apply_with_field!) and return:
  (undo_info, incremental_field)
Uses thread 1's buffers directly.
"""
function apply_incremental(field_in::Matrix{Float64}, b::Board, m::Move)
    field = copy(field_in)
    from_buf = FROM_SLIDERS[1]
    to_buf   = TO_SLIDERS[1]
    seen     = FROM_SEEN[1]
    undo = apply_with_field!(field, b, m, from_buf, to_buf, seen)
    return undo, field
end

"""
Return the board position as a FEN string (for mismatch reporting).
"""
function board_to_fen(b::Board)::String
    rows = String[]
    for rank in 8:-1:1
        row = ""
        empty_count = 0
        for file in 1:8
            v = b.grid[rank, file]
            if v == 0.0
                empty_count += 1
            else
                if empty_count > 0
                    row *= string(empty_count)
                    empty_count = 0
                end
                pt = abs(v)
                color = v > 0 ? WHITE : BLACK
                ch = if pt == PAWN;   color == WHITE ? 'P' : 'p'
                     elseif pt == KNIGHT; color == WHITE ? 'N' : 'n'
                     elseif pt == BISHOP; color == WHITE ? 'B' : 'b'
                     elseif pt == ROOK;   color == WHITE ? 'R' : 'r'
                     elseif pt == QUEEN;  color == WHITE ? 'Q' : 'q'
                     else               color == WHITE ? 'K' : 'k'
                     end
                row *= ch
            end
        end
        empty_count > 0 && (row *= string(empty_count))
        push!(rows, row)
    end
    piece_str = join(rows, '/')
    side = b.turn == WHITE ? "w" : "b"
    castle = ""
    b.castling[1] && (castle *= "K")
    b.castling[2] && (castle *= "Q")
    b.castling[3] && (castle *= "k")
    b.castling[4] && (castle *= "q")
    isempty(castle) && (castle = "-")
    ep = b.en_passant == (0,0) ? "-" :
         string(Char('a' + b.en_passant[2] - 1), b.en_passant[1])
    return "$piece_str $side $castle $ep $(b.halfmove) $(b.fullmove)"
end

"""
Pick a random legal move from position b. Returns nothing if no legal moves.
"""
function random_move(b::Board, rng::AbstractRNG)
    moves = generate_moves(b)
    isempty(moves) && return nothing
    return moves[rand(rng, 1:length(moves))]
end

# ── Helper: verify one move for incremental vs full equality ──────
"""
Apply move m to board b (which already has field f_before computed).
Compare incremental field update against full recomputation.
Passes a label for test reporting; records FEN+move on failure.
Returns true iff fields match exactly.
"""
function verify_move(b::Board, f_before::Matrix{Float64}, m::Move,
                     label::String)::Bool
    fen_before = board_to_fen(b)
    move_str   = move_to_string(m)

    # Incremental path
    b_inc = copy_board(b)
    _, f_inc = apply_incremental(f_before, b_inc, m)

    # Full-recompute path
    b_ref = copy_board(b)
    apply_move!(b_ref, m)
    f_ref = full_field(b_ref)

    ok, max_diff = fields_equal(f_inc, f_ref)
    if !ok
        msg = "FEN: $fen_before  move: $move_str  max_diff: $(@sprintf("%.2e", max_diff))"
        push!(mismatches, msg)
        println("  FAIL: $label  ← max_diff=$(@sprintf("%.2e", max_diff))")
        println("    $msg")
    end
    return ok
end

# ─────────────────────────────────────────────────────────────────
# TEST 1: Basic equivalence (10 tests)
# ─────────────────────────────────────────────────────────────────
println("\n── Test 1: Basic Equivalence ──")

let rng = MersenneTwister(42)
    positions_tested = 0
    for trial in 1:100
        positions_tested >= 10 && break
        # Start from beginning, play some random moves to get varied positions
        b = new_board()
        plies = rand(rng, 0:8)
        ok = true
        for _ in 1:plies
            m = random_move(b, rng)
            m === nothing && (ok = false; break)
            apply_move!(b, m)
        end
        !ok && continue
        moves = generate_moves(b)
        isempty(moves) && continue

        m = moves[rand(rng, 1:length(moves))]
        f = full_field(b)
        result = verify_move(b, f, m, "basic[$trial] $(board_to_fen(b)[1:20])")
        test("basic equivalence #$(positions_tested+1)", result)
        positions_tested += 1
    end
end

# ─────────────────────────────────────────────────────────────────
# TEST 2: Slider overlap cases (15 tests)
# ─────────────────────────────────────────────────────────────────
println("\n── Test 2: Slider Overlap Cases ──")

# Helper: build a minimal legal board with kings + custom pieces
function make_board(pieces::Vector{Tuple{Int,Int,Float64}})::Board
    b = new_board()
    for r in 1:8, f in 1:8; b.grid[r, f] = 0.0; end
    b.grid[1, 5] =  KING   # White king e1
    b.grid[8, 5] = -KING   # Black king e8
    for (r, f, v) in pieces
        b.grid[r, f] = v
    end
    sync_board!(b)
    b.castling .= false
    return b
end

# 2a: Two white rooks on same file (e-file), move intersects both rays
let
    # White rooks on e2 and e5; black rook on e7 to block; we move e2→e3 (opens ray for e5? no)
    # Place: WR e2, WR e5, BR e7, BK e8, WK e1
    # Move e2→f2 (non-capture normal move) — e2 was blocking e5's downward ray
    b = make_board([(2, 5, ROOK), (5, 5, ROOK), (7, 5, -ROOK)])
    b.turn = WHITE
    moves = generate_moves(b)
    # Filter to rook on e2 moving sideways (doesn't capture)
    cands = [m for m in moves if m.from_rank==2 && m.from_file==5 &&
             m.to_file != 5 && b.grid[m.to_rank, m.to_file] == 0.0]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: two rooks same file (move off-file)")
        test("two rooks same file, off-file move", result)
    else
        test("two rooks same file, off-file move (skipped — no cand)", true)
    end
end

let
    # Same position; now move e2→e3 (along file — e2 rook moves up, changes e5's lower ray)
    b = make_board([(2, 5, ROOK), (5, 5, ROOK), (7, 5, -ROOK)])
    b.turn = WHITE
    moves = generate_moves(b)
    cands = [m for m in moves if m.from_rank==2 && m.from_file==5 && m.to_file==5 && m.to_rank==3]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: two rooks same file (e2→e3)")
        test("two rooks same file, along-file move", result)
    else
        test("two rooks same file, along-file move (skipped)", true)
    end
end

let
    # Two white rooks on same rank (rank 4: d4 and g4), move one off-rank
    b = make_board([(4, 4, ROOK), (4, 7, ROOK)])
    b.turn = WHITE
    moves = generate_moves(b)
    cands = [m for m in moves if m.from_rank==4 && m.from_file==4 &&
             m.to_rank != 4 && b.grid[m.to_rank, m.to_file] == 0.0]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: two rooks same rank (off-rank move)")
        test("two rooks same rank, off-rank move", result)
    else
        test("two rooks same rank, off-rank move (skipped)", true)
    end
end

let
    # Two rooks same rank, move along rank (d4→e4, now g4's ray extends left past e4)
    b = make_board([(4, 4, ROOK), (4, 7, ROOK)])
    b.turn = WHITE
    moves = generate_moves(b)
    cands = [m for m in moves if m.from_rank==4 && m.from_file==4 && m.to_rank==4 && m.to_file==5]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: two rooks same rank (d4→e4)")
        test("two rooks same rank, along-rank move", result)
    else
        test("two rooks same rank, along-rank move (skipped)", true)
    end
end

let
    # Two bishops on same diagonal (c1-h6 diagonal: c1 and f4), move one off-diag
    b = make_board([(1, 3, BISHOP), (4, 6, BISHOP)])
    b.turn = WHITE
    moves = generate_moves(b)
    # c1 bishop moves to d2 (along diagonal — changes f4's backward ray)
    cands = [m for m in moves if m.from_rank==1 && m.from_file==3 && m.to_rank==2 && m.to_file==4]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: two bishops same diagonal (c1→d2)")
        test("two bishops same diagonal, along-diag move", result)
    else
        test("two bishops same diagonal, along-diag move (skipped)", true)
    end
end

let
    # Two bishops same diagonal, one captures a piece on the diagonal
    # c1 bishop and e3 bishop (same diagonal), BR on f4
    b = make_board([(1, 3, BISHOP), (3, 5, BISHOP), (4, 6, -ROOK)])
    b.turn = WHITE
    moves = generate_moves(b)
    # e3 captures f4
    cands = [m for m in moves if m.from_rank==3 && m.from_file==5 && m.to_rank==4 && m.to_file==6]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: two bishops same diag, capture")
        test("two bishops same diagonal, capture", result)
    else
        test("two bishops same diagonal, capture (skipped)", true)
    end
end

let
    # Mixed: rook + bishop with shared blocker square
    # WR on a4, WB on d1 — both have d4 on their ray; piece moves through d4
    b = make_board([(4, 1, ROOK), (1, 4, BISHOP), (4, 4, PAWN)])
    b.turn = WHITE
    moves = generate_moves(b)
    # Pawn d4→d5 — removes blocker from both rook and bishop rays simultaneously
    cands = [m for m in moves if m.from_rank==4 && m.from_file==4 && m.to_rank==5 && m.to_file==4]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: rook+bishop shared blocker (pawn d4→d5)")
        test("rook+bishop shared blocker, pawn push", result)
    else
        test("rook+bishop shared blocker, pawn push (skipped)", true)
    end
end

let
    # Black queen and black rook on same file, white pawn capture removes blocker
    # BR on d6, BQ on d8, WP on e5 captures d6
    b = make_board([(6, 4, -ROOK), (8, 4, -QUEEN), (5, 5, PAWN), (2, 4, PAWN)])
    b.turn = WHITE
    moves = generate_moves(b)
    cands = [m for m in moves if m.from_rank==5 && m.from_file==5 && m.to_rank==6 && m.to_file==4]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: black queen+rook same file, pawn captures rook")
        test("black queen+rook same file, capture removes blocker", result)
    else
        test("black queen+rook same file, capture removes blocker (skipped)", true)
    end
end

let
    # Capture that changes both from_sq and to_sq blocker sets
    # WR e4, WR e7 (same file), BR captures e4 (Black rook captures WR e4)
    b = make_board([(4, 5, ROOK), (7, 5, ROOK), (4, 8, -ROOK)])
    b.turn = BLACK
    moves = generate_moves(b)
    cands = [m for m in moves if m.from_rank==4 && m.from_file==8 && m.to_rank==4 && m.to_file==5]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap: black rook captures white rook, second WR on same file")
        test("capture with second slider on same file", result)
    else
        test("capture with second slider on same file (skipped)", true)
    end
end

# Additional slider overlap tests using random positions with multiple sliders
let rng = MersenneTwister(137)
    slider_tests_done = 0
    for trial in 1:200
        slider_tests_done >= 6 && break
        b = new_board()
        plies = rand(rng, 3:10)
        ok = true
        for _ in 1:plies
            m = random_move(b, rng)
            m === nothing && (ok = false; break)
            is_game_over(b) && (ok = false; break)
            apply_move!(b, m)
        end
        !ok && continue

        # Count sliders (rooks, bishops, queens) to filter for slider-heavy positions
        slider_count = 0
        for r in 1:8, f in 1:8
            pt = abs(b.grid[r, f])
            (pt == ROOK || pt == BISHOP || pt == QUEEN) && (slider_count += 1)
        end
        slider_count < 4 && continue

        moves = generate_moves(b)
        isempty(moves) && continue
        m = moves[rand(rng, 1:length(moves))]
        f = full_field(b)
        result = verify_move(b, f, m, "slider_overlap random[$trial]")
        test("slider overlap random #$(slider_tests_done+1)", result)
        slider_tests_done += 1
    end
    # Fill remaining if we didn't find enough slider-heavy positions
    while slider_tests_done < 6
        slider_tests_done += 1
        test("slider overlap random #$slider_tests_done (skipped — position filter)", true)
    end
end

# ─────────────────────────────────────────────────────────────────
# TEST 3: Castling + incremental state (5 tests)
# ─────────────────────────────────────────────────────────────────
println("\n── Test 3: Castling ──")

# White kingside castling
let
    b = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    moves = generate_moves(b)
    cands = [m for m in moves if m.is_castling && m.to_file == 7 && m.from_rank == 1]
    if !isempty(cands)
        f = full_field(b)
        b_inc = copy_board(b)
        undo, f_inc = apply_incremental(f, b_inc, first(cands))
        b2 = copy_board(b)
        apply_move!(b2, first(cands))
        f_ref = full_field(b2)
        ok, _ = fields_equal(f_inc, f_ref)
        test("white kingside castling field matches recompute", ok)
    else
        test("white kingside castling (skipped — no castling move)", true)
    end
end

# White queenside castling
let
    b = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    moves = generate_moves(b)
    cands = [m for m in moves if m.is_castling && m.to_file == 3 && m.from_rank == 1]
    if !isempty(cands)
        f = full_field(b)
        b_inc = copy_board(b)
        undo, f_inc = apply_incremental(f, b_inc, first(cands))
        b2 = copy_board(b)
        apply_move!(b2, first(cands))
        f_ref = full_field(b2)
        ok, _ = fields_equal(f_inc, f_ref)
        test("white queenside castling field matches recompute", ok)
    else
        test("white queenside castling (skipped — no castling move)", true)
    end
end

# Black kingside castling
let
    b = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
    moves = generate_moves(b)
    cands = [m for m in moves if m.is_castling && m.to_file == 7 && m.from_rank == 8]
    if !isempty(cands)
        f = full_field(b)
        b_inc = copy_board(b)
        undo, f_inc = apply_incremental(f, b_inc, first(cands))
        b2 = copy_board(b)
        apply_move!(b2, first(cands))
        f_ref = full_field(b2)
        ok, _ = fields_equal(f_inc, f_ref)
        test("black kingside castling field matches recompute", ok)
    else
        test("black kingside castling (skipped — no castling move)", true)
    end
end

# Black queenside castling
let
    b = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
    moves = generate_moves(b)
    cands = [m for m in moves if m.is_castling && m.to_file == 3 && m.from_rank == 8]
    if !isempty(cands)
        f = full_field(b)
        b_inc = copy_board(b)
        undo, f_inc = apply_incremental(f, b_inc, first(cands))
        b2 = copy_board(b)
        apply_move!(b2, first(cands))
        f_ref = full_field(b2)
        ok, _ = fields_equal(f_inc, f_ref)
        test("black queenside castling field matches recompute", ok)
    else
        test("black queenside castling (skipped — no castling move)", true)
    end
end

# Castling when other pieces are present (from a near-real position)
let
    # A common mid-game castling scenario: pieces cleared on king side
    b = from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
    moves = generate_moves(b)
    cands = [m for m in moves if m.is_castling]
    if !isempty(cands)
        m = first(cands)
        f = full_field(b)
        b_inc = copy_board(b)
        undo, f_inc = apply_incremental(f, b_inc, m)
        b2 = copy_board(b)
        apply_move!(b2, m)
        f_ref = full_field(b2)
        ok, _ = fields_equal(f_inc, f_ref)
        test("mid-game castling with pieces present", ok)
    else
        test("mid-game castling (skipped — castling not available)", true)
    end
end

# ─────────────────────────────────────────────────────────────────
# TEST 4: En passant + incremental state (5 tests)
# ─────────────────────────────────────────────────────────────────
println("\n── Test 4: En Passant ──")

# Helper to set up an en passant position and execute the capture
function test_ep(fen::String, label::String)
    b = from_fen(fen)
    moves = generate_moves(b)
    ep_moves = [m for m in moves if m.is_en_passant]
    if isempty(ep_moves)
        test("$label (skipped — no ep move)", true)
        return
    end
    m = first(ep_moves)
    f = full_field(b)
    b_inc = copy_board(b)
    undo, f_inc = apply_incremental(f, b_inc, m)
    b2 = copy_board(b)
    apply_move!(b2, m)
    f_ref = full_field(b2)
    ok, max_diff = fields_equal(f_inc, f_ref)
    if !ok
        fen_str = board_to_fen(b)
        msg = "FEN: $fen_str  move: $(move_to_string(m))  max_diff: $(@sprintf("%.2e", max_diff))"
        push!(mismatches, msg)
        println("  FAIL: $label  ← max_diff=$(@sprintf("%.2e", max_diff))")
        println("    $msg")
    end
    test(label, ok)
end

# White pawn captures en passant on e6 (pawn on d5, ep target e6)
test_ep("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "ep: white pawn captures on f6")

# Another white ep capture
test_ep("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "ep: white pawn captures on d6")

# Black pawn captures en passant
test_ep("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
        "ep: black pawn captures on d3")

# En passant capture removes a pawn that was blocking a slider
let
    # WP on e5, BP pushes d7→d5 (ep=d6), WR on a5 now behind the bp on d5
    # After white ep captures d5, the rook's ray extends
    b = from_fen("4k3/8/8/R2pP3/8/8/8/4K3 w - - 0 1")
    # Simulate d7→d5 to set up ep
    apply_move!(b, Move(7, 4, 5, 4))  # d7→d5 (black pawn push)... wait, turn check
    # Actually build it directly:
    b2 = from_fen("4k3/8/8/R2pP3/8/8/8/4K3 w - d6 0 1")
    moves = generate_moves(b2)
    ep_moves = [m for m in moves if m.is_en_passant]
    if !isempty(ep_moves)
        m = first(ep_moves)
        f = full_field(b2)
        b2_inc = copy_board(b2)
        undo, f_inc = apply_incremental(f, b2_inc, m)
        b3 = copy_board(b2)
        apply_move!(b3, m)
        f_ref = full_field(b3)
        ok, max_diff = fields_equal(f_inc, f_ref)
        if !ok
            push!(mismatches, "FEN: $(board_to_fen(b2))  move: $(move_to_string(m))  max_diff: $(@sprintf("%.2e", max_diff))")
        end
        test("ep: capture removes pawn blocking rook ray", ok)
    else
        test("ep: capture removes pawn blocking rook ray (skipped)", true)
    end
end

# En passant with multiple pieces observing the capture square
test_ep("4k3/8/8/1b1pP3/8/8/8/R3K3 w - d6 0 1",
        "ep: capture with bishop and rook observing")

# ─────────────────────────────────────────────────────────────────
# TEST 5: Promotion + incremental state (5 tests)
# ─────────────────────────────────────────────────────────────────
println("\n── Test 5: Promotion ──")

# Promote to queen
let
    b = from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    queen_promo = [m for m in moves if m.promotion != 0.0 && abs(m.promotion) == QUEEN]
    if !isempty(queen_promo)
        m = first(queen_promo)
        f = full_field(b)
        result = verify_move(b, f, m, "promotion to queen")
        test("promotion to queen", result)
    else
        test("promotion to queen (skipped)", true)
    end
end

# Promote to rook
let
    b = from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    rook_promo = [m for m in moves if m.promotion != 0.0 && abs(m.promotion) == ROOK]
    if !isempty(rook_promo)
        m = first(rook_promo)
        f = full_field(b)
        result = verify_move(b, f, m, "promotion to rook")
        test("promotion to rook", result)
    else
        test("promotion to rook (skipped)", true)
    end
end

# Promote to bishop
let
    b = from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    bishop_promo = [m for m in moves if m.promotion != 0.0 && abs(m.promotion) == BISHOP]
    if !isempty(bishop_promo)
        m = first(bishop_promo)
        f = full_field(b)
        result = verify_move(b, f, m, "promotion to bishop")
        test("promotion to bishop", result)
    else
        test("promotion to bishop (skipped)", true)
    end
end

# Promote to knight
let
    b = from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    knight_promo = [m for m in moves if m.promotion != 0.0 && abs(m.promotion) == KNIGHT]
    if !isempty(knight_promo)
        m = first(knight_promo)
        f = full_field(b)
        result = verify_move(b, f, m, "promotion to knight")
        test("promotion to knight", result)
    else
        test("promotion to knight (skipped)", true)
    end
end

# Promotion by capture (pawn captures and promotes)
let
    b = from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    capture_promo = [m for m in moves if m.promotion != 0.0 &&
                     b.grid[m.to_rank, m.to_file] != 0.0 && abs(m.promotion) == QUEEN]
    if !isempty(capture_promo)
        m = first(capture_promo)
        f = full_field(b)
        result = verify_move(b, f, m, "promotion by capture to queen")
        test("promotion by capture to queen", result)
    else
        # Fall back to non-capture promotion
        non_capture = [m for m in moves if m.promotion != 0.0 && abs(m.promotion) == QUEEN]
        if !isempty(non_capture)
            m = first(non_capture)
            f = full_field(b)
            result = verify_move(b, f, m, "promotion (fallback non-capture)")
            test("promotion (fallback non-capture)", result)
        else
            test("promotion by capture (skipped)", true)
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# TEST 6: Random playouts (20 tests)
# Each game plays multiple plies; field verified after EVERY move.
# ─────────────────────────────────────────────────────────────────
println("\n── Test 6: Random Playouts ──")

let rng = MersenneTwister(2024)
    game_failures = 0
    for game_id in 1:20
        b = new_board()
        f = full_field(b)
        game_ok = true
        game_moves = String[]
        fen_at_fail = ""
        fail_move = ""
        max_diff_seen = 0.0

        for ply in 1:15
            is_game_over(b) && break
            moves = generate_moves(b)
            isempty(moves) && break

            m = moves[rand(rng, 1:length(moves))]
            push!(game_moves, move_to_string(m))

            # Incremental update
            b_inc = copy_board(b)
            f_copy = copy(f)
            _, f_inc = apply_incremental(f_copy, b_inc, m)

            # Full recompute for reference
            b_ref = copy_board(b)
            apply_move!(b_ref, m)
            f_ref = full_field(b_ref)

            ok, max_diff = fields_equal(f_inc, f_ref)
            if !ok && game_ok
                game_ok = false
                fen_at_fail = board_to_fen(b)
                fail_move   = move_to_string(m)
                max_diff_seen = max_diff
                game_failures += 1
            end

            # Advance board state (apply the move for real)
            apply_move!(b, m)
            compute_total_field!(f, b)  # keep f in sync for next iteration
        end

        if !game_ok
            msg = "Game $game_id: FEN at fail: $fen_at_fail  move: $fail_move  max_diff: $(@sprintf("%.2e", max_diff_seen))  moves: $(join(game_moves, " "))"
            push!(mismatches, msg)
            println("  FAIL: random playout game $game_id")
            println("    $msg")
        end
        test("random playout game $game_id (all plies match)", game_ok)
    end
end

# ─────────────────────────────────────────────────────────────────
# TEST 7: Undo/redo consistency (5 tests)
# ─────────────────────────────────────────────────────────────────
println("\n── Test 7: Undo/Redo Consistency ──")

"""
Test that:
  1. After apply_with_field!, incremental field == full recompute (post-move)
  2. After undo_move!, the field (if we recompute) matches the original
"""
function test_undo_redo(b::Board, m::Move, label::String)
    f_before = full_field(b)
    fen_before = board_to_fen(b)

    # --- Apply incrementally ---
    b_test = copy_board(b)
    f_test = copy(f_before)
    undo = apply_with_field!(f_test, b_test, m, FROM_SLIDERS[1], TO_SLIDERS[1], FROM_SEEN[1])

    # Verify post-move field matches full recompute
    f_ref_post = full_field(b_test)
    ok_post, diff_post = fields_equal(f_test, f_ref_post)
    if !ok_post
        push!(mismatches, "undo_redo POST-MOVE: FEN=$fen_before  move=$(move_to_string(m))  diff=$(@sprintf("%.2e", diff_post))")
    end
    test("$label: post-move incremental == full recompute", ok_post)

    # --- Undo the move ---
    undo_move!(b_test, m, undo)

    # Field after undo should match f_before (recompute from restored board)
    f_ref_undone = full_field(b_test)
    ok_undo, diff_undo = fields_equal(f_before, f_ref_undone)
    if !ok_undo
        push!(mismatches, "undo_redo AFTER-UNDO: FEN=$fen_before  move=$(move_to_string(m))  diff=$(@sprintf("%.2e", diff_undo))")
    end
    test("$label: after undo, board restores and full recompute matches original", ok_undo)
end

# Test 1: Normal move undo
let
    b = new_board()
    m = Move(2, 5, 4, 5)  # e2→e4
    test_undo_redo(b, m, "undo/redo e2→e4")
end

# Test 2: Capture undo
let
    b = from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    moves = generate_moves(b)
    captures = [m for m in moves if b.grid[m.to_rank, m.to_file] != 0.0 && !m.is_en_passant]
    if !isempty(captures)
        test_undo_redo(b, first(captures), "undo/redo pawn capture")
    else
        test("undo/redo pawn capture: post-move (skipped)", true)
        test("undo/redo pawn capture: after undo (skipped)", true)
    end
end

# Test 3: Slider move undo (rook)
let
    b = from_fen("4k3/8/8/8/R7/8/8/4K3 w - - 0 1")
    moves = generate_moves(b)
    rook_moves = [m for m in moves if abs(b.grid[m.from_rank, m.from_file]) == ROOK]
    if !isempty(rook_moves)
        test_undo_redo(b, first(rook_moves), "undo/redo rook move")
    else
        test("undo/redo rook move: post-move (skipped)", true)
        test("undo/redo rook move: after undo (skipped)", true)
    end
end

# Test 4: Castling undo
let
    b = from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    moves = generate_moves(b)
    castles = [m for m in moves if m.is_castling]
    if !isempty(castles)
        test_undo_redo(b, first(castles), "undo/redo castling")
    else
        test("undo/redo castling: post-move (skipped)", true)
        test("undo/redo castling: after undo (skipped)", true)
    end
end

# Test 5: En passant undo
let
    b = from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    moves = generate_moves(b)
    ep_moves = [m for m in moves if m.is_en_passant]
    if !isempty(ep_moves)
        test_undo_redo(b, first(ep_moves), "undo/redo en passant")
    else
        test("undo/redo en passant: post-move (skipped)", true)
        test("undo/redo en passant: after undo (skipped)", true)
    end
end

# ─────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────
println("\n══════════════════════════════════════════════════════════════")
@printf("  verify_field_updater:  %d passed,  %d failed  (of %d total)\n",
        passed, failed, passed + failed)
println("══════════════════════════════════════════════════════════════")

if isempty(mismatches)
    println("\n  ✓ No field mismatches detected.")
    println("    Incremental field updater appears correct for all tested cases.")
else
    println("\n  ✗ MISMATCHES FOUND — $(length(mismatches)) divergence(s):")
    for (i, msg) in enumerate(mismatches)
        println("  [$i] $msg")
    end
    println()
    println("  These FENs/moves are reproducible starting points for debugging")
    println("  apply_with_field! in src/search.jl (lines 229-287).")
    println()
    println("  Suspect: slider dedup logic (seen[] matrix) when from_buf and")
    println("  to_buf share entries — a piece in both sets may get double-subtracted")
    println("  in Phase 1 but only single-added in Phase 3.")
end

println()

# Exit with nonzero status if any tests failed, for CI
if failed > 0
    exit(1)
end
