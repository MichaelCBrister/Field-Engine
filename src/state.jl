#=
state.jl — The Board as a Mathematical Object

THE FUNDAMENTAL IDEA:
    A chess position is an 8×8 matrix of signed real numbers.

    S[rank, file] > 0  →  White piece (magnitude = charge)
    S[rank, file] < 0  →  Black piece (magnitude = charge)
    S[rank, file] = 0  →  Empty square

    That's it. No piece objects, no bitboards, no special types.
    Just a matrix of numbers that we can do math on.

WHY SIGNED NUMBERS:
    The sign carries team information. When we compute fields later,
    we can simply sum over all pieces — positive contributions from
    friendly pieces and negative from enemy pieces automatically
    cancel and reinforce. The linear algebra just works.

PIECE CHARGES (magnitudes):
    Pawn   = 1.0    — weakest field emitter
    Knight = 3.0    — medium range, unusual geometry
    Bishop = 3.25   — slightly stronger than knight (long diagonals)
    Rook   = 5.0    — strong along ranks/files
    Queen  = 9.0    — strongest mobile piece
    King   = 100.0  — massive charge, the gravitational center

    The king's charge is intentionally absurd (100 vs 9 for queen).
    This means the king dominates every field calculation. Positions
    where the enemy king is exposed will have extreme field values,
    naturally causing the engine to attack it — or protect its own.
    No "king safety" heuristic needed. The math handles it.

WHY THESE SPECIFIC VALUES:
    Pawn=1 is arbitrary — it sets the scale. Knight=3, Bishop=3.25,
    Rook=5, Queen=9 are close to the classical chess piece values
    that centuries of play have validated. But for us, they're not
    "values" — they're field charges. A queen emits 9× the field
    of a pawn, which is roughly how much more influence it has on
    the board. The 0.25 bonus for bishop over knight reflects that
    bishops project along long diagonals (more field reach).

COORDINATE SYSTEM:
    We use (rank, file) where:
    - rank 1 = White's back rank (row 1 of the matrix)
    - rank 8 = Black's back rank (row 8 of the matrix)
    - file 1 = a-file (column 1), file 8 = h-file (column 8)

    So S[1, 5] = 100.0 means White's king is on e1 (starting position).
    And S[8, 5] = -100.0 means Black's king is on e8.
=#

module State

using Random: MersenneTwister, rand

export Board, new_board, from_fen, print_board, copy_board
export WHITE, BLACK, EMPTY
export PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
export piece_at, is_empty, is_color, piece_type, piece_color
export Move, move_to_string, apply_move!, undo_move!, generate_moves, generate_moves!
export find_king, is_in_check, is_checkmate, is_stalemate, is_game_over, game_result, is_repetition
export sync_board!
export ZOBRIST_SIDE, ZOBRIST_EP  # needed by optimize.jl for null-move hash updates

# ── Constants ──────────────────────────────────────────────────

# Colors are just signs: +1 for White, -1 for Black
const WHITE =  1
const BLACK = -1
const EMPTY =  0

# Piece charges (magnitudes)
const PAWN   = 1.0
const KNIGHT = 3.0
const BISHOP = 3.25
const ROOK   = 5.0
const QUEEN  = 9.0
const KING   = 100.0

#= ── Zobrist Hashing ────────────────────────────────────────────

   Zobrist hashing assigns each (square, piece_type) pair a random
   UInt64. The board hash is the XOR of all active piece keys plus
   flags for side to move, castling rights, and en passant file.

   XOR is its own inverse: h ⊻ k ⊻ k == h. This means apply_move!
   can update the hash incrementally by XORing out old features and
   XORing in new ones. undo_move! restores it in O(1) from UndoInfo
   without any recomputation.

   The RNG is seeded for reproducibility — the same position always
   produces the same hash, which matters for checkpoint compatibility.

   Convention: ZOBRIST_SIDE is XORed in whenever BLACK is to move.
   White-to-move positions have the raw piece XOR; Black-to-move
   positions also include ZOBRIST_SIDE. Every apply_move! XORs
   ZOBRIST_SIDE once, flipping the turn flag in the hash.
=#

const _Z_RNG         = MersenneTwister(0xDEADBEEF_CAFE_F00D)
const ZOBRIST_PIECES  = rand(_Z_RNG, UInt64, 8, 8, 12)  # [rank, file, piece_idx]
const ZOBRIST_SIDE    = rand(_Z_RNG, UInt64)              # XOR when Black to move
const ZOBRIST_CASTLE  = rand(_Z_RNG, UInt64, 4)           # one per castling right [WK,WQ,BK,BQ]
const ZOBRIST_EP      = rand(_Z_RNG, UInt64, 8)           # one per ep file (1-8)

# Maps a signed piece charge Q to a Zobrist table index in 1:12.
# White: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
# Black: Pawn=7, Knight=8, Bishop=9, Rook=10, Queen=11, King=12
@inline function piece_zobrist_idx(Q::Float64)::Int
    off = Q > 0 ? 0 : 6
    pt  = abs(Q)
    pt == PAWN   && return 1 + off
    pt == KNIGHT && return 2 + off
    pt == BISHOP && return 3 + off
    pt == ROOK   && return 4 + off
    pt == QUEEN  && return 5 + off
    return 6 + off  # KING
end

#= ── The Board ──────────────────────────────────────────────────

    The Board struct holds:
    1. grid:  8×8 matrix of Float64 (the mathematical state)
    2. turn:  whose move it is (+1 or -1)
    3. castling: 4 bools (WK, WQ, BK, BQ) — purely for legal move generation
    4. en_passant: target square (rank, file) or (0,0) if none
    5. halfmove: moves since last capture/pawn push (for 50-move rule)
    6. fullmove: total full moves (increments after Black moves)

    The castling and en_passant fields are the one concession to
    chess-specific rules. The math doesn't need them, but legal
    move generation does. Chess rules are non-negotiable — the
    engine must play legal chess, even if it thinks in fields.
=#

mutable struct Board
    grid::Matrix{Float64}       # 8×8 state matrix
    turn::Int                   # WHITE (+1) or BLACK (-1)
    castling::Vector{Bool}      # [WK, WQ, BK, BQ]
    en_passant::Tuple{Int,Int}  # (rank, file) or (0,0)
    halfmove::Int
    fullmove::Int
    # ── Incrementally maintained ─────────────────────────────────
    # These eliminate expensive scans at every node:
    #   material:   replaces sum(grid) — O(64) → O(1)
    #   white_king / black_king: replaces find_king O(64) scan.
    #     find_king is called inside generate_moves! for EVERY
    #     pseudo-legal move's legality check — storing it saves
    #     millions of scans per game.
    #   hash:       Zobrist hash for transposition table lookup.
    material::Float64            # sum(grid): net signed charge (White − Black)
    white_king::Tuple{Int,Int}   # current position of White king
    black_king::Tuple{Int,Int}   # current position of Black king
    hash::UInt64                 # Zobrist hash of current position
    # ── Position history for repetition detection ─────────────────
    # Stores the Zobrist hash of every position reached in the current
    # game/search line. apply_move! pushes the pre-move hash;
    # undo_move! pops it. This lets is_repetition() detect draws by
    # threefold repetition in O(halfmove) time — we only need to scan
    # back to the last irreversible move (capture or pawn push) since
    # positions before that point can never repeat.
    history::Vector{UInt64}
end

function copy_board(b::Board)::Board
    Board(
        copy(b.grid),
        b.turn,
        copy(b.castling),
        b.en_passant,
        b.halfmove,
        b.fullmove,
        b.material,
        b.white_king,
        b.black_king,
        b.hash,
        copy(b.history)
    )
end

#= ── Initial Position ───────────────────────────────────────────

    The starting position as a matrix. Read this bottom-to-top
    to match how a chessboard looks:

    Row 8: -R  -N  -B  -Q  -K  -B  -N  -R   (Black pieces)
    Row 7: -P  -P  -P  -P  -P  -P  -P  -P   (Black pawns)
    Row 6:  .   .   .   .   .   .   .   .    (empty)
    Row 5:  .   .   .   .   .   .   .   .    (empty)
    Row 4:  .   .   .   .   .   .   .   .    (empty)
    Row 3:  .   .   .   .   .   .   .   .    (empty)
    Row 2: +P  +P  +P  +P  +P  +P  +P  +P   (White pawns)
    Row 1: +R  +N  +B  +Q  +K  +B  +N  +R   (White pieces)

    Positive values = White. Negative = Black.
=#

function new_board()::Board
    grid = zeros(Float64, 8, 8)

    # White pieces (rank 1) — positive values
    grid[1, 1] = ROOK;   grid[1, 2] = KNIGHT; grid[1, 3] = BISHOP
    grid[1, 4] = QUEEN;  grid[1, 5] = KING;   grid[1, 6] = BISHOP
    grid[1, 7] = KNIGHT; grid[1, 8] = ROOK

    # White pawns (rank 2)
    for f in 1:8
        grid[2, f] = PAWN
    end

    # Black pawns (rank 7) — negative values
    for f in 1:8
        grid[7, f] = -PAWN
    end

    # Black pieces (rank 8)
    grid[8, 1] = -ROOK;   grid[8, 2] = -KNIGHT; grid[8, 3] = -BISHOP
    grid[8, 4] = -QUEEN;  grid[8, 5] = -KING;   grid[8, 6] = -BISHOP
    grid[8, 7] = -KNIGHT; grid[8, 8] = -ROOK

    # Compute initial Zobrist hash.
    # Convention: White to move → ZOBRIST_SIDE not included.
    # All 4 castling rights active → XOR all ZOBRIST_CASTLE entries.
    # No en passant → ZOBRIST_EP not included.
    h = UInt64(0)
    for i in 1:4; h ⊻= ZOBRIST_CASTLE[i]; end
    for r in 1:8, f in 1:8
        Q = grid[r, f]
        Q != 0.0 && (h ⊻= ZOBRIST_PIECES[r, f, piece_zobrist_idx(Q)])
    end

    # material = sum(grid) = 0 at start (symmetric position, signs cancel).
    hist = UInt64[]
    sizehint!(hist, 512)   # typical game + search depth, avoids realloc
    Board(grid, WHITE, [true, true, true, true], (0, 0), 0, 1,
          0.0,     # material
          (1, 5),  # white_king at e1
          (8, 5),  # black_king at e8
          h,
          hist)
end

# ── Derived-state synchronization ──────────────────────────────
# Board stores incremental caches (material, king coordinates, hash).
# If callers edit b.grid directly to construct test/custom positions,
# these caches can become stale. sync_board! rebuilds them from scratch.
function _scan_king(b::Board, color::Int)::Tuple{Int,Int}
    target = color == WHITE ? KING : -KING
    for r in 1:8, f in 1:8
        b.grid[r, f] == target && return (r, f)
    end
    side = color == WHITE ? "White" : "Black"
    error("Invalid board: missing $side king")
end

function _recompute_hash(b::Board)::UInt64
    h = UInt64(0)
    b.turn == BLACK && (h ⊻= ZOBRIST_SIDE)
    for i in 1:4
        b.castling[i] && (h ⊻= ZOBRIST_CASTLE[i])
    end
    b.en_passant != (0, 0) && (h ⊻= ZOBRIST_EP[b.en_passant[2]])
    for r in 1:8, f in 1:8
        Q = b.grid[r, f]
        Q != 0.0 && (h ⊻= ZOBRIST_PIECES[r, f, piece_zobrist_idx(Q)])
    end
    return h
end

function sync_board!(b::Board)::Board
    b.material   = sum(b.grid)
    b.white_king = _scan_king(b, WHITE)
    b.black_king = _scan_king(b, BLACK)
    b.hash       = _recompute_hash(b)
    return b
end

# ── Square Queries ─────────────────────────────────────────────
# These helper functions let us ask questions about squares
# without caring about the underlying representation.

"""Is the square within the 8×8 grid?"""
in_bounds(r, f) = 1 ≤ r ≤ 8 && 1 ≤ f ≤ 8

"""What's on this square? Returns the raw signed float."""
piece_at(b::Board, r, f) = b.grid[r, f]

"""Is this square empty?"""
is_empty(b::Board, r, f) = b.grid[r, f] == 0.0

"""Does this square have a piece of the given color?"""
is_color(b::Board, r, f, color) = sign(b.grid[r, f]) == color

"""What type of piece is here? Returns the magnitude (PAWN, KNIGHT, etc)."""
piece_type(b::Board, r, f) = abs(b.grid[r, f])

"""What color is the piece here? Returns WHITE, BLACK, or EMPTY."""
function piece_color(b::Board, r, f)
    v = b.grid[r, f]
    v > 0 ? WHITE : v < 0 ? BLACK : EMPTY
end

# ── Pretty Printing ────────────────────────────────────────────

const PIECE_CHARS = Dict(
    (WHITE, PAWN) => '♙', (WHITE, KNIGHT) => '♘', (WHITE, BISHOP) => '♗',
    (WHITE, ROOK) => '♖', (WHITE, QUEEN) => '♕', (WHITE, KING) => '♔',
    (BLACK, PAWN) => '♟', (BLACK, KNIGHT) => '♞', (BLACK, BISHOP) => '♝',
    (BLACK, ROOK) => '♜', (BLACK, QUEEN) => '♛', (BLACK, KING) => '♚',
)

function print_board(b::Board)
    println("\n    a   b   c   d   e   f   g   h")
    println("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
    for rank in 8:-1:1
        print("$rank │")
        for file in 1:8
            v = b.grid[rank, file]
            if v == 0.0
                print("   │")
            else
                color = v > 0 ? WHITE : BLACK
                pt = abs(v)
                ch = get(PIECE_CHARS, (color, pt), '?')
                print(" $ch │")
            end
        end
        println(" $rank")
        if rank > 1
            println("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
        end
    end
    println("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    println("    a   b   c   d   e   f   g   h")
    side = b.turn == WHITE ? "White" : "Black"
    println("  $side to move\n")
end

#= ── Moves ──────────────────────────────────────────────────────

    A Move is just: "take the value at (r1,f1) and put it at (r2,f2)."

    In matrix terms, a move is a sparse permutation operator on the
    state matrix — it zeros one cell and writes to another. Captures
    happen when the target cell is nonzero (the old value is overwritten,
    meaning that "charge" is removed from the system).

    Special moves (castling, en passant, promotion) need extra handling,
    but mathematically they're still just matrix element operations.
=#

struct Move
    from_rank::Int
    from_file::Int
    to_rank::Int
    to_file::Int
    promotion::Float64      # 0.0 if not a promotion, else the new piece charge
    is_castling::Bool
    is_en_passant::Bool
end

# Convenience constructor for normal moves
Move(fr, ff, tr, tf) = Move(fr, ff, tr, tf, 0.0, false, false)

function move_to_string(m::Move)
    from_str = string(Char('a' + m.from_file - 1), m.from_rank)
    to_str = string(Char('a' + m.to_file - 1), m.to_rank)
    promo = ""
    if m.promotion != 0.0
        pt = abs(m.promotion)
        promo = pt == QUEEN ? "q" : pt == ROOK ? "r" : pt == BISHOP ? "b" : "n"
    end
    return from_str * to_str * promo
end

#= ── Apply and Undo Moves ──────────────────────────────────────

    apply_move! modifies the board in place (fast, no allocation).
    It returns an "undo info" tuple so we can reverse the move during
    search without copying the entire board.

    This is critical for search performance: we apply a move,
    evaluate, then undo — thousands of times per second.
=#

struct UndoInfo
    captured::Float64              # What was at the target square (0 if empty)
    prev_castling::NTuple{4,Bool}  # Stack-allocated — no heap allocation per move
    prev_en_passant::Tuple{Int,Int}
    prev_halfmove::Int
    # Incrementally maintained fields — restored directly, no recomputation:
    prev_hash::UInt64
    prev_material::Float64
    prev_white_king::Tuple{Int,Int}
    prev_black_king::Tuple{Int,Int}
end

function apply_move!(b::Board, m::Move)::UndoInfo
    fr, ff = m.from_rank, m.from_file
    tr, tf = m.to_rank, m.to_file

    captured = b.grid[tr, tf]
    undo = UndoInfo(captured,
                    (b.castling[1], b.castling[2], b.castling[3], b.castling[4]),
                    b.en_passant, b.halfmove,
                    b.hash, b.material, b.white_king, b.black_king)

    # Record the current position hash for repetition detection.
    # This must happen before any state modification so undo_move! can
    # simply pop it.  push!/pop! on a pre-sized Vector is O(1).
    push!(b.history, b.hash)

    piece = b.grid[fr, ff]
    color = sign(piece)
    pt    = abs(piece)

    # ── Zobrist: XOR out old state ──────────────────────────────
    h = b.hash
    h ⊻= ZOBRIST_SIDE                                              # flip side to move
    b.en_passant != (0,0) && (h ⊻= ZOBRIST_EP[b.en_passant[2]])  # remove old ep
    for i in 1:4; b.castling[i] && (h ⊻= ZOBRIST_CASTLE[i]); end # remove old castling
    h ⊻= ZOBRIST_PIECES[fr, ff, piece_zobrist_idx(piece)]         # piece leaves from_sq
    captured != 0.0 && (h ⊻= ZOBRIST_PIECES[tr, tf, piece_zobrist_idx(captured)])

    # ── Material: remove captured piece ─────────────────────────
    b.material -= captured

    # ── Move the piece ──────────────────────────────────────────
    b.grid[fr, ff] = 0.0
    if m.promotion != 0.0
        new_piece = color * abs(m.promotion)
        b.grid[tr, tf] = new_piece
        b.material += new_piece - piece          # pawn → promoted piece (net delta)
        h ⊻= ZOBRIST_PIECES[tr, tf, piece_zobrist_idx(new_piece)]
    else
        b.grid[tr, tf] = piece
        h ⊻= ZOBRIST_PIECES[tr, tf, piece_zobrist_idx(piece)]
        if pt == KING
            if color == WHITE; b.white_king = (tr, tf)
            else;              b.black_king = (tr, tf)
            end
        end
    end

    # ── En passant capture: remove the captured pawn ─────────────
    if m.is_en_passant
        ep_pawn = b.grid[fr, tf]
        b.grid[fr, tf] = 0.0
        b.material    -= ep_pawn
        h ⊻= ZOBRIST_PIECES[fr, tf, piece_zobrist_idx(ep_pawn)]
    end

    # ── Castling: also move the rook ─────────────────────────────
    if m.is_castling
        if tf == 7  # Kingside
            rook = b.grid[fr, 8]
            b.grid[fr, 8] = 0.0
            b.grid[fr, 6] = rook
            h ⊻= ZOBRIST_PIECES[fr, 8, piece_zobrist_idx(rook)]
            h ⊻= ZOBRIST_PIECES[fr, 6, piece_zobrist_idx(rook)]
        else  # Queenside (to_file == 3)
            rook = b.grid[fr, 1]
            b.grid[fr, 1] = 0.0
            b.grid[fr, 4] = rook
            h ⊻= ZOBRIST_PIECES[fr, 1, piece_zobrist_idx(rook)]
            h ⊻= ZOBRIST_PIECES[fr, 4, piece_zobrist_idx(rook)]
        end
    end

    # ── Update en passant target ─────────────────────────────────
    b.en_passant = (0, 0)
    if pt == PAWN && abs(tr - fr) == 2
        b.en_passant = (div(fr + tr, 2), ff)
        h ⊻= ZOBRIST_EP[ff]
    end

    # ── Update castling rights ───────────────────────────────────
    if pt == KING
        if color == WHITE; b.castling[1] = false; b.castling[2] = false
        else;              b.castling[3] = false; b.castling[4] = false
        end
    end
    fr == 1 && ff == 1 && (b.castling[2] = false)
    fr == 1 && ff == 8 && (b.castling[1] = false)
    fr == 8 && ff == 1 && (b.castling[4] = false)
    fr == 8 && ff == 8 && (b.castling[3] = false)
    tr == 1 && tf == 1 && (b.castling[2] = false)
    tr == 1 && tf == 8 && (b.castling[1] = false)
    tr == 8 && tf == 1 && (b.castling[4] = false)
    tr == 8 && tf == 8 && (b.castling[3] = false)
    for i in 1:4; b.castling[i] && (h ⊻= ZOBRIST_CASTLE[i]); end  # XOR in new castling

    # ── Update halfmove clock ─────────────────────────────────────
    b.halfmove = (pt == PAWN || captured != 0.0) ? 0 : b.halfmove + 1

    # ── Update fullmove counter ───────────────────────────────────
    color == BLACK && (b.fullmove += 1)

    # ── Switch turn and store hash ────────────────────────────────
    b.turn = -b.turn
    b.hash = h

    return undo
end

function undo_move!(b::Board, m::Move, undo::UndoInfo)
    b.turn = -b.turn
    color  = b.turn  # now restored to moving side

    fr, ff = m.from_rank, m.from_file
    tr, tf = m.to_rank, m.to_file

    piece = b.grid[tr, tf]
    m.promotion != 0.0 && (piece = color * PAWN)  # undo promotion: restore pawn

    b.grid[fr, ff] = piece
    b.grid[tr, tf] = undo.captured

    if m.is_en_passant
        b.grid[fr, tf] = -color * PAWN
        b.grid[tr, tf] = 0.0  # ep target square was empty
    end

    if m.is_castling
        if tf == 7  # Kingside
            rook = b.grid[fr, 6]
            b.grid[fr, 6] = 0.0
            b.grid[fr, 8] = rook
        else  # Queenside
            rook = b.grid[fr, 4]
            b.grid[fr, 4] = 0.0
            b.grid[fr, 1] = rook
        end
    end

    b.castling[1] = undo.prev_castling[1]
    b.castling[2] = undo.prev_castling[2]
    b.castling[3] = undo.prev_castling[3]
    b.castling[4] = undo.prev_castling[4]
    b.en_passant  = undo.prev_en_passant
    b.halfmove    = undo.prev_halfmove
    color == BLACK && (b.fullmove -= 1)

    # Restore incrementally maintained fields — O(1), no recomputation needed
    b.hash       = undo.prev_hash
    b.material   = undo.prev_material
    b.white_king = undo.prev_white_king
    b.black_king = undo.prev_black_king

    # Remove the hash that apply_move! pushed onto the history stack.
    pop!(b.history)
end

#= ── Move Generation ───────────────────────────────────────────

    This is the most tedious part — chess rules are intricate.
    We must generate every legal move correctly. No shortcuts.

    The approach:
    1. Generate all "pseudo-legal" moves (moves that follow piece
       movement rules but might leave your king in check)
    2. Filter out moves that leave the king in check

    Pseudo-legal generation is piece-by-piece:
    - Pawns: push, double push, captures, en passant, promotion
    - Knights: 8 possible L-shapes
    - Bishops: 4 diagonal rays until blocked
    - Rooks: 4 straight rays until blocked
    - Queens: bishop rays + rook rays
    - King: 8 adjacent squares + castling
=#

# Direction vectors for sliding pieces
const ROOK_DIRS   = [(1,0), (-1,0), (0,1), (0,-1)]
const BISHOP_DIRS = [(1,1), (1,-1), (-1,1), (-1,-1)]
const QUEEN_DIRS  = [ROOK_DIRS; BISHOP_DIRS]
const KNIGHT_JUMPS = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
const KING_DIRS   = QUEEN_DIRS  # King moves like queen but 1 step
const PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)

function generate_pseudo_legal(b::Board)::Vector{Move}
    moves = Move[]
    color = b.turn

    for r in 1:8, f in 1:8
        !is_color(b, r, f, color) && continue
        pt = piece_type(b, r, f)

        if pt == PAWN
            gen_pawn_moves!(moves, b, r, f, color)
        elseif pt == KNIGHT
            gen_jump_moves!(moves, b, r, f, color, KNIGHT_JUMPS)
        elseif pt == BISHOP
            gen_sliding_moves!(moves, b, r, f, color, BISHOP_DIRS)
        elseif pt == ROOK
            gen_sliding_moves!(moves, b, r, f, color, ROOK_DIRS)
        elseif pt == QUEEN
            gen_sliding_moves!(moves, b, r, f, color, QUEEN_DIRS)
        elseif pt == KING
            gen_jump_moves!(moves, b, r, f, color, KING_DIRS)
            gen_castling_moves!(moves, b, color)
        end
    end

    return moves
end

function gen_pawn_moves!(moves, b, r, f, color)
    # Pawns move "forward" which depends on color
    dir = color  # WHITE (+1) moves up, BLACK (-1) moves down
    start_rank = color == WHITE ? 2 : 7
    promo_rank = color == WHITE ? 8 : 1

    # Single push
    nr = r + dir
    if in_bounds(nr, f) && is_empty(b, nr, f)
        if nr == promo_rank
            # Promotion — generate one move for each promotion piece
            for pp in PROMOTION_PIECES
                push!(moves, Move(r, f, nr, f, color * pp, false, false))
            end
        else
            push!(moves, Move(r, f, nr, f))
        end

        # Double push from starting rank
        nnr = r + 2dir
        if r == start_rank && in_bounds(nnr, f) && is_empty(b, nnr, f)
            push!(moves, Move(r, f, nnr, f))
        end
    end

    # Captures (diagonal)
    for df in (-1, 1)
        nf = f + df
        nr = r + dir
        if !in_bounds(nr, nf); continue; end

        # Normal capture
        if is_color(b, nr, nf, -color) && piece_type(b, nr, nf) != KING
            if nr == promo_rank
                for pp in PROMOTION_PIECES
                    push!(moves, Move(r, f, nr, nf, color * pp, false, false))
                end
            else
                push!(moves, Move(r, f, nr, nf))
            end
        end

        # En passant capture
        if b.en_passant == (nr, nf)
            push!(moves, Move(r, f, nr, nf, 0.0, false, true))
        end
    end
end

function gen_sliding_moves!(moves, b, r, f, color, directions)
    # Slide along each direction until hitting the edge or a piece
    for (dr, df) in directions
        nr, nf = r + dr, f + df
        while in_bounds(nr, nf)
            if is_empty(b, nr, nf)
                push!(moves, Move(r, f, nr, nf))
            elseif is_color(b, nr, nf, -color)
                piece_type(b, nr, nf) != KING && push!(moves, Move(r, f, nr, nf))  # Capture
                break  # Can't slide through a piece
            else
                break  # Blocked by own piece
            end
            nr += dr; nf += df
        end
    end
end

function gen_jump_moves!(moves, b, r, f, color, jumps)
    for (dr, df) in jumps
        nr, nf = r + dr, f + df
        if !in_bounds(nr, nf); continue; end
        if is_color(b, nr, nf, color); continue; end  # Can't capture own piece
        if is_color(b, nr, nf, -color) && piece_type(b, nr, nf) == KING; continue; end
        push!(moves, Move(r, f, nr, nf))
    end
end

function gen_castling_moves!(moves, b, color)
    rank = color == WHITE ? 1 : 8
    ki = color == WHITE ? 1 : 3  # Index into castling array (kingside)
    qi = color == WHITE ? 2 : 4  # Index into castling array (queenside)
    king = color * KING
    rook = color * ROOK

    # If the king or rook is missing from home squares, castling is illegal
    # even if stale castling rights were left true by a custom setup.
    piece_at(b, rank, 5) == king || return

    # Can't castle out of check
    if is_square_attacked(b, rank, 5, -color); return; end

    # Kingside: king on e, rook on h, f and g must be empty and safe
    if b.castling[ki] && piece_at(b, rank, 8) == rook
        if is_empty(b, rank, 6) && is_empty(b, rank, 7)
            if !is_square_attacked(b, rank, 6, -color) &&
               !is_square_attacked(b, rank, 7, -color)
                push!(moves, Move(rank, 5, rank, 7, 0.0, true, false))
            end
        end
    end

    # Queenside: king on e, rook on a, b/c/d must be empty, c/d safe
    if b.castling[qi] && piece_at(b, rank, 1) == rook
        if is_empty(b, rank, 2) && is_empty(b, rank, 3) && is_empty(b, rank, 4)
            if !is_square_attacked(b, rank, 3, -color) &&
               !is_square_attacked(b, rank, 4, -color)
                push!(moves, Move(rank, 5, rank, 3, 0.0, true, false))
            end
        end
    end
end

#= ── Attack Detection ──────────────────────────────────────────

    "Is square (r,f) attacked by 'attacker' color?"

    This is needed for:
    1. Filtering illegal moves (can't leave king in check)
    2. Castling (can't castle through check)
    3. Check/checkmate/stalemate detection

    We check if any enemy piece could reach this square.
=#

function is_square_attacked(b::Board, r, f, attacker)
    # Pawn attacks
    pawn_dir = -attacker  # Pawns attack in opposite direction of movement
    for df in (-1, 1)
        pr, pf = r + pawn_dir, f + df
        if in_bounds(pr, pf) && piece_at(b, pr, pf) == attacker * PAWN
            return true
        end
    end

    # Knight attacks
    for (dr, df) in KNIGHT_JUMPS
        nr, nf = r + dr, f + df
        if in_bounds(nr, nf) && piece_at(b, nr, nf) == attacker * KNIGHT
            return true
        end
    end

    # Sliding attacks (rook/queen along ranks+files, bishop/queen along diags)
    for (dr, df) in ROOK_DIRS
        nr, nf = r + dr, f + df
        while in_bounds(nr, nf)
            v = piece_at(b, nr, nf)
            if v != 0.0
                av = abs(v)
                if sign(v) == attacker && (av == ROOK || av == QUEEN)
                    return true
                end
                break  # Blocked
            end
            nr += dr; nf += df
        end
    end

    for (dr, df) in BISHOP_DIRS
        nr, nf = r + dr, f + df
        while in_bounds(nr, nf)
            v = piece_at(b, nr, nf)
            if v != 0.0
                av = abs(v)
                if sign(v) == attacker && (av == BISHOP || av == QUEEN)
                    return true
                end
                break
            end
            nr += dr; nf += df
        end
    end

    # King attacks (adjacent squares)
    for (dr, df) in KING_DIRS
        nr, nf = r + dr, f + df
        if in_bounds(nr, nf) && piece_at(b, nr, nf) == attacker * KING
            return true
        end
    end

    return false
end

#= ── Legal Move Generation ─────────────────────────────────────

    A move is legal if and only if, after making the move,
    the moving side's king is NOT in check.

    Simple but correct: generate pseudo-legal, try each one,
    check if king is safe, undo.
=#

# King positions are stored directly in the Board struct — O(1) lookup.
# This eliminates an O(64) scan that was previously called inside
# generate_moves! for every pseudo-legal move's legality check.
function find_king(b::Board, color)
    if color == WHITE
        (r, f) = b.white_king
        if !in_bounds(r, f) || b.grid[r, f] != KING
            b.white_king = _scan_king(b, WHITE)
        end
        return b.white_king
    else
        (r, f) = b.black_king
        if !in_bounds(r, f) || b.grid[r, f] != -KING
            b.black_king = _scan_king(b, BLACK)
        end
        return b.black_king
    end
end

function is_in_check(b::Board, color)
    kr, kf = find_king(b, color)
    return is_square_attacked(b, kr, kf, -color)
end

function generate_moves(b::Board)::Vector{Move}
    pseudo = generate_pseudo_legal(b)
    legal = Move[]
    color = b.turn

    for m in pseudo
        undo = apply_move!(b, m)
        # After the move, it's opponent's turn, but we check if
        # OUR king (the one that just moved) is in check
        if !is_in_check(b, color)
            push!(legal, m)
        end
        undo_move!(b, m, undo)
    end

    return legal
end

# ── Game State Detection ───────────────────────────────────────

#= ── Repetition Detection ─────────────────────────────────────
   Returns true if the current position (b.hash) has been seen
   before in the game/search history.

   The Zobrist hash already encodes piece placement, side to move,
   castling rights, and en passant — so equal hashes mean equal
   positions for the purpose of the repetition rule.

   Optimization: we only scan back `halfmove` entries because any
   position before the last irreversible move (capture or pawn push)
   can never repeat — the board is fundamentally different.

   In search, a single repetition (2-fold) is treated as a draw.
   This is standard practice: it prevents the engine from entering
   cycles and is conservative (the game WILL be drawn if neither
   side deviates from the loop).
=#
function is_repetition(b::Board)::Bool
    h = b.hash
    n = length(b.history)
    # Positions before the last irreversible move can't repeat
    check = min(b.halfmove, n)
    @inbounds for i in (n - check + 1):n
        b.history[i] == h && return true
    end
    return false
end

function is_checkmate(b::Board)
    isempty(generate_moves(b)) && is_in_check(b, b.turn)
end

function is_stalemate(b::Board)
    isempty(generate_moves(b)) && !is_in_check(b, b.turn)
end

function is_game_over(b::Board)
    isempty(generate_moves(b)) || b.halfmove ≥ 100 || is_repetition(b)
end

function game_result(b::Board)
    if is_checkmate(b)
        return -b.turn  # The side to move is mated, other side wins
    end
    return 0  # Draw (stalemate, 50-move, repetition, etc.)
end

#= ── In-place Move Generation ──────────────────────────────────
   generate_moves! writes legal moves into a pre-allocated buffer,
   avoiding the allocation that generate_moves creates every call.
   Pass a second buffer for the pseudo-legal scratch space.

   In the optimizer's hot path (millions of calls at depth=5),
   these two allocations per node add up to gigabytes of garbage.
   Reusing buffers drops GC overhead to near zero.
=#

"""
    generate_pseudo_legal!(buf, board)

Fill `buf` with pseudo-legal moves for the side to move.
Clears `buf` first. No allocation.
"""
function generate_pseudo_legal!(buf::Vector{Move}, b::Board)
    empty!(buf)
    color = b.turn
    for r in 1:8, f in 1:8
        !is_color(b, r, f, color) && continue
        pt = piece_type(b, r, f)
        if pt == PAWN
            gen_pawn_moves!(buf, b, r, f, color)
        elseif pt == KNIGHT
            gen_jump_moves!(buf, b, r, f, color, KNIGHT_JUMPS)
        elseif pt == BISHOP
            gen_sliding_moves!(buf, b, r, f, color, BISHOP_DIRS)
        elseif pt == ROOK
            gen_sliding_moves!(buf, b, r, f, color, ROOK_DIRS)
        elseif pt == QUEEN
            gen_sliding_moves!(buf, b, r, f, color, QUEEN_DIRS)
        elseif pt == KING
            gen_jump_moves!(buf, b, r, f, color, KING_DIRS)
            gen_castling_moves!(buf, b, color)
        end
    end
end

"""
    generate_moves!(legal_buf, board, pseudo_buf)

Fill `legal_buf` with all legal moves. Uses `pseudo_buf` as scratch
space for pseudo-legal generation. Both buffers are pre-allocated by
the caller (one pair per thread per ply) so no heap allocation occurs.
"""
function generate_moves!(legal_buf::Vector{Move}, b::Board, pseudo_buf::Vector{Move})
    generate_pseudo_legal!(pseudo_buf, b)
    empty!(legal_buf)
    color = b.turn
    for m in pseudo_buf
        undo = apply_move!(b, m)
        !is_in_check(b, color) && push!(legal_buf, m)
        undo_move!(b, m, undo)
    end
end

#= ── FEN Parser ────────────────────────────────────────────────
   Constructs a Board from a FEN (Forsyth–Edwards Notation) string.
   FEN is the universal standard for describing chess positions:

       r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1

   Fields (space-separated):
     1. Piece placement: ranks 8→1, '/' between ranks, digits = empty squares
     2. Active color: 'w' or 'b'
     3. Castling rights: any combo of K/Q/k/q, or '-'
     4. En passant target square in algebraic notation, or '-'
     5. Halfmove clock (for 50-move rule)
     6. Fullmove number

   This is needed for perft testing — the standard suite uses FEN strings
   to set up positions like Kiwipete, promotion-heavy endgames, etc.
=#

const FEN_PIECE_MAP = Dict{Char, Float64}(
    'P' =>  PAWN,   'N' =>  KNIGHT, 'B' =>  BISHOP,
    'R' =>  ROOK,   'Q' =>  QUEEN,  'K' =>  KING,
    'p' => -PAWN,   'n' => -KNIGHT, 'b' => -BISHOP,
    'r' => -ROOK,   'q' => -QUEEN,  'k' => -KING,
)

function from_fen(fen::String)::Board
    parts = split(strip(fen))
    length(parts) >= 4 || error("Invalid FEN: need at least 4 fields, got $(length(parts))")

    # ── 1. Piece placement ─────────────────────────────────────
    grid = zeros(Float64, 8, 8)
    ranks = split(parts[1], '/')
    length(ranks) == 8 || error("Invalid FEN: piece placement must have 8 ranks")

    for (i, rank_str) in enumerate(ranks)
        rank = 9 - i   # FEN goes rank 8→1; grid row 1 = rank 1
        file = 1
        for ch in rank_str
            if isdigit(ch)
                file += ch - '0'   # skip empty squares
            else
                haskey(FEN_PIECE_MAP, ch) || error("Invalid FEN piece char: '$ch'")
                grid[rank, file] = FEN_PIECE_MAP[ch]
                file += 1
            end
        end
        file == 9 || error("Invalid FEN: rank $rank has wrong number of squares (file=$file)")
    end

    # ── 2. Active color ────────────────────────────────────────
    turn = parts[2] == "w" ? WHITE :
           parts[2] == "b" ? BLACK :
           error("Invalid FEN: active color must be 'w' or 'b', got '$(parts[2])'")

    # ── 3. Castling rights ─────────────────────────────────────
    castling = [false, false, false, false]   # [WK, WQ, BK, BQ]
    if parts[3] != "-"
        for ch in parts[3]
            ch == 'K' && (castling[1] = true)
            ch == 'Q' && (castling[2] = true)
            ch == 'k' && (castling[3] = true)
            ch == 'q' && (castling[4] = true)
        end
    end

    # ── 4. En passant target ───────────────────────────────────
    en_passant = (0, 0)
    if parts[4] != "-"
        ep_file = Int(parts[4][1]) - Int('a') + 1
        ep_rank = Int(parts[4][2]) - Int('0')
        (1 <= ep_file <= 8 && 1 <= ep_rank <= 8) ||
            error("Invalid FEN: en passant square '$(parts[4])'")
        en_passant = (ep_rank, ep_file)
    end

    # ── 5 & 6. Clocks (optional — default to 0/1) ─────────────
    halfmove = length(parts) >= 5 ? parse(Int, parts[5]) : 0
    fullmove = length(parts) >= 6 ? parse(Int, parts[6]) : 1

    b = Board(grid, turn, castling, en_passant, halfmove, fullmove,
              0.0, (0, 0), (0, 0), UInt64(0), UInt64[])
    sync_board!(b)   # computes material, king positions, Zobrist hash
    return b
end

end # module State
