#=
fields.jl — Potential Field Computation

═══════════════════════════════════════════════════════════════
THE BIG IDEA:
═══════════════════════════════════════════════════════════════

Every piece on the board radiates an "influence field" — like a
charged particle emitting an electric field, or a star emitting
gravity. The field gets weaker with distance, and different pieces
have different field shapes:

    • A rook's field extends along straight lines (rank and file)
    • A bishop's field extends along diagonals
    • A knight's field "jumps" — it's strong at L-shaped squares
      and zero everywhere else (very unusual physics!)
    • A queen combines rook + bishop fields
    • A pawn's field is directional — it only projects forward
    • A king's field is intense but short-range (1-2 squares)

The TOTAL field at any square is the sum (superposition) of every
piece's individual field. Friendly pieces contribute positively,
enemy pieces contribute negatively (because their charges have
opposite signs).

═══════════════════════════════════════════════════════════════
WHY THIS WORKS FOR CHESS:
═══════════════════════════════════════════════════════════════

Think about what "controlling a square" means in chess. A square
is "controlled" by White if White has more pieces that can reach
it than Black does. That's exactly what our field computes — the
net influence at each square. Positive = White controls it.
Negative = Black controls it. Zero = contested.

A "weak square" in chess is one where your opponent's field is
strong and yours is weak. A "outpost" is a square where your field
is strong and your opponent can't challenge it. These concepts
emerge naturally from the field math.

═══════════════════════════════════════════════════════════════
THE FIELD EQUATION:
═══════════════════════════════════════════════════════════════

For a single piece at position (pr, pc) with charge Q:

    Φ(r, c) = Q / (1 + d²)

Where d is the "piece-specific distance" from (pr, pc) to (r, c).

Why 1/(1+d²)?
    • It's always positive (for positive Q) — no weird negatives
    • It equals Q when d=0 (at the piece's own square)
    • It decays smoothly with distance — nearby squares feel
      more influence than far ones
    • The +1 prevents division by zero
    • It's the same form as a gravitational/electric potential

But the KEY insight is that d is NOT just geometric distance.
Each piece type has its own distance metric:

    • Rook:   distance along rank/file only (ignoring diagonals)
              AND infinity if blocked by another piece
    • Bishop: distance along diagonals only
              AND infinity if blocked
    • Knight: 1 if reachable by L-jump, infinity otherwise
    • Queen:  minimum of rook-distance and bishop-distance
    • Pawn:   forward-directional distance
    • King:   Chebyshev distance (max of |Δrank|, |Δfile|)

The blocking is crucial — a rook behind 3 pawns has almost no
field beyond those pawns, just like in real chess where a blocked
rook is nearly useless.

═══════════════════════════════════════════════════════════════
FIELD TENSION (∇Φ):
═══════════════════════════════════════════════════════════════

The gradient of the field — how fast it changes between adjacent
squares — tells us where the action is. High gradient means both
sides have strong, competing influence. In chess terms, these are
the "critical squares" where tactics happen.

    Tension(r,c) = |Φ(r,c) - Φ(r+1,c)|² + |Φ(r,c) - Φ(r,c+1)|²

We compute this as a second 8×8 matrix. The engine will be
attracted to positions where tension is high near the enemy king
(attacking) and low near its own king (safety).
=#

module Fields

using Printf

# We need to access the board representation from state.jl
using ..State

export compute_piece_field, compute_total_field, compute_total_field!
export compute_field_tension
export compute_mobility_field, compute_mobility_count, print_field
export update_piece_field!, find_ray_blockers!

# ── Pre-computed field decay factors ─────────────────────────────
# Φ(d) = Q / (1 + d²)  →  Q * SLIDING_DECAY[d]
# Avoids a float division in the tightest inner loop of the search.
# Maximum ray length on an 8×8 board is 7 squares.
const SLIDING_DECAY = ntuple(d -> 1.0 / (1.0 + d * d), 7)

#= ═══════════════════════════════════════════════════════════════
   INDIVIDUAL PIECE FIELDS
   ═══════════════════════════════════════════════════════════════

   Each function below computes the field emitted by a single piece
   at position (pr, pf) with charge Q, writing into the 8×8 matrix
   `field`. We accumulate (+=) so that calling this for every piece
   builds up the total field by superposition.

   The charge Q already has the sign baked in:
     Q > 0 for White pieces → positive field contribution
     Q < 0 for Black pieces → negative field contribution
   
   This means the total field is automatically:
     Positive where White dominates
     Negative where Black dominates
     Near zero where both sides contest
=#

"""
    compute_sliding_field!(field, board, pr, pf, Q, directions)

Compute the field for a sliding piece (rook, bishop, queen).

A sliding piece radiates along straight lines (rays), and its
field is BLOCKED by intervening pieces — just like light being
blocked by an opaque object. This is physically realistic and
also chess-accurate: a rook behind a wall of pawns has very
limited influence.

For each ray direction:
  1. Start at the piece's square
  2. Walk outward one square at a time
  3. At each square, add field contribution: Q / (1 + distance²)
  4. If we hit a piece, add one more contribution (you can
     "see" the piece you're blocked by) then stop the ray

The distance is measured in steps along the ray, not Euclidean
distance. This means a rook's field treats rank/file distance
as "close" and diagonal distance as "infinitely far" — exactly
matching how a rook moves in chess.
"""
function compute_sliding_field!(field::Matrix{Float64}, b::Board,
                                 pr::Int, pf::Int, Q::Float64,
                                 directions::Vector{Tuple{Int,Int}})
    # The piece's own square gets the full charge
    # (We handle this in compute_piece_field so we don't double-count)

    for (dr, df) in directions
        # Walk along this ray
        dist = 1       # Start at distance 1 (first square away)
        r, f = pr + dr, pf + df

        while State.in_bounds(r, f)
            # Field decays with distance squared: Q * 1/(1+d²)
            # SLIDING_DECAY is pre-computed to avoid division in tight loop.
            field[r, f] += Q * @inbounds SLIDING_DECAY[dist]

            # If this square has a piece, the ray is blocked
            # (We still added the field AT this square — you can
            #  see a piece that blocks you, you just can't see past it)
            if !is_empty(b, r, f)
                break
            end

            # Continue along the ray
            dist += 1
            r += dr
            f += df
        end
    end
end

"""
    compute_knight_field!(field, board, pr, pf, Q)

Compute the field for a knight.

Knights are the weirdest piece, physically. Their field doesn't
decay smoothly with distance — it's either ON (at the 8 L-shaped
squares) or OFF (everywhere else). This is like a quantum particle
that can only exist at certain discrete positions.

Mathematically:
    Φ(r,c) = Q × 0.8  if (r,c) is an L-jump from (pr,pf)
    Φ(r,c) = 0         otherwise

The 0.8 factor (instead of the full Q) represents the knight's
slightly reduced influence compared to being directly on a square.
A knight "controls" its target squares but with less authority
than a piece sitting right there.

Knights also can't be blocked — their field passes over pieces,
just like the knight jumps over pieces in chess. This is the one
piece whose field is truly non-physical.
"""
function compute_knight_field!(field::Matrix{Float64}, b::Board,
                                pr::Int, pf::Int, Q::Float64)
    # The 8 possible L-jumps a knight can make
    for (dr, df) in State.KNIGHT_JUMPS
        r, f = pr + dr, pf + df
        if State.in_bounds(r, f)
            # Knight field is strong but doesn't decay — it's either
            # there or it isn't. We use 0.8 * Q as the field strength.
            #
            # Why 0.8? Because a knight "influences" its target squares
            # but doesn't physically sit on them. It's a compromise
            # between "full control" (1.0) and "mere threat" (less).
            field[r, f] += Q * 0.8
        end
    end
end

"""
    compute_pawn_field!(field, board, pr, pf, Q)

Compute the field for a pawn.

Pawns are unique — they're the only piece with DIRECTIONAL fields.
A pawn's influence goes forward (toward the enemy), not backward.
This captures the chess reality that pawns are attacking pieces
that project force in one direction.

A pawn emits field in three ways:
  1. ATTACK FIELD: Strong influence on the two diagonal-forward
     squares (these are where the pawn captures). This is the
     pawn's primary contribution — it "guards" these squares.
  
  2. PUSH FIELD: Moderate influence directly forward (where it
     can advance). This represents the pawn's potential to
     advance and claim space.
  
  3. STRUCTURE FIELD: Weak influence on the squares behind it.
     Pawns create a "shadow" of control behind them — this models
     how pawns define the structure of a position.

The pawn's direction depends on color:
    White pawns (Q > 0) project upward (increasing rank)
    Black pawns (Q < 0) project downward (decreasing rank)
"""
function compute_pawn_field!(field::Matrix{Float64}, b::Board,
                              pr::Int, pf::Int, Q::Float64)
    # Direction: White goes up (+1), Black goes down (-1)
    dir = Q > 0 ? 1 : -1

    # ── Attack field (diagonal forward squares) ──
    # These are the squares the pawn threatens to capture on.
    # They get strong field because pawn threats are one of the
    # most important tactical elements in chess.
    for df in (-1, 1)
        r, f = pr + dir, pf + df
        if State.in_bounds(r, f)
            # Strong influence — pawns are excellent defenders/attackers
            # of their diagonal squares
            field[r, f] += Q * 0.9
        end

        # Extended diagonal influence (2 squares forward-diagonal)
        # Weaker, but represents the pawn's potential to advance
        # and threaten more squares
        r2, f2 = pr + 2dir, pf + df
        if State.in_bounds(r2, f2)
            field[r2, f2] += Q * 0.3
        end
    end

    # ── Push field (straight forward) ──
    # The pawn's ability to advance projects influence forward
    r1 = pr + dir
    if State.in_bounds(r1, pf)
        field[r1, pf] += Q * 0.4

        # Second square forward (pawn's march potential)
        r2 = pr + 2dir
        if State.in_bounds(r2, pf)
            field[r2, pf] += Q * 0.2
        end
    end

    # ── Structure field (behind the pawn) ──
    # Pawns create a "base" of control behind them.
    # This is weak but models how pawn chains work:
    # pawns support each other from behind.
    r_back = pr - dir
    if State.in_bounds(r_back, pf)
        field[r_back, pf] += Q * 0.15
    end
end

"""
    compute_king_field!(field, board, pr, pf, Q)

Compute the field for a king.

The king has MASSIVE charge (±100) but very SHORT range. This
creates a field that's overwhelmingly strong in the immediate
vicinity but drops off rapidly — like a super-dense star.

Mathematically, we use Chebyshev distance (d = max(|Δr|, |Δf|))
which treats all 8 directions equally. The king's field only
extends 3 squares, but within that range it's by far the
strongest field on the board.

This huge local field has two effects:
  1. DEFENSE: The king's field repels enemy piece fields nearby,
     making it hard for the enemy to build up influence close
     to the king. If the defense is breached (enemy field
     penetrates despite the king's field), that signals danger.
  
  2. ATTACK MAGNET: Because the enemy king has an equally massive
     field of opposite sign, the gradient between the two kings
     is enormous. The engine will naturally try to project its
     pieces' fields toward the enemy king — not because we told
     it to attack, but because that's where the biggest energy
     gains are in the math.
"""
function compute_king_field!(field::Matrix{Float64}, b::Board,
                              pr::Int, pf::Int, Q::Float64)
    # Short range but intense — scan a 5×5 area centered on king
    # (Chebyshev distance ≤ 2)
    for dr in -2:2, df in -2:2
        # Skip the king's own square (handled separately)
        if dr == 0 && df == 0; continue; end

        r, f = pr + dr, pf + df
        if !State.in_bounds(r, f); continue; end

        # Chebyshev distance: max of horizontal and vertical distance
        # This is the "king distance" — how many king moves to get there
        dist = max(abs(dr), abs(df))

        # Field decays with distance squared, same formula as others.
        # But Q is ±100, so even at distance 2 the field is
        # 100*0.2 = 20.0, which is stronger than a queen's
        # field at distance 0 (9.0)!
        field[r, f] += Q * @inbounds SLIDING_DECAY[dist]
    end
end

#= ═══════════════════════════════════════════════════════════════
   TOTAL FIELD COMPUTATION
   ═══════════════════════════════════════════════════════════════

   The total field is the superposition (sum) of every individual
   piece's field. This is exactly like computing the total electric
   field from multiple charges, or the total gravitational field
   from multiple masses.

   Because our charges are signed (+ for White, - for Black),
   the superposition automatically handles friend vs foe:

   • Where White has lots of pieces nearby → strongly positive
   • Where Black has lots of pieces nearby → strongly negative
   • Where both sides compete → the fields partially cancel,
     leaving a small net value (contested territory)

   The resulting 8×8 matrix IS the engine's understanding of
   the position. Every strategic concept — control, safety,
   weakness, pressure — is encoded in these 64 numbers.
═══════════════════════════════════════════════════════════════ =#

"""
    compute_piece_field(board, pr, pf) -> Matrix{Float64}

Compute the 8×8 field emitted by a single piece at (pr, pf).
Returns a fresh matrix (useful for visualization of individual
piece contributions).
"""
function compute_piece_field(b::Board, pr::Int, pf::Int)::Matrix{Float64}
    field = zeros(Float64, 8, 8)

    # Get the piece's charge (signed value from the board matrix)
    Q = piece_at(b, pr, pf)
    if Q == 0.0; return field; end  # Empty square, no field

    pt = abs(Q)  # Piece type (magnitude)

    # The piece's own square gets its full charge
    # (The piece is "here", so its field is strongest here)
    field[pr, pf] = Q

    # Compute the appropriate field based on piece type
    if pt == PAWN
        compute_pawn_field!(field, b, pr, pf, Q)

    elseif pt == KNIGHT
        compute_knight_field!(field, b, pr, pf, Q)

    elseif pt == BISHOP
        # Bishop slides along diagonals
        compute_sliding_field!(field, b, pr, pf, Q, State.BISHOP_DIRS)

    elseif pt == ROOK
        # Rook slides along ranks and files
        compute_sliding_field!(field, b, pr, pf, Q, State.ROOK_DIRS)

    elseif pt == QUEEN
        # Queen = rook + bishop (slides in all 8 directions)
        compute_sliding_field!(field, b, pr, pf, Q, State.QUEEN_DIRS)

    elseif pt == KING
        compute_king_field!(field, b, pr, pf, Q)
    end

    return field
end

"""
    compute_total_field(board) -> Matrix{Float64}

Compute the superposition of all piece fields on the board.

This is the main field computation — the result is an 8×8 matrix
where each cell contains the net influence at that square.

    Φ_total(r, f) = Σ  Φ_piece(r, f)  for all pieces on the board

Positive values = White-controlled territory
Negative values = Black-controlled territory
Near zero = contested ground
"""
function compute_total_field(b::Board)::Matrix{Float64}
    # Start with an empty field (no influence anywhere)
    field = zeros(Float64, 8, 8)

    # Iterate over every square on the board
    for r in 1:8, f in 1:8
        Q = piece_at(b, r, f)

        # Skip empty squares — they don't emit fields
        if Q == 0.0; continue; end

        pt = abs(Q)

        # Add this piece's charge at its own square
        field[r, f] += Q

        # Add the piece's field radiation to surrounding squares
        if pt == PAWN
            compute_pawn_field!(field, b, r, f, Q)
        elseif pt == KNIGHT
            compute_knight_field!(field, b, r, f, Q)
        elseif pt == BISHOP
            compute_sliding_field!(field, b, r, f, Q, State.BISHOP_DIRS)
        elseif pt == ROOK
            compute_sliding_field!(field, b, r, f, Q, State.ROOK_DIRS)
        elseif pt == QUEEN
            compute_sliding_field!(field, b, r, f, Q, State.QUEEN_DIRS)
        elseif pt == KING
            compute_king_field!(field, b, r, f, Q)
        end
    end

    return field
end

"""
    compute_total_field!(field, board)

Mutating variant of compute_total_field — writes into a pre-allocated 8×8
matrix instead of allocating a new one. The caller must pass a buffer of
the right size; this function resets it with fill! before accumulating.

Use this in hot paths (search, optimizer) to eliminate per-call allocation.
"""
function compute_total_field!(field::Matrix{Float64}, b::Board)
    fill!(field, 0.0)
    for r in 1:8, f in 1:8
        Q = piece_at(b, r, f)
        Q == 0.0 && continue
        pt = abs(Q)
        field[r, f] += Q
        if pt == PAWN
            compute_pawn_field!(field, b, r, f, Q)
        elseif pt == KNIGHT
            compute_knight_field!(field, b, r, f, Q)
        elseif pt == BISHOP
            compute_sliding_field!(field, b, r, f, Q, State.BISHOP_DIRS)
        elseif pt == ROOK
            compute_sliding_field!(field, b, r, f, Q, State.ROOK_DIRS)
        elseif pt == QUEEN
            compute_sliding_field!(field, b, r, f, Q, State.QUEEN_DIRS)
        elseif pt == KING
            compute_king_field!(field, b, r, f, Q)
        end
    end
end

#= ═══════════════════════════════════════════════════════════════
   FIELD TENSION (GRADIENT)
   ═══════════════════════════════════════════════════════════════

   The gradient of the field tells us where it's CHANGING rapidly.
   In physics, high gradient = high force. In chess, high gradient
   = high tactical tension.

   Think about it: if the field goes from +20 (White-controlled)
   to -15 (Black-controlled) across one square, that boundary is
   a battleground. Pieces near that boundary are in danger or
   creating threats. That's where tactics happen — captures, forks,
   pins all occur at high-tension squares.

   We compute the gradient magnitude at each square:

       Tension(r,f) = (Φ(r,f) - Φ(r+1,f))² + (Φ(r,f) - Φ(r,f+1))²

   This is the discrete analog of |∇Φ|² from calculus.
   High tension squares are where the engine should focus its
   search — they're the most tactically rich.
═══════════════════════════════════════════════════════════════ =#

"""
    compute_field_tension(field) -> Matrix{Float64}

Compute the tension (gradient magnitude) of a field.

Returns an 8×8 matrix where high values indicate squares with
rapidly changing field — these are the tactical hotspots.
"""
function compute_field_tension(field::Matrix{Float64})::Matrix{Float64}
    tension = zeros(Float64, 8, 8)

    for r in 1:8, f in 1:8
        grad_sq = 0.0

        # Gradient in rank direction (vertical)
        # Compare this square to the one above it
        if r < 8
            diff_r = field[r, f] - field[r + 1, f]
            grad_sq += diff_r * diff_r
        end

        # Gradient in file direction (horizontal)
        # Compare this square to the one to the right
        if f < 8
            diff_f = field[r, f] - field[r, f + 1]
            grad_sq += diff_f * diff_f
        end

        # Also check the diagonal gradient — this catches tension
        # along diagonal lines (important for bishops and pawns)
        if r < 8 && f < 8
            diff_d = field[r, f] - field[r + 1, f + 1]
            grad_sq += 0.5 * diff_d * diff_d  # Half weight for diagonals
        end

        tension[r, f] = grad_sq
    end

    return tension
end

#= ═══════════════════════════════════════════════════════════════
   MOBILITY FIELD
   ═══════════════════════════════════════════════════════════════

   Mobility = how many squares a side's pieces can reach.
   
   In field terms, mobility is the "flux" of the field — the
   total field flowing outward from all your pieces. A piece
   with lots of legal moves emits field to many squares (high
   flux). A blocked piece emits field to few squares (low flux).

   We compute this as a simple count: for each square, how many
   of your pieces could move there? This is a simpler field than
   the potential field above, but it captures an important concept:
   piece activity.

   A rook on an open file has high mobility (field reaches far).
   A bishop hemmed in by pawns has low mobility (field is blocked).
   The engine should prefer positions with high friendly mobility
   and low enemy mobility.
═══════════════════════════════════════════════════════════════ =#

"""
    compute_mobility_field(board, color) -> Matrix{Float64}

Compute a field where each square's value is the number of
times pieces of `color` can reach it.

This is a "how active are your pieces" field.
"""
function compute_mobility_field(b::Board, color::Int)::Matrix{Float64}
    mob = zeros(Float64, 8, 8)

    for r in 1:8, f in 1:8
        # Only look at pieces of the requested color
        if !is_color(b, r, f, color); continue; end

        pt = piece_type(b, r, f)

        if pt == PAWN
            # Pawn mobility: the squares it can move to
            dir = color
            # Forward push
            nr = r + dir
            if State.in_bounds(nr, f) && is_empty(b, nr, f)
                mob[nr, f] += 1.0
            end
            # Diagonal captures (count even if no enemy there —
            # it's potential mobility)
            for df in (-1, 1)
                nf = f + df
                if State.in_bounds(nr, nf)
                    mob[nr, nf] += 1.0
                end
            end

        elseif pt == KNIGHT
            for (dr, df) in State.KNIGHT_JUMPS
                nr, nf = r + dr, f + df
                if State.in_bounds(nr, nf) && !is_color(b, nr, nf, color)
                    mob[nr, nf] += 1.0
                end
            end

        elseif pt == BISHOP
            for (dr, df) in State.BISHOP_DIRS
                nr, nf = r + dr, f + df
                while State.in_bounds(nr, nf)
                    if is_color(b, nr, nf, color); break; end
                    mob[nr, nf] += 1.0
                    if !is_empty(b, nr, nf); break; end
                    nr += dr; nf += df
                end
            end

        elseif pt == ROOK
            for (dr, df) in State.ROOK_DIRS
                nr, nf = r + dr, f + df
                while State.in_bounds(nr, nf)
                    if is_color(b, nr, nf, color); break; end
                    mob[nr, nf] += 1.0
                    if !is_empty(b, nr, nf); break; end
                    nr += dr; nf += df
                end
            end

        elseif pt == QUEEN
            for (dr, df) in State.QUEEN_DIRS
                nr, nf = r + dr, f + df
                while State.in_bounds(nr, nf)
                    if is_color(b, nr, nf, color); break; end
                    mob[nr, nf] += 1.0
                    if !is_empty(b, nr, nf); break; end
                    nr += dr; nf += df
                end
            end

        elseif pt == KING
            for (dr, df) in State.KING_DIRS
                nr, nf = r + dr, f + df
                if State.in_bounds(nr, nf) && !is_color(b, nr, nf, color)
                    mob[nr, nf] += 1.0
                end
            end
        end
    end

    return mob
end

"""
    compute_mobility_count(board, color) -> Float64

Non-allocating version of sum(compute_mobility_field(b, color)).
Returns the total mobility score directly without building the 8×8 matrix.
Use this in hot paths to eliminate the per-call matrix allocation.
"""
function compute_mobility_count(b::Board, color::Int)::Float64
    total = 0.0
    for r in 1:8, f in 1:8
        is_color(b, r, f, color) || continue
        pt = piece_type(b, r, f)

        if pt == PAWN
            dir = color
            nr  = r + dir
            if State.in_bounds(nr, f) && is_empty(b, nr, f)
                total += 1.0
            end
            for df in (-1, 1)
                nf = f + df
                State.in_bounds(nr, nf) && (total += 1.0)
            end

        elseif pt == KNIGHT
            for (dr, df) in State.KNIGHT_JUMPS
                nr, nf = r + dr, f + df
                State.in_bounds(nr, nf) && !is_color(b, nr, nf, color) && (total += 1.0)
            end

        elseif pt == BISHOP || pt == ROOK || pt == QUEEN
            dirs = pt == BISHOP ? State.BISHOP_DIRS :
                   pt == ROOK   ? State.ROOK_DIRS   : State.QUEEN_DIRS
            for (dr, df) in dirs
                nr, nf = r + dr, f + df
                while State.in_bounds(nr, nf)
                    is_color(b, nr, nf, color) && break
                    total += 1.0
                    !is_empty(b, nr, nf) && break
                    nr += dr; nf += df
                end
            end

        elseif pt == KING
            for (dr, df) in State.KING_DIRS
                nr, nf = r + dr, f + df
                State.in_bounds(nr, nf) && !is_color(b, nr, nf, color) && (total += 1.0)
            end
        end
    end
    return total
end

#= ═══════════════════════════════════════════════════════════════
   VISUALIZATION
   ═══════════════════════════════════════════════════════════════

   Being able to SEE the field is crucial for understanding what
   the engine "thinks." We print the field as a heatmap using
   characters:

   Strong positive (White control):  █ ▓ ▒
   Near zero (contested):            ·
   Strong negative (Black control):  ░ ▒ ▓

   We also print the raw numbers for precise analysis.
═══════════════════════════════════════════════════════════════ =#

"""
    print_field(field; title="Field", show_numbers=true)

Print a field as a visual heatmap with optional numeric values.

The heatmap uses block characters to show field intensity:
  Strong positive (White territory)  →  bright/dense blocks
  Near zero (contested)              →  dots
  Strong negative (Black territory)  →  dark blocks with minus
"""
function print_field(field::Matrix{Float64};
                     title::String = "Field",
                     show_numbers::Bool = true)

    # Find the range for scaling the heatmap
    max_abs = maximum(abs.(field))
    if max_abs == 0.0; max_abs = 1.0; end  # Prevent division by zero

    println("\n  ═══ $title ═══")
    println()

    if show_numbers
        # Print with actual values
        println("    a      b      c      d      e      f      g      h")
        for rank in 8:-1:1
            @printf(" %d ", rank)
            for file in 1:8
                v = field[rank, file]
                # Color-code: positive values are White territory
                if abs(v) < 0.1
                    print("   ·   ")
                else
                    @printf("%+6.1f ", v)
                end
            end
            println()
        end
    end

    # Print as heatmap
    println()
    println("    a  b  c  d  e  f  g  h")
    for rank in 8:-1:1
        print(" $rank  ")
        for file in 1:8
            v = field[rank, file]
            intensity = abs(v) / max_abs

            # Choose character based on intensity and sign
            if intensity < 0.05
                print("·  ")        # Nearly zero — contested
            elseif v > 0
                # Positive (White territory) — brighter = stronger
                if intensity > 0.7
                    print("█  ")    # Strong White control
                elseif intensity > 0.4
                    print("▓  ")    # Medium White control
                elseif intensity > 0.2
                    print("▒  ")    # Weak White control
                else
                    print("░  ")    # Slight White influence
                end
            else
                # Negative (Black territory) — with minus indicator
                if intensity > 0.7
                    print("▼  ")    # Strong Black control
                elseif intensity > 0.4
                    print("◆  ")    # Medium Black control
                elseif intensity > 0.2
                    print("◇  ")    # Weak Black control
                else
                    print("○  ")    # Slight Black influence
                end
            end
        end
        println()
    end
    println()
end

#= ═══════════════════════════════════════════════════════════════
   INCREMENTAL FIELD MAINTENANCE
   ═══════════════════════════════════════════════════════════════

   These functions allow the optimizer's search to maintain the
   field matrix incrementally across apply_move!/undo_move! calls,
   rather than recomputing it from scratch at every leaf node.

   HOW IT WORKS:
   When a piece moves from A → B, the field changes because:
   1. The piece's own contribution moves from A to B
   2. Any captured piece at B is removed
   3. Sliding pieces whose rays passed THROUGH A can now extend
      past A (A is now empty)
   4. Sliding pieces whose rays passed THROUGH B are now clipped
      at B (B is now occupied) — only relevant if B was empty

   We handle this with two operations:
     update_piece_field!(field, b, r, f, sign)
       Add (sign=+1) or subtract (sign=-1) one piece's contribution.
       Uses the CURRENT board state for blocking — so call before
       the move to subtract old contribution, after to add new.

     find_ray_blockers!(buf, b, r, f)
       Fill `buf` with all sliding pieces that have (r,f) on their
       active ray. These must be re-evaluated when (r,f) changes.
═══════════════════════════════════════════════════════════════ =#

# ── Sliding field delta helper ─────────────────────────────────
# Same logic as compute_sliding_field! but uses Q_delta (which may
# be negative for subtraction) rather than the raw piece charge.
function _sliding_field_delta!(field::Matrix{Float64}, b::Board,
                                pr::Int, pf::Int, Q_delta::Float64,
                                directions)
    for (dr, df) in directions
        dist = 1
        r, f = pr + dr, pf + df
        while State.in_bounds(r, f)
            field[r, f] += Q_delta * @inbounds SLIDING_DECAY[dist]
            is_empty(b, r, f) || break
            dist += 1
            r += dr; f += df
        end
    end
end

"""
    update_piece_field!(field, board, pr, pf, sign)

Add (sign = +1) or subtract (sign = -1) the field contribution of
the piece currently at (pr, pf), using the current board for blocking.

Call with sign = -1 BEFORE apply_move! to remove the old contribution.
Call with sign = +1 AFTER apply_move! to add the new contribution.
If the square is empty, this is a no-op.
"""
function update_piece_field!(field::Matrix{Float64}, b::Board,
                              pr::Int, pf::Int, sign::Int)
    Q = b.grid[pr, pf]
    Q == 0.0 && return
    pt  = abs(Q)
    # Q_delta carries the sign for addition/subtraction.
    # For direction-sensitive pieces (pawns) we still use the actual Q.
    Q_delta = Q * sign

    # Own square
    field[pr, pf] += Q_delta

    if pt == PAWN
        # Direction is determined by the ACTUAL piece charge (not delta).
        dir = Q > 0 ? 1 : -1
        for df in (-1, 1)
            r, f = pr + dir, pf + df
            State.in_bounds(r, f) && (field[r, f] += Q_delta * 0.9)
            r2 = pr + 2dir
            State.in_bounds(r2, pf + df) && (field[r2, pf + df] += Q_delta * 0.3)
        end
        r1 = pr + dir
        if State.in_bounds(r1, pf)
            field[r1, pf] += Q_delta * 0.4
            r2 = pr + 2dir
            State.in_bounds(r2, pf) && (field[r2, pf] += Q_delta * 0.2)
        end
        r_back = pr - dir
        State.in_bounds(r_back, pf) && (field[r_back, pf] += Q_delta * 0.15)

    elseif pt == KNIGHT
        for (dr, df) in State.KNIGHT_JUMPS
            r, f = pr + dr, pf + df
            State.in_bounds(r, f) && (field[r, f] += Q_delta * 0.8)
        end

    elseif pt == BISHOP
        _sliding_field_delta!(field, b, pr, pf, Q_delta, State.BISHOP_DIRS)

    elseif pt == ROOK
        _sliding_field_delta!(field, b, pr, pf, Q_delta, State.ROOK_DIRS)

    elseif pt == QUEEN
        _sliding_field_delta!(field, b, pr, pf, Q_delta, State.QUEEN_DIRS)

    elseif pt == KING
        for dr in -2:2, df in -2:2
            (dr == 0 && df == 0) && continue
            r, f = pr + dr, pf + df
            !State.in_bounds(r, f) && continue
            dist = max(abs(dr), abs(df))
            field[r, f] += Q_delta * @inbounds SLIDING_DECAY[dist]
        end
    end
end

"""
    find_ray_blockers!(buf, board, r, f)

Fill `buf` with all sliding pieces that currently have square (r, f)
on one of their active rays. Does NOT allocate — appends to `buf`
(caller must call empty!(buf) first if needed).

"Active ray" means: the piece can see (r, f) along a rank, file, or
diagonal with no intervening pieces between them.

We find these by scanning BACKWARD from (r, f) in each of the 8
directions and stopping at the first piece we encounter. If that
piece is a slider capable of projecting along that direction, then
(r, f) is on its ray.
"""
function find_ray_blockers!(buf::Vector{Tuple{Int,Int}}, b::Board, r::Int, f::Int)
    # Rook-type rays (rank + file) → look for rooks or queens
    for (dr, df) in State.ROOK_DIRS
        nr, nf = r - dr, f - df          # scan backward toward the potential slider
        while State.in_bounds(nr, nf)
            Q = b.grid[nr, nf]
            if Q != 0.0
                pt = abs(Q)
                (pt == ROOK || pt == QUEEN) && push!(buf, (nr, nf))
                break
            end
            nr -= dr; nf -= df
        end
    end
    # Bishop-type rays (diagonals) → look for bishops or queens
    for (dr, df) in State.BISHOP_DIRS
        nr, nf = r - dr, f - df
        while State.in_bounds(nr, nf)
            Q = b.grid[nr, nf]
            if Q != 0.0
                pt = abs(Q)
                (pt == BISHOP || pt == QUEEN) && push!(buf, (nr, nf))
                break
            end
            nr -= dr; nf -= df
        end
    end
end

end # module Fields
