#=
demo_fields.jl — Visualize the potential fields.

Run with:
    julia test/demo_fields.jl

This script shows you what the engine "sees" — the invisible
force fields radiating from every piece on the board. It's the
most important script for building intuition about how the
engine thinks.

You'll see:
  1. The total field for the starting position
  2. Individual piece fields (what does a single rook emit?)
  3. Field tension (where are the tactical hotspots?)
  4. A custom position to demonstrate attacking fields
=#

using Printf

# Load our modules
include(joinpath(@__DIR__, "..", "src", "state.jl"))
include(joinpath(@__DIR__, "..", "src", "fields.jl"))

using .State
using .Fields

println("╔══════════════════════════════════════════════════╗")
println("║          FieldEngine — Field Visualizer          ║")
println("╚══════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════
# DEMO 1: Starting position total field
# ═══════════════════════════════════════════════════════════════

println("\n" * "─"^55)
println("DEMO 1: Total field — Starting position")
println("─"^55)
println("""
This is the superposition of ALL piece fields in the starting
position. Positive values (█▓▒░) = White territory.
Negative values (▼◆◇○) = Black territory.

Notice how the field is roughly symmetric — both sides have
equal material, so their fields roughly cancel out. But look
at the CENTER (d4, d5, e4, e5) — that's where the fields
are most contested. This is why chess players fight for the
center: it's where the field tension is highest!
""")

b = new_board()
print_board(b)

field = compute_total_field(b)
print_field(field, title="Total Field — Starting Position")

# ═══════════════════════════════════════════════════════════════
# DEMO 2: Individual piece fields
# ═══════════════════════════════════════════════════════════════

println("\n" * "─"^55)
println("DEMO 2: Individual piece fields")
println("─"^55)
println("""
Let's see what a single piece's field looks like.
We'll look at the White queen on d1 — the strongest
mobile piece (charge = +9.0).

Notice how her field radiates along all 8 directions
(she combines rook + bishop movement), but gets blocked
by the pawns in front of her. In the starting position,
the queen's field barely reaches past rank 2!

This is why developing pieces matters: you're unblocking
their field emissions.
""")

# Queen is at d1 = (rank 1, file 4)
queen_field = compute_piece_field(b, 1, 4)
print_field(queen_field, title="White Queen (d1) Field")

println("""
Now compare with a knight on g1 (rank 1, file 7).
The knight's field JUMPS over the pawns — it's the only
piece whose field isn't blocked! That's why knights are
valuable in cramped positions.
""")

knight_field = compute_piece_field(b, 1, 7)
print_field(knight_field, title="White Knight (g1) Field")

# ═══════════════════════════════════════════════════════════════
# DEMO 3: Field tension
# ═══════════════════════════════════════════════════════════════

println("\n" * "─"^55)
println("DEMO 3: Field tension (tactical hotspots)")
println("─"^55)
println("""
Field tension = how fast the field changes between adjacent
squares. High tension means both sides have competing influence.

In the starting position, tension is highest in the CENTER —
that's where White's field (radiating upward from ranks 1-2)
meets Black's field (radiating downward from ranks 7-8).

The edge squares have low tension because only one side's
field reaches them strongly.
""")

tension = compute_field_tension(field)
print_field(tension, title="Field Tension — Starting Position")

# ═══════════════════════════════════════════════════════════════
# DEMO 4: Open position — see how fields change
# ═══════════════════════════════════════════════════════════════

println("\n" * "─"^55)
println("DEMO 4: Open position with active pieces")
println("─"^55)
println("""
Let's set up a position where pieces are active and see
how the fields look. White has a bishop on c4 aiming at f7
(a common attacking pattern), and a rook on an open e-file.

Watch how the active pieces' fields PENETRATE into Black's
territory — this is what "piece activity" looks like in
field terms.
""")

# Create a custom position
b2 = new_board()
b2.grid .= 0.0  # Clear the board

# White pieces
b2.grid[1, 5] = KING      # King on e1
b2.grid[3, 3] = BISHOP    # Bishop on c3 — aiming at Black's king
b2.grid[4, 3] = BISHOP    # Active bishop on c4
b2.grid[1, 1] = ROOK      # Rook on a1
b2.grid[2, 1] = PAWN; b2.grid[2, 2] = PAWN; b2.grid[2, 6] = PAWN
b2.grid[2, 7] = PAWN; b2.grid[2, 8] = PAWN

# Black pieces
b2.grid[8, 5] = -KING     # King on e8
b2.grid[8, 1] = -ROOK     # Rook on a8
b2.grid[7, 1] = -PAWN; b2.grid[7, 2] = -PAWN; b2.grid[7, 4] = -PAWN
b2.grid[7, 6] = -PAWN; b2.grid[7, 7] = -PAWN; b2.grid[6, 5] = -KNIGHT

b2.castling = [false, false, false, false]
sync_board!(b2)

print_board(b2)

field2 = compute_total_field(b2)
print_field(field2, title="Total Field — Active White Pieces")

tension2 = compute_field_tension(field2)
print_field(tension2, title="Tension — Where Should the Action Be?")

println("""
Look at the tension map! The highest tension should be near
Black's king and along the diagonals/files where White's pieces
are aiming. The engine will naturally search these squares
more deeply — not because we told it to, but because the math
says that's where the position is most volatile.
""")

# ═══════════════════════════════════════════════════════════════
# DEMO 5: Mobility fields
# ═══════════════════════════════════════════════════════════════

println("─"^55)
println("DEMO 5: Mobility (piece activity)")
println("─"^55)
println("""
Mobility field shows how many of your pieces can reach each
square. Active pieces create high mobility; blocked pieces
create low mobility. This is the "flux" of the field.
""")

b3 = new_board()
white_mob = compute_mobility_field(b3, WHITE)
print_field(white_mob, title="White Mobility — Starting Position")

println("""
Notice ranks 3-4 have the highest mobility — that's where
White's pieces can reach from the starting position. Deeper
ranks (5-8) have almost zero White mobility because the pawns
and pieces haven't developed yet. This is why opening theory
focuses on piece development: you're increasing your mobility
field (flux).
""")

println("═"^55)
println("  Field visualization complete!")
println("  Next: energy.jl will combine these fields into a")
println("  single evaluation score for any position.")
println("═"^55)
