#=
play.jl — Interactive terminal play against FieldEngine.

Run from the project root:
    julia src/play.jl              # human plays White, engine plays Black
    julia src/play.jl black        # human plays Black, engine plays White
    julia src/play.jl white 5      # human is White, engine searches 5 plies deep

Enter moves in coordinate notation: e2e4, g1f3, e7e8q (queen promotion).
Omitting the promotion piece defaults to queen.

Commands:
    help     list all legal moves in the current position
    eval     print a verbose evaluation breakdown
    quit     exit the program
=#

include(joinpath(@__DIR__, "state.jl"))
include(joinpath(@__DIR__, "fields.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "search.jl"))

using .State, .Fields, .Energy, .Search
using Printf

# ── Move parser ────────────────────────────────────────────────
# Convert "e2e4" or "e7e8q" to a Move by matching against the
# legal move list.  Matching against legal moves is intentional:
# it automatically handles castling and en-passant flags that
# can't be reconstructed from the string alone.
function parse_move(s::String, b::Board)::Union{Move, Nothing}
    length(s) < 4 && return nothing

    from_file = Int(s[1]) - Int('a') + 1
    from_rank = Int(s[2]) - Int('0')
    to_file   = Int(s[3]) - Int('a') + 1
    to_rank   = Int(s[4]) - Int('0')

    (1 ≤ from_file ≤ 8 && 1 ≤ from_rank ≤ 8 &&
     1 ≤ to_file   ≤ 8 && 1 ≤ to_rank   ≤ 8) || return nothing

    # Promotion suffix q/r/b/n — default to queen if the move is a
    # promotion but the user didn't specify.
    promo_char = length(s) ≥ 5 ? lowercase(s[5]) : 'q'
    promo_val  = promo_char == 'r' ? ROOK   :
                 promo_char == 'b' ? BISHOP :
                 promo_char == 'n' ? KNIGHT : QUEEN   # 'q' or default

    for m in generate_moves(b)
        m.from_rank == from_rank && m.from_file == from_file &&
        m.to_rank   == to_rank   && m.to_file   == to_file   || continue
        # For promotions, require the piece to match the suffix (or queen default).
        m.promotion != 0.0 && abs(m.promotion) != promo_val && continue
        return m
    end
    return nothing
end

# ── Game loop ──────────────────────────────────────────────────
# human_color: WHITE (+1) or BLACK (-1).
# max_depth:   how many plies deep the engine searches.
function play(; human_color::Int = WHITE, max_depth::Int = 4)
    b = new_board()
    # best_move() creates a fresh transposition table per call — no reset needed.

    human_side  = human_color  == WHITE ? "White" : "Black"
    engine_side = human_color  == WHITE ? "Black" : "White"

    println()
    println("══════════════════════════════════════════════════")
    println("  FieldEngine — Human vs Computer")
    @printf("  You play %-5s  |  Engine plays %s (depth %d)\n",
            human_side, engine_side, max_depth)
    println("  Moves: e2e4  g1f3  e7e8q   |  help  eval  quit")
    println("══════════════════════════════════════════════════")

    while !is_game_over(b)
        print_board(b)

        if b.turn == human_color
            # ── Human turn ──
            m_chosen = nothing
            while isnothing(m_chosen)
                print("  Your move: ")
                input = strip(readline())

                if input in ("quit", "q", "exit")
                    println("  Goodbye.\n")
                    return
                end

                if input in ("help", "h", "?")
                    legal = sort(generate_moves(b), by = move_to_string)
                    println("  Legal: ", join(move_to_string.(legal), "  "))
                    continue
                end

                if input in ("eval", "e")
                    evaluate_verbose(b)
                    continue
                end

                m_chosen = parse_move(String(input), b)
                if isnothing(m_chosen)
                    println("  Not a legal move.  Type 'help' to see legal moves.")
                end
            end

            apply_move!(b, m_chosen)
            println("  You played: $(move_to_string(m_chosen))\n")

        else
            # ── Engine turn ──
            println("  Engine thinking (depth $max_depth)...")
            m, score = best_move(b; max_depth = max_depth, verbose = true)
            apply_move!(b, m)
            @printf("  Engine plays: %-6s  score %+.3f\n\n",
                    move_to_string(m), score)
        end
    end

    # ── Game over ──
    print_board(b)
    result = game_result(b)
    println("══════════════════════════════════════════════════")
    if result == WHITE
        println("  White wins by checkmate!")
    elseif result == BLACK
        println("  Black wins by checkmate!")
    elseif is_threefold_repetition(b)
        println("  Draw by threefold repetition.")
    elseif b.halfmove ≥ 100
        println("  Draw by the 50-move rule.")
    else
        println("  Draw by stalemate.")
    end
    println("══════════════════════════════════════════════════\n")
end

# ── Entry point ────────────────────────────────────────────────
# Optional command-line args: [white|black] [depth]
#   julia src/play.jl black 5  → human Black, engine depth 5
human_color = WHITE
max_depth   = 4

if length(ARGS) ≥ 1
    lowercase(ARGS[1]) in ("black", "b") && (human_color = BLACK)
end
if length(ARGS) ≥ 2
    max_depth = parse(Int, ARGS[2])
end

play(human_color = human_color, max_depth = max_depth)
