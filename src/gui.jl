#=
gui.jl — Browser-based GUI for FieldEngine.

Starts a local HTTP server and opens the browser automatically.
The frontend (gui.html) renders the board and handles clicks;
this file provides the JSON API it talks to.

Run from the project root:
    julia src/gui.jl              # human White, engine Black, depth 4
    julia src/gui.jl black        # human plays Black
    julia src/gui.jl white 5      # White, engine searches 5 plies

Endpoints:
    GET  /            — serve gui.html
    GET  /state       — board state JSON
    GET  /legal?from= — legal moves for a square, e.g. ?from=e2
    POST /move        — body: "e2e4"; returns updated state JSON
    POST /new_game    — body: {"human":"white","depth":4}
=#

include(joinpath(@__DIR__, "state.jl"))
include(joinpath(@__DIR__, "fields.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "search.jl"))

using .State, .Fields, .Energy, .Search
using HTTP, Printf

# ── Game state ──────────────────────────────────────────────────
mutable struct GameState
    board     ::Board
    human     ::Int       # WHITE or BLACK
    depth     ::Int
    last_move ::String    # "" when no move has been made
    last_score::Float64
    status    ::Symbol    # :playing | :checkmate | :stalemate | :draw50
end

# Module-level ref — one game at a time.
const GAME = Ref{GameState}()

function new_game!(human::Int = WHITE, depth::Int = 4)
    # best_move() creates a fresh transposition table per call — no reset needed.
    g = GameState(new_board(), human, depth, "", 0.0, :playing)
    GAME[] = g
    # If the engine has the first move (human plays Black), make it now.
    g.board.turn != human && engine_move!(g)
    return g
end

# ── Helpers ─────────────────────────────────────────────────────

function update_status!(g::GameState)
    b = g.board
    if b.halfmove >= 100
        g.status = :draw50
    elseif is_checkmate(b)
        g.status = :checkmate
    elseif is_stalemate(b)
        g.status = :stalemate
    else
        g.status = :playing
    end
end

function engine_move!(g::GameState)
    m, score = best_move(g.board; max_depth = g.depth, verbose = false)
    apply_move!(g.board, m)
    g.last_move  = move_to_string(m)
    g.last_score = score
    update_status!(g)
end

# Convert "e2" → {file, rank} and check if valid.
function parse_square(s::AbstractString)
    length(s) < 2 && return nothing, nothing
    file = Int(s[1]) - Int('a') + 1
    rank = Int(s[2]) - Int('0')
    (1 ≤ file ≤ 8 && 1 ≤ rank ≤ 8) || return nothing, nothing
    return rank, file
end

# Match move string "e2e4" or "e7e8q" against legal moves.
# Matching against the legal list handles castling / en-passant flags automatically.
function parse_move_str(s::String, b::Board)::Union{Move, Nothing}
    length(s) < 4 && return nothing
    from_rank, from_file = parse_square(s[1:2])
    to_rank,   to_file   = parse_square(s[3:4])
    (isnothing(from_rank) || isnothing(to_rank)) && return nothing

    # Promotion suffix: q / r / b / n. Default to queen.
    pc = length(s) >= 5 ? lowercase(s[5]) : 'q'
    pv = pc == 'r' ? ROOK : pc == 'b' ? BISHOP : pc == 'n' ? KNIGHT : QUEEN

    for m in generate_moves(b)
        m.from_rank == from_rank && m.from_file == from_file &&
        m.to_rank   == to_rank   && m.to_file   == to_file   || continue
        m.promotion != 0.0 && abs(m.promotion) != pv && continue
        return m
    end
    return nothing
end

# ── JSON serialization ───────────────────────────────────────────
# Hand-rolled to avoid adding JSON3 as a dependency.

function state_json(g::GameState)::String
    b = g.board
    buf = IOBuffer()

    print(buf, "{")
    # pieces: [[rank,file,value],...]
    print(buf, "\"pieces\":[")
    first = true
    for r in 1:8, f in 1:8
        v = b.grid[r, f]
        v == 0.0 && continue
        first || print(buf, ",")
        fv = isfinite(v) ? v : sign(v) * 999.0  # guard against NaN/Inf in JSON
        @printf(buf, "[%d,%d,%g]", r, f, fv)
        first = false
    end
    print(buf, "],")

    turn   = b.turn  == WHITE ? "white" : "black"
    human  = g.human == WHITE ? "white" : "black"
    last   = isempty(g.last_move) ? "null" : "\"$(g.last_move)\""
    chk    = is_in_check(b, b.turn) ? "true" : "false"

    @printf(buf, "\"turn\":\"%s\",\"human\":\"%s\",\"status\":\"%s\",",
            turn, human, g.status)
    score = isfinite(g.last_score) ? g.last_score : 0.0
    @printf(buf, "\"last_move\":%s,\"score\":%.6f,\"depth\":%d,\"in_check\":%s}",
            last, score, g.depth, chk)

    String(take!(buf))
end

function legal_json(g::GameState, from_str::AbstractString)::String
    rank, file = parse_square(from_str)
    isnothing(rank) && return "[]"
    moves = filter(m -> m.from_rank == rank && m.from_file == file,
                   generate_moves(g.board))
    "[" * join(("\"$(move_to_string(m))\"" for m in moves), ",") * "]"
end

# ── Request handlers ─────────────────────────────────────────────

const HTML_BYTES = read(joinpath(@__DIR__, "gui.html"))

function handle_root(::HTTP.Request)
    HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], HTML_BYTES)
end

function handle_state(::HTTP.Request)
    HTTP.Response(200, ["Content-Type" => "application/json"], state_json(GAME[]))
end

function handle_legal(req)
    m = match(r"from=([a-h][1-8])", req.target)
    from = isnothing(m) ? "" : m.captures[1]
    HTTP.Response(200, ["Content-Type" => "application/json"], legal_json(GAME[], from))
end

function handle_move(req)
    g = GAME[]
    g.status != :playing  && return HTTP.Response(409, "Game is over")
    g.board.turn != g.human && return HTTP.Response(409, "Not your turn")

    move_str = strip(String(req.body))
    m = parse_move_str(String(move_str), g.board)
    isnothing(m) && return HTTP.Response(400, "Illegal move: $move_str")

    apply_move!(g.board, m)
    g.last_move  = String(move_str)
    g.last_score = 0.0
    update_status!(g)

    # Engine replies immediately if the game continues.
    g.status == :playing && engine_move!(g)

    HTTP.Response(200, ["Content-Type" => "application/json"], state_json(g))
end

function handle_new_game(req)
    body = String(req.body)
    hm   = match(r"\"human\"\s*:\s*\"(\w+)\"", body)
    dm   = match(r"\"depth\"\s*:\s*(\d+)",    body)
    human = isnothing(hm) ? WHITE : hm.captures[1] == "black" ? BLACK : WHITE
    depth = isnothing(dm) ? 4     : parse(Int, dm.captures[1])
    g = new_game!(human, depth)
    HTTP.Response(200, ["Content-Type" => "application/json"], state_json(g))
end

# ── Router ───────────────────────────────────────────────────────
function router(req)
    t = req.target
    if t == "/" || t == "/index.html"      ; return handle_root(req)
    elseif t == "/state"                   ; return handle_state(req)
    elseif startswith(t, "/legal")         ; return handle_legal(req)
    elseif t == "/move"                    ; return handle_move(req)
    elseif t == "/new_game"                ; return handle_new_game(req)
    end
    HTTP.Response(404, "Not found")
end

# ── Entry point ──────────────────────────────────────────────────
human_color = WHITE
max_depth   = 4

if length(ARGS) >= 1
    lowercase(ARGS[1]) in ("black", "b") && (human_color = BLACK)
end
if length(ARGS) >= 2
    max_depth = parse(Int, ARGS[2])
end

new_game!(human_color, max_depth)

port = 8080
url  = "http://localhost:$port"
println()
println("  FieldEngine GUI  →  $url")
println("  Press Ctrl+C to stop.")
println()

# Open the default browser (best-effort, non-blocking).
try
    if Sys.isapple()
        run(`open $url`, wait = false)
    elseif Sys.iswindows()
        run(`cmd /c start $url`, wait = false)
    elseif Sys.islinux()
        run(`xdg-open $url`, wait = false)
    end
catch
end

HTTP.serve(router, "127.0.0.1", port)
