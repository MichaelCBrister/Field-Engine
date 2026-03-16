#=
optimize.jl — CMA-ES weight tuning via self-play or Stockfish matches.

Finds better evaluation weights by running the engine in repeated games.
Two fitness modes are supported:
  1) selfplay  — candidate vs baseline FieldEngine
  2) stockfish — candidate vs Stockfish (UCI)

The optimizer (CMA-ES) searches the 5-dimensional weight space to maximize
win rate under the selected mode.

This file is the optimizer entry point.  It handles CMA-ES, Stockfish UCI
integration, game loops, and checkpointing.  The search algorithm itself
(negamax, quiescence, TT, buffers) lives in search.jl and is shared with
the interactive engine — "tune what you play."

Run from the project root:
    julia --threads auto src/optimize.jl [depth] [lambda] [generations]
    julia --threads auto src/optimize.jl 5 64 200 --mode stockfish --stockfish bin/stockfish --sf-nodes 50000

Examples:
    julia --threads auto src/optimize.jl 1 32 200    # fast, laptop
    julia --threads auto src/optimize.jl 3 64 100    # c7g.16xlarge (64 vCPUs)

Checkpoint / resume:
    Progress is saved to optimize_checkpoint.txt after every generation.
    If the file exists when you start, the run resumes automatically.
    To start fresh, delete optimize_checkpoint.txt first.
    Redirect output to a file to tail it on a server:
        julia --threads auto src/optimize.jl 3 64 100 > opt.log 2>&1 &
        tail -f opt.log

On a c7g.16xlarge (64 vCPUs): set λ = 64 (or a multiple) so every thread
handles exactly one candidate per generation — perfect load balance.

The five weights being optimized (from energy.jl):
    w[1] = W_MATERIAL    — raw piece charge balance
    w[2] = W_FIELD       — net board influence
    w[3] = W_KING_SAFETY — enemy field near opponent's king
    w[4] = W_TENSION     — field gradient sharpness near kings
    w[5] = W_MOBILITY    — piece activity (reachable squares)
=#

include(joinpath(@__DIR__, "state.jl"))
include(joinpath(@__DIR__, "fields.jl"))
include(joinpath(@__DIR__, "energy.jl"))
include(joinpath(@__DIR__, "search.jl"))

using .State, .Fields, .Energy, .Search
using Printf, Random, LinearAlgebra

# ── JSON helpers (no dependency) ────────────────────────────────
# Minimal JSON serialization for checkpoint files. Avoids pulling in
# the JSON.jl package (slow precompile, unnecessary dependency).
function _json_encode(io::IO, v::Float64)
    if isnan(v); print(io, "null")
    elseif isinf(v); print(io, v > 0 ? "1e308" : "-1e308")
    else @printf(io, "%.10g", v)
    end
end
function _json_encode(io::IO, v::Int)
    print(io, v)
end
function _json_encode(io::IO, v::String)
    print(io, '"', replace(replace(v, '\\' => "\\\\"), '"' => "\\\""), '"')
end
function _json_encode(io::IO, v::Vector{Float64})
    print(io, '[')
    for (i, x) in enumerate(v)
        i > 1 && print(io, ", ")
        _json_encode(io, x)
    end
    print(io, ']')
end
function _json_encode(io::IO, v::Vector{Vector{Float64}})
    print(io, '[')
    for (i, row) in enumerate(v)
        i > 1 && print(io, ", ")
        _json_encode(io, row)
    end
    print(io, ']')
end
function _json_encode(io::IO, d::Dict)
    print(io, "{\n")
    first = true
    for (k, v) in sort(collect(d), by=x->string(x[1]))
        first || print(io, ",\n")
        first = false
        print(io, "  ")
        _json_encode(io, string(k))
        print(io, ": ")
        _json_encode(io, v)
    end
    print(io, "\n}")
end

# ── Baseline weights ─────────────────────────────────────────────
# Starting point for optimization. Pulled from energy.jl so optimizer and
# interactive engine stay in sync by default.
const BASELINE = Float64[
    Energy.W_MATERIAL,
    Energy.W_FIELD,
    Energy.W_KING_SAFETY,
    Energy.W_TENSION,
    Energy.W_MOBILITY,
]

# Rolling baseline for self-play mode. Updated every ROLLING_BASELINE_PERIOD
# generations with the best weights found so far, so the opponent gradually
# gets stronger rather than staying frozen at the initial BASELINE forever.
# Declared as a Ref to a Vector so threads can read it lock-free — Julia's
# Ref assignment is atomic at the pointer level, so readers either see the
# old vector or the new one, never a partial write.
const ROLLING_BASELINE     = Ref{Vector{Float64}}(copy(BASELINE))
const ROLLING_BASELINE_PERIOD = 10   # update every N generations

const DEFAULT_STOCKFISH_PATH = joinpath(@__DIR__, "..", "bin", Sys.iswindows() ? "stockfish.exe" : "stockfish")

struct StockfishConfig
    path::String
    movetime_ms::Int
    skill::Int
    hash_mb::Int
    threads::Int
    nodes::Int              # if > 0, use "go nodes N" instead of "go movetime"
end

mutable struct StockfishSession
    proc::Base.Process
    cfg::StockfishConfig
    linebuf::Vector{UInt8}   # partial-line buffer for timeout-safe reading
    buf_offset::Int          # read start index into linebuf (avoids O(n) deleteat!)
    stdin_fd::Int32          # raw POSIX fd for stdin (bypass libuv writes)
    stdout_fd::Int32         # raw POSIX fd for stdout (bypass libuv reads)
end

const STOCKFISH_CONFIG    = Ref{Union{Nothing, StockfishConfig}}(nothing)
const STOCKFISH_SESSIONS  = Ref{Union{Nothing, StockfishSession}}[]
const _SESSIONS_LOCK      = ReentrantLock()

function _ensure_session_slots!(tid::Int)
    tid <= length(STOCKFISH_SESSIONS) && return
    lock(_SESSIONS_LOCK) do
        while length(STOCKFISH_SESSIONS) < tid
            push!(STOCKFISH_SESSIONS, Ref{Union{Nothing, StockfishSession}}(nothing))
        end
    end
end

function _init_session_slots!()
    n = Threads.nthreads() + Threads.nthreads(:interactive)
    resize!(STOCKFISH_SESSIONS, 0)
    for _ in 1:n
        push!(STOCKFISH_SESSIONS, Ref{Union{Nothing, StockfishSession}}(nothing))
    end
end


# ── Weight constraints ───────────────────────────────────────────
# Keep CMA-ES inside a physically reasonable region so it doesn't "solve"
# the noisy objective by blasting coefficients to extreme values.
# Ranges are intentionally broad around hand-tuned/observed-good weights,
# but tight enough to prevent runaway scales.
const WEIGHT_MIN = Float64[2.0,  -1.5, -1.5, -0.02, -1.5]
const WEIGHT_MAX = Float64[20.0,  1.5,  2.5,  0.02,  2.5]

# Cap CMA-ES step-size growth. σ explosion was causing near-random sampling
# and huge coefficient magnitudes that don't transfer to real play strength.
const CMAES_SIGMA_MAX = 0.5

function clamp_weights(w::Vector{Float64})::Vector{Float64}
    out = copy(w)
    @inbounds for i in eachindex(out)
        out[i] = clamp(out[i], WEIGHT_MIN[i], WEIGHT_MAX[i])
    end
    return out
end

# ── Stockfish / UCI integration ──────────────────────────────────
@inline function piece_to_fen_char(Q::Float64)::Char
    pt = abs(Q)
    c = pt == PAWN   ? 'p' :
        pt == KNIGHT ? 'n' :
        pt == BISHOP ? 'b' :
        pt == ROOK   ? 'r' :
        pt == QUEEN  ? 'q' : 'k'
    return Q > 0 ? uppercase(c) : c
end

@inline function square_to_uci(rank::Int, file::Int)::String
    return string(Char('a' + file - 1), rank)
end

function castling_to_fen(b::Board)::String
    io = IOBuffer()
    b.castling[1] && print(io, 'K')
    b.castling[2] && print(io, 'Q')
    b.castling[3] && print(io, 'k')
    b.castling[4] && print(io, 'q')
    rights = String(take!(io))
    return isempty(rights) ? "-" : rights
end

function board_to_fen(b::Board)::String
    io = IOBuffer()
    for rank in 8:-1:1
        empty_run = 0
        for file in 1:8
            Q = b.grid[rank, file]
            if Q == 0.0
                empty_run += 1
            else
                empty_run > 0 && print(io, empty_run)
                empty_run = 0
                print(io, piece_to_fen_char(Q))
            end
        end
        empty_run > 0 && print(io, empty_run)
        rank > 1 && print(io, '/')
    end

    print(io, ' ', b.turn == WHITE ? 'w' : 'b', ' ', castling_to_fen(b), ' ')
    if b.en_passant == (0, 0)
        print(io, '-')
    else
        print(io, square_to_uci(b.en_passant[1], b.en_passant[2]))
    end
    print(io, ' ', b.halfmove, ' ', b.fullmove)
    return String(take!(io))
end

@inline function promo_from_uci_char(c::Char)::Float64
    lc = lowercase(c)
    lc == 'q' && return QUEEN
    lc == 'r' && return ROOK
    lc == 'b' && return BISHOP
    lc == 'n' && return KNIGHT
    return 0.0
end

function parse_uci_move_from_legal(uci::String, legal_buf::Vector{Move})::Union{Move, Nothing}
    tok = lowercase(strip(uci))
    (length(tok) == 4 || length(tok) == 5) || return nothing

    ff = Int(tok[1]) - Int('a') + 1
    fr = Int(tok[2]) - Int('0')
    tf = Int(tok[3]) - Int('a') + 1
    tr = Int(tok[4]) - Int('0')
    (1 <= ff <= 8 && 1 <= fr <= 8 && 1 <= tf <= 8 && 1 <= tr <= 8) || return nothing

    promo_abs = length(tok) == 5 ? promo_from_uci_char(tok[5]) : 0.0
    if length(tok) == 5 && promo_abs == 0.0
        return nothing
    end

    for m in legal_buf
        m.from_rank == fr && m.from_file == ff &&
        m.to_rank   == tr && m.to_file   == tf || continue

        if m.promotion == 0.0
            promo_abs == 0.0 && return m
            continue
        end

        # If suffix omitted, treat it as queen promotion.
        want = promo_abs == 0.0 ? QUEEN : promo_abs
        abs(m.promotion) == want && return m
    end
    return nothing
end

# ── Pure POSIX I/O layer ────────────────────────────────────────
# ALL Stockfish I/O bypasses Julia's libuv event loop entirely.
#
# Why: Julia's open(Cmd, "r+") creates child processes via uv_spawn,
# which binds ALL pipe handles to thread 1's libuv event loop.
# Under @threads :static with 32 threads:
#   - flush(proc) for writes needs thread 1 to pump its event loop
#   - readavailable(pipe) for reads needs thread 1 to pump its event loop
#   - When 32 threads all need thread 1 simultaneously → deadlock
#
# Fix: Extract raw POSIX fds at session creation time. Use POSIX write()
# for sends and POSIX poll()+read() for receives. No libuv, no event
# loop, no cross-thread dependency. Each thread operates on its own
# Stockfish process's fds independently.

const POSIX_READ_BUF_SIZE = 4096

# Send a UCI command via POSIX write — bypasses libuv entirely.
@inline function uci_send!(sess::StockfishSession, cmd::String)
    data = cmd * "\n"
    buf = Vector{UInt8}(codeunits(data))
    total = length(buf)
    written = 0
    while written < total
        n = ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
                  sess.stdin_fd, pointer(buf, written + 1), total - written)
        if n < 0
            errno = Base.Libc.errno()
            errno == 4 && continue  # EINTR — retry
            throw(ErrorException("POSIX write() to Stockfish stdin failed: errno=$errno"))
        end
        written += n
    end
end

# Legacy overload for initial handshake (before session is fully constructed).
@inline function uci_send_raw!(fd::Int32, cmd::String)
    data = cmd * "\n"
    buf = Vector{UInt8}(codeunits(data))
    total = length(buf)
    written = 0
    while written < total
        n = ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
                  fd, pointer(buf, written + 1), total - written)
        if n < 0
            errno = Base.Libc.errno()
            errno == 4 && continue  # EINTR
            throw(ErrorException("POSIX write() to Stockfish stdin failed: errno=$errno"))
        end
        written += n
    end
end

Base.@kwdef struct PollFD
    fd::Int32
    events::Int16
    revents::Int16
end
const POLL_IN = Int16(0x0001)
const POLL_ERR = Int16(0x0008)
const POLL_HUP = Int16(0x0010)
const POLL_NVAL = Int16(0x0020)

@inline function wait_fd_readable!(fd::Int32, timeout_ms::Int)::Bool
    timeout_ms <= 0 && return false
    pfd = Ref(PollFD(fd=fd, events=POLL_IN, revents=Int16(0)))
    ret = ccall(:poll, Int32, (Ptr{PollFD}, UInt64, Int32), pfd, 1, timeout_ms)
    ret < 0 && throw(ErrorException("poll() failed: errno=$(Base.Libc.errno())"))
    ret == 0 && return false
    revents = pfd[].revents
    return (revents & (POLL_IN | POLL_ERR | POLL_HUP | POLL_NVAL)) != 0
end

# POSIX read — bypasses libuv entirely. Returns bytes read, 0 on EAGAIN.
@inline function posix_read!(fd::Int32, buf::Vector{UInt8})::Int
    n = ccall(:read, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
              fd, buf, length(buf))
    if n < 0
        errno = Base.Libc.errno()
        (errno == 11 || errno == 4) && return 0  # EAGAIN or EINTR
        throw(ErrorException("POSIX read() failed: errno=$errno"))
    end
    return Int(n)
end

# Timeout-safe readline using pure POSIX I/O.
# Uses poll() for instant wakeup + read() to consume data.
# No libuv involvement — cannot deadlock under @threads :static.
#
# buf_offset tracks the read start index to avoid O(n) deleteat! on every
# line. The buffer is compacted only when the offset exceeds half its length.
function readline_timed!(fd::Int32, linebuf::Vector{UInt8},
                         timeout_s::Float64;
                         buf_offset::Ref{Int} = Ref(1))::String
    deadline = time() + timeout_s
    readbuf = Vector{UInt8}(undef, POSIX_READ_BUF_SIZE)
    while true
        # Search for newline from current offset
        off = buf_offset[]
        idx = nothing
        for i in off:length(linebuf)
            if @inbounds linebuf[i] == UInt8('\n')
                idx = i
                break
            end
        end
        if idx !== nothing
            line = String(@view linebuf[off:idx-1])
            buf_offset[] = idx + 1
            # Compact when offset exceeds half the buffer
            if buf_offset[] > 1 && buf_offset[] > length(linebuf) ÷ 2 + 64
                deleteat!(linebuf, 1:buf_offset[]-1)
                buf_offset[] = 1
            end
            return rstrip(line, '\r')
        end

        # Wait for data at the OS level (instant wakeup, no polling overhead)
        rem_ms = Int(floor((deadline - time()) * 1000.0))
        rem_ms <= 0 && throw(ErrorException("Stockfish readline timed out"))
        if !wait_fd_readable!(fd, min(rem_ms, 500))
            continue  # poll timed out, loop back to check deadline
        end

        # poll() confirmed data — read directly via POSIX.
        n = posix_read!(fd, readbuf)
        if n > 0
            append!(linebuf, @view readbuf[1:n])
        end
    end
end

function uci_wait_for_prefix!(stdout_fd::Int32, prefix::String;
                              timeout_s::Float64 = 30.0,
                              linebuf::Vector{UInt8} = UInt8[],
                              buf_offset::Ref{Int} = Ref(1))::String
    deadline = time() + timeout_s
    while true
        rem_s = deadline - time()
        rem_s <= 0.0 && error("Stockfish timeout while waiting for '$prefix'")
        line = try
            readline_timed!(stdout_fd, linebuf, rem_s; buf_offset=buf_offset)
        catch err
            if occursin("timed out", string(err))
                error("Stockfish timeout while waiting for '$prefix'")
            end
            rethrow()
        end
        startswith(line, prefix) && return line
    end
end

function resolve_stockfish_path(path::String)::String
    stripped = strip(path)
    isempty(stripped) && error("Stockfish path is empty.")

    if stripped == "stockfish"
        found = Sys.which("stockfish")
        isnothing(found) && error(
            "Stockfish not found in PATH. " *
            "Install it or pass --stockfish /absolute/path/to/stockfish."
        )
        return found
    end

    resolved = abspath(stripped)
    isfile(resolved) || error("Stockfish binary not found at: $resolved")
    return resolved
end

function create_stockfish_session(cfg::StockfishConfig; init_timeout_s::Float64 = 30.0)::StockfishSession
    proc = try
        open(`$(cfg.path)`, "r+")
    catch err
        error("Failed to launch Stockfish at $(cfg.path): $(sprint(showerror, err))")
    end

    # Extract raw POSIX fds ONCE — all subsequent I/O bypasses libuv entirely.
    # This prevents the cross-thread libuv event-loop deadlock that occurs
    # when 32 @threads :static workers all need thread 1 to pump events.
    stdin_fd  = reinterpret(Int32, Base._fd(proc.in))
    stdout_fd = reinterpret(Int32, Base._fd(proc.out))

    linebuf = UInt8[]
    off = Ref(1)
    uci_send_raw!(stdin_fd, "uci")
    uci_wait_for_prefix!(stdout_fd, "uciok"; timeout_s=init_timeout_s, linebuf=linebuf, buf_offset=off)

    uci_send_raw!(stdin_fd, "setoption name Threads value $(cfg.threads)")
    uci_send_raw!(stdin_fd, "setoption name Hash value $(cfg.hash_mb)")
    uci_send_raw!(stdin_fd, "setoption name Skill Level value $(cfg.skill)")
    uci_send_raw!(stdin_fd, "isready")
    uci_wait_for_prefix!(stdout_fd, "readyok"; timeout_s=init_timeout_s, linebuf=linebuf, buf_offset=off)

    return StockfishSession(proc, cfg, linebuf, off[], stdin_fd, stdout_fd)
end

function terminate_stockfish_process!(proc::Base.Process)
    try
        stdin_fd = reinterpret(Int32, Base._fd(proc.in))
        uci_send_raw!(stdin_fd, "quit")
    catch
    end

    # Allow graceful exit first, then force-kill if needed.
    t0 = time()
    while process_running(proc) && (time() - t0) < 1.0
        sleep(0.01)
    end
    if process_running(proc)
        try
            kill(proc)
        catch
        end
    end
    try
        close(proc)
    catch
    end
    try
        wait(proc)
    catch
    end
end

function close_stockfish_sessions!()
    for sess_ref in STOCKFISH_SESSIONS
        sess = sess_ref[]
        sess === nothing && continue
        terminate_stockfish_process!(sess.proc)
        sess_ref[] = nothing
    end
end

function close_stockfish_session!(tid::Int)
    _ensure_session_slots!(tid)
    sess = STOCKFISH_SESSIONS[tid][]
    sess === nothing && return
    terminate_stockfish_process!(sess.proc)
    STOCKFISH_SESSIONS[tid][] = nothing
end

function configure_stockfish!(; path::String,
                               movetime_ms::Int,
                               skill::Int,
                               hash_mb::Int,
                               threads::Int,
                               nodes::Int = 0)
    resolved = resolve_stockfish_path(path)
    cfg = StockfishConfig(
        resolved,
        max(movetime_ms, 1),
        clamp(skill, 0, 20),
        max(hash_mb, 1),
        max(threads, 1),
        max(nodes, 0),
    )
    close_stockfish_sessions!()
    STOCKFISH_CONFIG[] = cfg
    return cfg
end

function get_stockfish_session!(tid::Int; init_timeout_s::Float64 = 30.0)::StockfishSession
    _ensure_session_slots!(tid)
    sess = STOCKFISH_SESSIONS[tid][]
    # Recreate if missing or process has died
    if sess === nothing || !process_running(sess.proc)
        cfg = STOCKFISH_CONFIG[]
        cfg === nothing && error("Stockfish mode requested, but Stockfish is not configured.")
        sess = create_stockfish_session(cfg; init_timeout_s=init_timeout_s)
        STOCKFISH_SESSIONS[tid][] = sess
    end
    return sess
end

function reset_stockfish_game!(sess::StockfishSession)
    empty!(sess.linebuf)    # clear partial data from previous game
    sess.buf_offset = 1
    uci_send!(sess, "ucinewgame")
    uci_send!(sess, "isready")
    off = Ref(sess.buf_offset)
    uci_wait_for_prefix!(sess.stdout_fd, "readyok"; linebuf=sess.linebuf, buf_offset=off)
    sess.buf_offset = off[]
end

@inline function stockfish_timeout_s(sess::StockfishSession)::Float64
    # Keep timeouts tight enough to fail fast on wedged child processes, but
    # with headroom for host-level scheduling variance under heavy parallel load.
    if sess.cfg.nodes > 0
        # Node-based: can't predict wall time from node count alone.
        # Use a generous fixed ceiling — if Stockfish takes longer than
        # this on any reasonable node budget, the process is wedged.
        return 30.0
    end
    t = 1.0 + 30.0 * (sess.cfg.movetime_ms / 1000.0)
    return clamp(t, 4.0, 20.0)
end

function stockfish_bestmove!(sess::StockfishSession, move_history::Vector{String})::String
    # Send position using move history instead of FEN.
    # This avoids regenerating the full FEN on every move (~80× per game)
    # and lets Stockfish reuse NNUE accumulators incrementally.
    if isempty(move_history)
        uci_send!(sess, "position startpos")
    else
        uci_send!(sess, "position startpos moves " * join(move_history, ' '))
    end

    # Fixed-node mode removes timing noise from EC2 background load, JIT
    # variance, and thread scheduling — the same node budget always produces
    # the same search tree, making fitness measurements reproducible.
    if sess.cfg.nodes > 0
        uci_send!(sess, "go nodes $(sess.cfg.nodes)")
    else
        uci_send!(sess, "go movetime $(sess.cfg.movetime_ms)")
    end

    # Use readline_timed! so one hung child cannot stall a whole generation.
    # UCI "info" lines are drained until we see "bestmove".
    timeout_s = stockfish_timeout_s(sess)
    deadline = time() + timeout_s
    off_ref = Ref(sess.buf_offset)
    while true
        rem_s = deadline - time()
        rem_s <= 0.0 && (sess.buf_offset = off_ref[]; throw(ErrorException("Stockfish timeout — process may have hung")))
        line = try
            readline_timed!(sess.stdout_fd, sess.linebuf, rem_s; buf_offset=off_ref)
        catch err
            sess.buf_offset = off_ref[]
            if occursin("timed out", string(err))
                throw(ErrorException("Stockfish timeout — process may have hung"))
            end
            rethrow()
        end

        startswith(line, "bestmove ") || continue
        sess.buf_offset = off_ref[]
        # Extract move directly — "bestmove " is exactly 9 chars.
        # Avoids allocating a Vector{SubString} on every Stockfish move.
        space2 = findnext(' ', line, 10)
        return isnothing(space2) ? SubString(line, 10) : SubString(line, 10, space2 - 1)
    end
end

# Obtain and parse Stockfish's move with one auto-retry on IO/parse failure.
# Returns `nothing` when no safe move could be recovered (caller treats as draw).
function stockfish_move_from_legal!(tid::Int, b::Board, legal_buf::Vector{Move},
                                    move_history::Vector{String};
                                    max_attempts::Int = 2)::Union{Move, Nothing}
    local last_err::String = ""
    local last_uci::String = ""
    for attempt in 1:max_attempts
        sess = get_stockfish_session!(tid)
        try
            uci = stockfish_bestmove!(sess, move_history)
            last_uci = uci
            uci in ("(none)", "0000") && return nothing
            parsed = parse_uci_move_from_legal(uci, legal_buf)
            parsed !== nothing && return parsed
            last_err = "illegal move token for current position"
        catch err
            last_err = sprint(showerror, err)
        end
        # Reset this thread's engine process and retry from a fresh UCI state.
        close_stockfish_session!(tid)
        attempt < max_attempts && continue
    end
    @printf(stderr,
            "  [warn] Stockfish move recovery failed on thread %d: bestmove='%s' err='%s'\n",
            tid, last_uci, last_err)
    return nothing
end

_init_session_slots!()
atexit(close_stockfish_sessions!)

# ── Checkpoint I/O ───────────────────────────────────────────────
# Saves full CMA-ES state to a plain-text file after every generation.
# Written atomically (tmp → rename) so an interrupt never leaves a
# half-written file that would corrupt a subsequent resume.
function save_checkpoint(path::String, gen::Int,
                         best_f::Float64, best_x::Vector{Float64},
                         rolling_baseline::Vector{Float64},
                         m::Vector{Float64}, σ::Float64,
                         C::Matrix{Float64},
                         pc::Vector{Float64}, ps::Vector{Float64})
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "gen ",              gen)
        println(io, "best_f ",           best_f)
        println(io, "best_x ",           join(best_x,          " "))
        println(io, "rolling_baseline ", join(rolling_baseline, " "))
        println(io, "m ",                join(m,                " "))
        println(io, "sigma ",            σ)
        println(io, "pc ",               join(pc,               " "))
        println(io, "ps ",               join(ps,               " "))
        for i in 1:size(C, 1)
            println(io, "C_$i ", join(C[i, :], " "))
        end
    end
    mv(tmp, path, force=true)   # atomic on POSIX filesystems
end

# Reads checkpoint saved by save_checkpoint. Returns a NamedTuple with
# the full state, or nothing if the file doesn't exist or is malformed.
function load_checkpoint(path::String)
    isfile(path) || return nothing
    d = Dict{String, String}()
    try
        open(path) do io
            for line in eachline(io)
                parts = split(line, " ", limit=2)
                length(parts) == 2 && (d[strip(parts[1])] = strip(parts[2]))
            end
        end
        haskey(d, "gen") || return nothing

        parse_vec = s -> parse.(Float64, split(s))

        gen    = parse(Int,     d["gen"])
        best_f = parse(Float64, d["best_f"])
        best_x = parse_vec(d["best_x"])
        # rolling_baseline is optional for backwards-compat with old checkpoints;
        # fall back to best_x so resume uses the best weights seen so far.
        rolling_baseline = haskey(d, "rolling_baseline") ?
            parse_vec(d["rolling_baseline"]) : copy(best_x)
        m      = parse_vec(d["m"])
        σ      = parse(Float64, d["sigma"])
        pc     = parse_vec(d["pc"])
        ps     = parse_vec(d["ps"])
        n      = length(m)
        C      = zeros(Float64, n, n)
        for i in 1:n
            C[i, :] = parse_vec(d["C_$i"])
        end
        return (gen=gen, best_f=best_f, best_x=best_x,
                rolling_baseline=rolling_baseline,
                m=m, σ=σ, C=C, pc=pc, ps=ps)
    catch
        return nothing   # malformed file — start fresh
    end
end

# ── JSON checkpoint (every N generations) ─────────────────────────
# Saves best params + stats to a human-readable JSON file.
# Complements the per-gen plain-text checkpoint with a richer snapshot
# at configurable intervals (default: every 50 gens).
const JSON_CHECKPOINT_INTERVAL = 50

function save_json_checkpoint(path::String, gen::Int,
                               best_f::Float64, best_x::Vector{Float64},
                               σ::Float64, m::Vector{Float64},
                               fitness_history::Vector{Float64},
                               gen_times::Vector{Float64})
    tmp = path * ".tmp"
    open(tmp, "w") do io
        d = Dict{String, Any}(
            "generation"       => gen,
            "best_fitness"     => best_f,
            "best_weights"     => best_x,
            "sigma"            => σ,
            "mean"             => m,
            "fitness_history"  => fitness_history,
            "gen_times_s"      => gen_times,
            "avg_gen_time_s"   => isempty(gen_times) ? 0.0 : sum(gen_times) / length(gen_times),
            "total_time_s"     => isempty(gen_times) ? 0.0 : sum(gen_times),
        )
        _json_encode(io, d)
        println(io)
    end
    mv(tmp, path, force=true)
end

# ── Single self-play game ────────────────────────────────────────
# Each game gets its own fresh TT (no cross-game pollution).
# Field is initialized once and maintained incrementally through play.
function play_game(w_white::Vector{Float64}, w_black::Vector{Float64};
                   depth::Int = 1,
                   max_moves::Int = 50)::Float64
    b   = new_board()
    tid = Threads.threadid()
    ensure_ply_buffers!(tid, 2)

    field      = FIELD_BUFS[tid]
    legal_buf  = LEGAL_BUFS[tid][1]
    pseudo_buf = PSEUDO_BUFS[tid][1]
    from_buf   = FROM_SLIDERS[tid]
    to_buf     = TO_SLIDERS[tid]
    seen       = FROM_SEEN[tid]

    # Fresh transposition table for this game
    tt = new_tt(1 << 18)  # 256K entries — better hit rate at depth 5+

    Fields.compute_total_field!(field, b)
    move_count = 0

    for _ in 1:max_moves
        b.halfmove >= 100 && return 0.0
        is_repetition(b) && return 0.0

        State.generate_moves!(legal_buf, b, pseudo_buf)
        if isempty(legal_buf)
            is_in_check(b, b.turn) && return b.turn == WHITE ? -1.0 : +1.0
            return 0.0
        end

        w = b.turn == WHITE ? w_white : w_black
        best_m = choose_move!(b, w, depth, field, tt, tid, legal_buf)
        move_count += 1
        apply_with_field!(field, b, best_m, from_buf, to_buf, seen)
    end

    # At the move limit the game is undecided but not a true draw.
    # Score from White's perspective using BOTH weight sets symmetrically:
    #   tanh((white_eval - black_eval) * k)
    # This is positive when White is ahead (from its own weights) and negative
    # when Black is ahead (from its own weights).  Because play_single_game
    # negates the result when the candidate is Black, the sign is always from
    # the candidate's perspective regardless of color — fixing the asymmetry
    # that arose when only w_white was used here.
    raw_eval = (eval_w(b, w_white, field) - eval_w(b, w_black, field)) * 0.075
    length_factor = 1.0 - 0.3 * (move_count / max_moves)
    return tanh(raw_eval) * length_factor
end

# Candidate vs Stockfish game. Returns score from candidate perspective.
function play_game_vs_stockfish(w_cand::Vector{Float64}, cand_as_white::Bool;
                                depth::Int = 1,
                                max_moves::Int = 80)::Float64
    b          = new_board()
    cand_color = cand_as_white ? WHITE : BLACK
    tid        = Threads.threadid()
    ensure_ply_buffers!(tid, 2)

    field      = FIELD_BUFS[tid]
    legal_buf  = LEGAL_BUFS[tid][1]
    pseudo_buf = PSEUDO_BUFS[tid][1]
    from_buf   = FROM_SLIDERS[tid]
    to_buf     = TO_SLIDERS[tid]
    seen       = FROM_SEEN[tid]
    tt         = new_tt()

    # Move history for UCI "position startpos moves ..." — avoids
    # regenerating a FEN string on every Stockfish move.
    move_history = String[]
    sizehint!(move_history, 160)

    # Session startup and game reset can throw if the Stockfish binary is
    # missing, the process fails to launch, or the pipe is broken.  Wrap both
    # calls so a per-game failure is scored as a draw (0.0) rather than
    # bubbling into a TaskFailedException that aborts the whole generation.
    sess = try
        s = get_stockfish_session!(tid)
        reset_stockfish_game!(s)
        s
    catch err
        @warn "Stockfish session error (scoring game as draw)" tid exception=err
        return 0.0
    end

    Fields.compute_total_field!(field, b)
    move_count = 0

    for _ in 1:max_moves
        b.halfmove >= 100 && return 0.0
        is_repetition(b) && return 0.0

        State.generate_moves!(legal_buf, b, pseudo_buf)
        if isempty(legal_buf)
            if is_in_check(b, b.turn)
                winner = -b.turn
                return winner == cand_color ? 1.0 : -1.0
            end
            return 0.0
        end

        move = if b.turn == cand_color
            choose_move!(b, w_cand, depth, field, tt, tid, legal_buf)
        else
            parsed = stockfish_move_from_legal!(tid, b, legal_buf, move_history)
            parsed === nothing && return 0.0
            parsed
        end

        push!(move_history, State.move_to_string(move))
        move_count += 1
        apply_with_field!(field, b, move, from_buf, to_buf, seen)
    end

    # Same soft-score logic as play_game — return final eval from the
    # candidate's perspective rather than a flat 0.0 for timed-out games.
    # Float64(cand_color) is +1 for White, -1 for Black, which flips the
    # White-perspective eval into the candidate's perspective.
    #
    # Game-length discount: positions reached quickly with a clear advantage
    # score higher than drawn-out games, reducing fitness noise for CMA-ES.
    raw_eval = Float64(cand_color) * eval_w(b, w_cand, field) * 0.15
    length_factor = 1.0 - 0.3 * (move_count / max_moves)
    return tanh(raw_eval) * length_factor
end

# ── Single game helper ───────────────────────────────────────────
# Plays one game and returns score from the candidate's perspective.
# as_white=true  → candidate plays White, result is +1/0/-1 for candidate
# as_white=false → candidate plays Black, result is flipped accordingly
function play_single_game(w_cand::Vector{Float64}, as_white::Bool;
                          depth::Int = 1,
                          mode::Symbol = :stockfish)::Float64
    if mode === :selfplay
        # Use the rolling baseline as opponent so the bar rises as we improve.
        # ROLLING_BASELINE[] is updated every ROLLING_BASELINE_PERIOD gens in cmaes().
        baseline = ROLLING_BASELINE[]
        return as_white ? play_game(w_cand, baseline; depth=depth) :
                          -play_game(baseline, w_cand; depth=depth)
    elseif mode === :stockfish
        return play_game_vs_stockfish(w_cand, as_white; depth=depth)
    else
        error("Unknown evaluation mode: $mode")
    end
end

# ── Batch fitness function ────────────────────────────────────────
# Evaluates an entire population of candidates in parallel at the GAME level.
#
# Instead of one thread per candidate (each running n_pairs*2 games
# sequentially), we flatten all λ × n_pairs × 2 games into a single pool
# and dispatch them across all available threads. Idle threads immediately
# pick up the next game rather than waiting for a slow candidate to finish.
#
# On a c7g.8xlarge (32 vCPUs) with λ=32 and n_pairs=2 this gives 128
# independent tasks — far better load balance than 32 tasks of unequal length.
function batch_evaluate(candidates::Vector{Vector{Float64}};
                        depth::Int = 1,
                        n_pairs::Int = 2,
                        mode::Symbol = :stockfish,
                        game_workers::Int = Threads.nthreads(),
                        progress_every::Int = 0)::Vector{Float64}
    λ          = length(candidates)
    n_per      = 2 * n_pairs                          # games per candidate
    n_total    = λ * n_per                            # total game pool
    raw        = Vector{Float64}(undef, n_total)      # flat results buffer

    n_workers = clamp(game_workers, 1, max(1, n_total))
    step = if progress_every > 0
        progress_every
    elseif mode === :stockfish
        max(1, n_total ÷ 16)
    else
        0
    end

    next_idx = Threads.Atomic{Int}(1)
    done_ctr = Threads.Atomic{Int}(0)
    t0 = time()
    print_lock = ReentrantLock()

    # Pull-scheduling via atomic next_idx keeps work balanced. We use :static
    # so each worker task stays thread-affine; this is required because search
    # and Stockfish sessions use thread-local buffers/process handles.
    Threads.@threads :static for _ in 1:n_workers
        while true
            idx = Threads.atomic_add!(next_idx, 1)
            idx > n_total && break

            cand_i   = (idx - 1) ÷ n_per + 1          # which candidate (1-based)
            game_j   = (idx - 1) % n_per + 1          # which game for that candidate
            as_white = isodd(game_j)                   # alternate colors each game
            raw[idx] = play_single_game(candidates[cand_i], as_white;
                                        depth=depth, mode=mode)

            done = Threads.atomic_add!(done_ctr, 1) + 1
            if step > 0 && (done == n_total || done % step == 0)
                elapsed = time() - t0
                rate = done / max(elapsed, 1.0e-9)
                eta = (n_total - done) / max(rate, 1.0e-9)
                lock(print_lock) do
                    @printf("    progress %4d/%d (%.1f%%)  elapsed=%6.1fs  eta=%6.1fs\n",
                            done, n_total, 100.0 * done / n_total, elapsed, eta)
                    flush(stdout)
                end
            end
        end
    end

    # Aggregate: sum each candidate's games and normalize to [-1, 1].
    fitness = Vector{Float64}(undef, λ)
    for i in 1:λ
        start      = (i - 1) * n_per + 1
        fitness[i] = sum(raw[start : start + n_per - 1]) / n_per
    end
    return fitness
end

# ── Rank-based fitness shaping ─────────────────────────────────────
# Replaces raw fitness with log-rank utilities, making CMA-ES invariant
# to monotone transformations of the fitness function. This dramatically
# improves robustness to noisy fitness signals (e.g., 4-game win rates).
# Standard approach from Hansen's CMA-ES tutorial.
function rank_fitness_shaping(fitness::Vector{Float64})::Vector{Float64}
    λ = length(fitness)
    order = sortperm(fitness, rev=true)  # best first
    shaped = similar(fitness)
    for (rank, idx) in enumerate(order)
        # log-linear utilities: best candidate gets highest value
        shaped[idx] = max(0.0, log(λ / 2 + 1.0) - log(Float64(rank)))
    end
    # Normalize to sum to 1 so downstream w_rec weighting stays correct
    s = sum(shaped)
    s > 0.0 && (shaped ./= s)
    return shaped
end

# ── CMA-ES ───────────────────────────────────────────────────────
# Covariance Matrix Adaptation Evolution Strategy.
# Saves a checkpoint after every generation and resumes from one if found.
#
# Parameters:
#   batch_f         — fitness function over a population: Vector{Vector} → Vector{Float64}
#   x0              — starting weight vector
#   σ0              — initial step size
#   C0              — initial covariance (scaled to x0's magnitudes)
#   λ               — population size; use a multiple of nthreads() for best
#                     CPU utilization (e.g. λ=64 on c7g.16xlarge)
#   max_gen         — total generations to run
#   rng             — seeded RNG for reproducibility
#   checkpoint_path — file to save/load progress; "" disables checkpointing
function cmaes(batch_f, x0::Vector{Float64};
               σ0::Float64 = 0.1,
               C0::Union{Matrix{Float64}, Nothing} = nothing,
               λ::Int = 8,
               max_gen::Int = 30,
               rng::AbstractRNG = Random.GLOBAL_RNG,
               checkpoint_path::String = "optimize_checkpoint.txt")::Vector{Float64}

    n = length(x0)
    μ = λ ÷ 2

    raw_w = [log(μ + 0.5) - log(Float64(i)) for i in 1:μ]
    w_rec = raw_w / sum(raw_w)
    μeff  = 1.0 / sum(w_rec .^ 2)

    cs   = (μeff + 2.0) / (n + μeff + 5.0)
    ds   = 1.0 + 2.0 * max(0.0, sqrt((μeff - 1.0) / (n + 1.0)) - 1.0) + cs
    chiN = sqrt(Float64(n)) * (1.0 - 1.0/(4n) + 1.0/(21*n^2))

    c1 = 2.0 / ((n + 1.3)^2 + μeff)
    cμ = min(1.0 - c1, 2.0 * (μeff - 2.0 + 1.0/μeff) / ((n + 2.0)^2 + μeff))
    cc = (4.0 + μeff/n) / (n + 4.0 + 2.0*μeff/n)

    # ── Initialize or resume state ───────────────────────────────
    gen_start = 1
    m      = copy(x0)
    σ      = σ0
    C      = isnothing(C0) ? Matrix{Float64}(I, n, n) : copy(C0)
    pc     = zeros(n)
    ps     = zeros(n)
    best_x = copy(x0)
    best_f = -Inf  # computed below only when starting fresh

    if checkpoint_path != ""
        ckpt = load_checkpoint(checkpoint_path)
        if !isnothing(ckpt) && ckpt.gen < max_gen
            gen_start = ckpt.gen + 1
            best_f    = ckpt.best_f
            best_x    = ckpt.best_x
            m         = ckpt.m
            σ         = ckpt.σ
            C         = ckpt.C
            pc        = ckpt.pc
            ps        = ckpt.ps
            # Restore the rolling baseline so resumed self-play evaluates
            # against the same opponent as before the interruption.
            ROLLING_BASELINE[] = copy(ckpt.rolling_baseline)
            @printf("  Resuming from checkpoint: gen %d/%d  best_f=%+.4f\n\n",
                    ckpt.gen, max_gen, best_f)
        elseif !isnothing(ckpt) && ckpt.gen >= max_gen
            println("  Checkpoint shows run already completed (gen $(ckpt.gen)/$(max_gen)).")
            println("  Delete $checkpoint_path to start fresh.\n")
            return clamp_weights(ckpt.best_x)
        else
            best_f = batch_f([x0])[1]
            @printf("  Starting CMA-ES: n=%d  λ=%d  μ=%d  threads=%d  σ0=%.3f  max_gen=%d\n",
                    n, λ, μ, Threads.nthreads(), σ0, max_gen)
            @printf("  Baseline fitness: %+.4f\n", best_f)
            @printf("  Checkpoint: %s\n\n", checkpoint_path)
        end
    else
        best_f = batch_f([x0])[1]
        @printf("  Starting CMA-ES: n=%d  λ=%d  μ=%d  threads=%d  σ0=%.3f  max_gen=%d\n",
                n, λ, μ, Threads.nthreads(), σ0, max_gen)
        @printf("  Baseline fitness: %+.4f\n\n", best_f)
    end

    # ── Early stopping state ──────────────────────────────────────
    # Track best_f over a rolling window to detect convergence plateaus.
    fitness_history = Float64[]
    early_stop_window = 15
    early_stop_threshold = 0.005

    # ── Plateau detection ────────────────────────────────────────
    # Separate from early stopping: warns when best_f hasn't improved
    # by > 0.001 for 30 consecutive generations. Does NOT stop — just
    # logs a warning so the operator can decide whether to intervene.
    PLATEAU_WINDOW    = 30
    PLATEAU_THRESHOLD = 0.001
    plateau_warned = false     # only warn once per plateau

    # ── Per-generation timing ────────────────────────────────────
    gen_times = Float64[]

    # ── JSON checkpoint path ─────────────────────────────────────
    json_checkpoint_path = if checkpoint_path != ""
        replace(checkpoint_path, r"\.\w+$" => "") * "_snapshot.json"
    else
        ""
    end

    for gen in gen_start:max_gen
        gen_t0 = time()

        # ── Sample λ candidates ──────────────────────────────────
        L = try
            cholesky(Symmetric(C + 1e-10 * I)).L
        catch
            E = eigen(Symmetric(C))
            E.values .= max.(E.values, 1e-8)
            C = E.vectors * Diagonal(E.values) * E.vectors'
            cholesky(Symmetric(C)).L
        end
        # Antithetic sampling: evaluate +/-z pairs around m.
        # This lowers Monte Carlo noise and improves sample efficiency.
        n_pairs_ant = λ ÷ 2
        n_seeds = n_pairs_ant + (isodd(λ) ? 1 : 0)
        seeds = rand(rng, UInt64, n_seeds)
        samples = Vector{Vector{Float64}}(undef, λ)

        sidx = 1
        for i in 1:n_pairs_ant
            z = L * randn(MersenneTwister(seeds[sidx]), n)
            sidx += 1
            j = 2i - 1
            samples[j]     = m + σ * z
            samples[j + 1] = m - σ * z
        end
        if isodd(λ)
            z = L * randn(MersenneTwister(seeds[sidx]), n)
            samples[end] = m + σ * z
        end

        # ── Evaluate fitness in parallel at game level ───────────
        # batch_f dispatches every game across all candidates as independent
        # tasks — threads never idle waiting for a slow candidate to finish.
        clamped = [clamp_weights(samples[i]) for i in 1:λ]
        fitness = batch_f(clamped)

        # ── Rank-based fitness shaping ─────────────────────────────
        # Replace noisy raw win rates with log-rank utilities so CMA-ES
        # can find a gradient even when per-candidate variance is high.
        raw_fitness = fitness
        fitness = rank_fitness_shaping(raw_fitness)

        # ── Sort and track global best ───────────────────────────
        order = sortperm(fitness, rev=true)
        if raw_fitness[order[1]] > best_f
            best_f = raw_fitness[order[1]]
            best_x = copy(samples[order[1]])
        end

        # ── Update mean ──────────────────────────────────────────
        # Use UNCLAMPED samples for mean/covariance updates. Clamping
        # distorts the search distribution and corrupts the covariance
        # estimate. Only the fitness evaluation (line above) sees clamped weights.
        m_old = copy(m)
        m     = sum(w_rec[k] * samples[order[k]] for k in 1:μ)
        step  = (m - m_old) / σ

        # ── Update evolution paths ───────────────────────────────
        invsqrtC = try
            inv(cholesky(Symmetric(C + 1e-10 * I)).L)'
        catch
            # Covariance degenerated — recondition by flooring eigenvalues.
            E = eigen(Symmetric(C))
            E.values .= max.(E.values, 1e-8)
            C = E.vectors * Diagonal(E.values) * E.vectors'
            inv(cholesky(Symmetric(C)).L)'
        end
        ps = (1.0 - cs) * ps + sqrt(cs * (2.0 - cs) * μeff) * invsqrtC * step

        hs = norm(ps) / sqrt(1.0 - (1.0 - cs)^(2.0*(gen + 1.0))) / chiN <
             (1.4 + 2.0/(n + 1.0)) ? 1.0 : 0.0

        pc = (1.0 - cc) * pc + hs * sqrt(cc * (2.0 - cc) * μeff) * step

        # ── Update covariance ────────────────────────────────────
        art = [(samples[order[k]] - m_old) / σ for k in 1:μ]
        C = (1.0 - c1 - cμ) * C +
            c1 * (pc * pc' + (1.0 - hs) * cc * (2.0 - cc) * C) +
            cμ * sum(w_rec[k] * art[k] * art[k]' for k in 1:μ)

        # ── Update step size ─────────────────────────────────────
        σ = σ * exp((cs / ds) * (norm(ps) / chiN - 1.0))
        σ = clamp(σ, 1e-7, CMAES_SIGMA_MAX)

        # ── Rolling baseline update (self-play mode) ─────────────
        # Every ROLLING_BASELINE_PERIOD gens, promote best_x to the self-play
        # opponent. This prevents the optimizer from only beating a stale
        # initial baseline while the real performance has moved on.
        # Must happen BEFORE the checkpoint write so the saved baseline
        # is always consistent with the generation just completed.
        if gen % ROLLING_BASELINE_PERIOD == 0
            ROLLING_BASELINE[] = copy(clamp_weights(best_x))
        end

        # ── Per-gen timing ─────────────────────────────────────────
        gen_elapsed = time() - gen_t0
        push!(gen_times, gen_elapsed)

        # ── Save checkpoint ──────────────────────────────────────
        # Written atomically so a Ctrl+C here never corrupts the file.
        if checkpoint_path != ""
            save_checkpoint(checkpoint_path, gen, best_f, best_x,
                            ROLLING_BASELINE[], m, σ, C, pc, ps)
        end

        # ── JSON checkpoint (every N generations) ─────────────────
        if json_checkpoint_path != "" && gen % JSON_CHECKPOINT_INTERVAL == 0
            save_json_checkpoint(json_checkpoint_path, gen, best_f,
                                 clamp_weights(best_x), σ, m,
                                 fitness_history, gen_times)
            @printf("  [checkpoint] JSON snapshot saved → %s\n", json_checkpoint_path)
        end

        # ── Progress report ──────────────────────────────────────
        gen_best = clamp_weights(samples[order[1]])
        @printf("  gen %3d/%d  best=%+.4f  σ=%.5f  t=%.1fs  top=[%.3f %.4f %.3f %.4f %.4f]\n",
                gen, max_gen, raw_fitness[order[1]], σ, gen_elapsed,
                gen_best[1], gen_best[2], gen_best[3], gen_best[4], gen_best[5])
        flush(stdout)   # ensure output appears immediately in redirected logs

        # ── Early stopping ──────────────────────────────────────
        push!(fitness_history, best_f)
        if length(fitness_history) >= early_stop_window
            window = @view fitness_history[end-early_stop_window+1:end]
            improvement = maximum(window) - minimum(window)
            if improvement < early_stop_threshold
                @printf("  Early stopping at gen %d: best_f plateau (%.5f range over %d gens)\n",
                        gen, improvement, early_stop_window)
                break
            end
        end
        if σ < 1e-5
            @printf("  Early stopping at gen %d: σ collapsed to %.2e\n", gen, σ)
            break
        end

        # ── Plateau detection (warning only, does not stop) ──────
        if length(fitness_history) >= PLATEAU_WINDOW
            window = @view fitness_history[end-PLATEAU_WINDOW+1:end]
            improvement = maximum(window) - minimum(window)
            if improvement < PLATEAU_THRESHOLD
                if !plateau_warned
                    @printf("  [PLATEAU WARNING] gen %d: best_f hasn't improved by >%.3f for %d gens (range=%.5f)\n",
                            gen, PLATEAU_THRESHOLD, PLATEAU_WINDOW, improvement)
                    @printf("  [PLATEAU WARNING] Consider: increase λ, raise σ, adjust n_pairs, or stop early.\n")
                    flush(stdout)
                    plateau_warned = true
                end
            else
                plateau_warned = false  # reset if we escape the plateau
            end
        end
    end

    # ── Final JSON checkpoint ──────────────────────────────────────
    if json_checkpoint_path != ""
        save_json_checkpoint(json_checkpoint_path, length(fitness_history),
                             best_f, clamp_weights(best_x), σ, m,
                             fitness_history, gen_times)
        @printf("  [checkpoint] Final JSON snapshot saved → %s\n", json_checkpoint_path)
    end

    println()
    @printf("  Overall best fitness: %+.4f\n", best_f)
    return clamp_weights(best_x)
end

# ── Top-level runner ─────────────────────────────────────────────
@inline function default_game_workers(mode::Symbol)::Int
    return Threads.nthreads()
end

function run_optimize(; depth::Int = 1,
                        n_pairs::Int = 2,
                        λ::Int = 8,
                        n_gen::Int = 30,
                        seed::UInt64 = 0x0000_1234_ABCD_0000,
                        checkpoint_path::String = "optimize_checkpoint.txt",
                        mode::Symbol = :stockfish,
                        stockfish_path::String = DEFAULT_STOCKFISH_PATH,
                        sf_movetime_ms::Int = 120,
                        sf_skill::Int = 5,
                        sf_hash_mb::Int = 16,
                        sf_threads::Int = 1,
                        sf_nodes::Int = 0,
                        game_workers::Int = 0)

    mode in (:selfplay, :stockfish) || error("Unknown mode: $mode")
    depth >= 1     || error("depth must be >= 1")
    λ >= 2         || error("λ must be >= 2")
    n_pairs >= 1   || error("n_pairs must be >= 1")
    n_gen >= 1     || error("n_gen must be >= 1")
    sf_movetime_ms >= 1 || error("sf_movetime_ms must be >= 1")
    sf_hash_mb >= 1 || error("sf_hash_mb must be >= 1")
    sf_threads >= 1 || error("sf_threads must be >= 1")
    sf_nodes >= 0   || error("sf_nodes must be >= 0")

    if depth >= 4 && n_pairs < 4
        println(stderr,
                "  [hint] depth=$depth with n_pairs=$n_pairs gives only $(2*n_pairs) games per candidate.\n" *
                "         Consider --n-pairs 4 (or higher) to reduce fitness noise at deep search.\n")
    end

    workers = game_workers == 0 ? default_game_workers(mode) : game_workers
    workers >= 1 || error("game_workers must be >= 1")
    workers = min(workers, Threads.nthreads())

    cfg = mode === :stockfish ? configure_stockfish!(;
        path=stockfish_path,
        movetime_ms=sf_movetime_ms,
        skill=sf_skill,
        hash_mb=sf_hash_mb,
        threads=sf_threads,
        nodes=sf_nodes
    ) : nothing

    rng = MersenneTwister(seed)

    println()
    println("  ══════════════════════════════════════════════════")
    println(mode === :stockfish ?
            "  FieldEngine Weight Optimizer — CMA-ES vs Stockfish" :
            "  FieldEngine Weight Optimizer — CMA-ES Self-Play")
    @printf("  depth=%d  n_pairs=%d  λ=%d  generations=%d\n",
            depth, n_pairs, λ, n_gen)
    n_games = λ * 2 * n_pairs
    @printf("  Games per generation: %d  (≈ %d total across all gens)\n",
            n_games, n_games * n_gen)
    @printf("  Game workers: %d  (julia threads=%d)\n", workers, Threads.nthreads())
    if mode === :stockfish
        limit_str = cfg.nodes > 0 ? "nodes=$(cfg.nodes)" : "time=$(cfg.movetime_ms)ms"
        @printf("  Stockfish: %s  %s  skill=%d  hash=%dMB  sf_threads=%d\n",
                cfg.path, limit_str, cfg.skill, cfg.hash_mb, cfg.threads)
    end
    println("  ══════════════════════════════════════════════════")
    println()
    flush(stdout)

    # Pre-warm Stockfish sessions on all threads before gen 1.
    # Cold-starting a process mid-game-batch is slow and can skew early fitness
    # measurements. Warming here means every thread hits gen 1 with an
    # already-ready engine process, and we catch path errors immediately.
    #
    # Stagger launches in batches of 8 to avoid overwhelming the system when
    # many threads (e.g. 32 on c7g.8xlarge) all try to spawn Stockfish at once.
    # Use a generous 60s timeout per session for UCI init under heavy load.
    if mode === :stockfish
        print("  Pre-warming Stockfish on $(workers) threads...")
        flush(stdout)
        # Stagger launches in batches of 8 to avoid overwhelming the system.
        # Since ALL I/O is now pure POSIX (no libuv), @spawn is safe — there
        # are no cross-thread event-loop dependencies to worry about.
        batch_size = min(8, workers)
        for batch_start in 1:batch_size:workers
            batch_end = min(batch_start + batch_size - 1, workers)
            tasks = Task[]
            for tid in batch_start:batch_end
                t = Threads.@spawn get_stockfish_session!($tid; init_timeout_s=60.0)
                push!(tasks, t)
            end
            for t in tasks
                fetch(t)
            end
            print(" $(batch_end)")
            flush(stdout)
        end
        println(" done.")
        println()
        flush(stdout)
    end

    # batch_evaluate closes over depth and n_pairs, dispatching all games
    # across the full thread pool for maximum CPU utilization.
    batch_f = candidates -> batch_evaluate(candidates;
                                           depth=depth,
                                           n_pairs=n_pairs,
                                           mode=mode,
                                           game_workers=workers)

    # Initial covariance scaled to BASELINE magnitudes so σ0=0.1 gives a
    # 10% relative perturbation for every weight regardless of magnitude.
    scales = abs.(BASELINE)
    C0     = Matrix(Diagonal(scales .^ 2))

    best_w = try
        cmaes(batch_f, copy(BASELINE);
              σ0              = 0.1,
              C0              = C0,
              λ               = λ,
              max_gen         = n_gen,
              rng             = rng,
              checkpoint_path = checkpoint_path)
    finally
        mode === :stockfish && close_stockfish_sessions!()
    end

    println()
    println("  ══════════════════════════════════════════════════")
    println("  Optimization complete. Best weights found:")
    println()
    println("  Paste these into src/energy.jl to use tuned weights:")
    println()
    @printf("  const W_MATERIAL    = %.4f\n", best_w[1])
    @printf("  const W_FIELD       = %.6f\n", best_w[2])
    @printf("  const W_KING_SAFETY = %.6f\n", best_w[3])
    @printf("  const W_TENSION     = %.6f\n", best_w[4])
    @printf("  const W_MOBILITY    = %.6f\n", best_w[5])
    println()
    println("  (Delete optimize_checkpoint.txt to start a fresh run.)")
    println("  ══════════════════════════════════════════════════")
    println()
    flush(stdout)

    return best_w
end

@inline function parse_mode(s::String)::Symbol
    x = lowercase(strip(s))
    x == "selfplay"  && return :selfplay
    x == "stockfish" && return :stockfish
    error("Invalid --mode '$s'. Expected 'selfplay' or 'stockfish'.")
end

@inline function parse_on_off(s::String)::Bool
    x = lowercase(strip(s))
    x in ("on", "true", "1", "yes")  && return true
    x in ("off", "false", "0", "no") && return false
    error("Invalid value '$s'. Expected on/off.")
end

function print_usage()
    println("Usage:")
    println("  julia --threads auto src/optimize.jl [depth] [lambda] [generations] [options]")
    println()
    println("Stockfish is the default opponent. Run scripts/setup_stockfish.sh first")
    println("to download a pre-compiled binary to bin/stockfish, then:")
    println()
    println("  julia --threads auto src/optimize.jl 5 32 100")
    println("  julia --threads auto src/optimize.jl 5 32 100 --sf-nodes 50000")
    println()
    println("Options:")
    println("  --mode selfplay|stockfish    (default: stockfish)")
    println("  --stockfish PATH             (default: bin/stockfish)")
    println("  --sf-nodes N                 (default: 0 = use movetime)")
    println("                               Recommended for EC2: removes timing noise.")
    println("                               Try 50000 for skill 5, 100000 for skill 8+.")
    println("  --sf-time-ms N               (default: 120, used when --sf-nodes is 0)")
    println("  --sf-skill N                 (0..20, default: 5)")
    println("  --sf-hash-mb N               (default: 16)")
    println("  --sf-threads N               (default: 1)")
    println("  --n-pairs N                  (default: 2)")
    println("  --checkpoint PATH            (default: optimize_checkpoint.txt)")
    println("  --seed UINT64")
    println("  --game-workers N             (0=auto)")
    println()
    println("Self-play mode (no Stockfish required):")
    println("  julia --threads auto src/optimize.jl 5 32 100 --mode selfplay")
end

function parse_cli_args(args::Vector{String})
    depth = 1
    λ = 8
    n_gen = 30
    n_pairs = 2
    mode = :stockfish
    stockfish_path = DEFAULT_STOCKFISH_PATH
    sf_movetime_ms = 120
    sf_skill = 5
    sf_hash_mb = 16
    sf_threads = 1
    sf_nodes = 0
    game_workers = 0
    checkpoint_path = "optimize_checkpoint.txt"
    seed = UInt64(0x0000_1234_ABCD_0000)

    positional = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_usage()
            exit(0)
        elseif arg == "--mode"
            i == length(args) && error("Missing value for --mode")
            i += 1
            mode = parse_mode(args[i])
        elseif startswith(arg, "--mode=")
            mode = parse_mode(split(arg, "=", limit=2)[2])
        elseif arg == "--n-pairs"
            i == length(args) && error("Missing value for --n-pairs")
            i += 1
            n_pairs = parse(Int, args[i])
        elseif startswith(arg, "--n-pairs=")
            n_pairs = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--checkpoint"
            i == length(args) && error("Missing value for --checkpoint")
            i += 1
            checkpoint_path = args[i]
        elseif startswith(arg, "--checkpoint=")
            checkpoint_path = split(arg, "=", limit=2)[2]
        elseif arg == "--seed"
            i == length(args) && error("Missing value for --seed")
            i += 1
            seed = parse(UInt64, args[i])
        elseif startswith(arg, "--seed=")
            seed = parse(UInt64, split(arg, "=", limit=2)[2])
        elseif arg == "--game-workers"
            i == length(args) && error("Missing value for --game-workers")
            i += 1
            game_workers = parse(Int, args[i])
        elseif startswith(arg, "--game-workers=")
            game_workers = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--stockfish"
            i == length(args) && error("Missing value for --stockfish")
            i += 1
            stockfish_path = args[i]
        elseif startswith(arg, "--stockfish=")
            stockfish_path = split(arg, "=", limit=2)[2]
        elseif arg == "--sf-time-ms"
            i == length(args) && error("Missing value for --sf-time-ms")
            i += 1
            sf_movetime_ms = parse(Int, args[i])
        elseif startswith(arg, "--sf-time-ms=")
            sf_movetime_ms = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--sf-skill"
            i == length(args) && error("Missing value for --sf-skill")
            i += 1
            sf_skill = parse(Int, args[i])
        elseif startswith(arg, "--sf-skill=")
            sf_skill = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--sf-hash-mb"
            i == length(args) && error("Missing value for --sf-hash-mb")
            i += 1
            sf_hash_mb = parse(Int, args[i])
        elseif startswith(arg, "--sf-hash-mb=")
            sf_hash_mb = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--sf-threads"
            i == length(args) && error("Missing value for --sf-threads")
            i += 1
            sf_threads = parse(Int, args[i])
        elseif startswith(arg, "--sf-threads=")
            sf_threads = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--sf-nodes"
            i == length(args) && error("Missing value for --sf-nodes")
            i += 1
            sf_nodes = parse(Int, args[i])
        elseif startswith(arg, "--sf-nodes=")
            sf_nodes = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
        end
        i += 1
    end

    length(positional) > 3 && error("Too many positional arguments. Use --help.")
    length(positional) >= 1 && (depth = parse(Int, positional[1]))
    length(positional) >= 2 && (λ = parse(Int, positional[2]))
    length(positional) >= 3 && (n_gen = parse(Int, positional[3]))

    return (
        depth=depth,
        n_pairs=n_pairs,
        λ=λ,
        n_gen=n_gen,
        seed=seed,
        checkpoint_path=checkpoint_path,
        mode=mode,
        stockfish_path=stockfish_path,
        sf_movetime_ms=sf_movetime_ms,
        sf_skill=sf_skill,
        sf_hash_mb=sf_hash_mb,
        sf_threads=sf_threads,
        sf_nodes=sf_nodes,
        game_workers=game_workers,
    )
end

# ── Entry point ──────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_cli_args(ARGS)
    run_optimize(; opts...)
end
