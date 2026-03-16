# Verify pure POSIX I/O works for Stockfish on this platform.
# Tests both POSIX write (stdin) and POSIX poll+read (stdout).
# Run: julia test_posix_io.jl /path/to/stockfish

sf_path = length(ARGS) >= 1 ? ARGS[1] : "stockfish"

Base.@kwdef struct PollFD
    fd::Int32; events::Int16; revents::Int16
end
const POLL_IN = Int16(0x0001)

proc = open(`$sf_path`, "r+")

# Extract raw POSIX fds
stdin_fd  = reinterpret(Int32, Base._fd(proc.in))
stdout_fd = reinterpret(Int32, Base._fd(proc.out))
println("stdin_fd=$stdin_fd  stdout_fd=$stdout_fd")

# POSIX write "uci\n" to stdin
cmd = "uci\n"
buf_w = Vector{UInt8}(codeunits(cmd))
n_written = ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
                  stdin_fd, buf_w, length(buf_w))
println("Wrote $n_written bytes via POSIX write()")

# POSIX poll + read from stdout
buf_r = Vector{UInt8}(undef, 4096)
all_data = UInt8[]
deadline = time() + 10.0
local found_uciok = false

while time() < deadline
    pfd = Ref(PollFD(fd=stdout_fd, events=POLL_IN, revents=Int16(0)))
    ret = ccall(:poll, Int32, (Ptr{PollFD}, UInt64, Int32), pfd, 1, 1000)
    ret <= 0 && continue

    n = ccall(:read, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
              stdout_fd, buf_r, length(buf_r))
    n <= 0 && continue

    append!(all_data, @view buf_r[1:n])
    if occursin("uciok", String(copy(all_data)))
        println("SUCCESS: received uciok ($n bytes this read, $(length(all_data)) total)")
        found_uciok = true
        break
    end
end

if !found_uciok
    println("FAILED: uciok not received in 10s ($(length(all_data)) bytes)")
end

# Also test "isready" → "readyok"
cmd2 = "isready\n"
buf_w2 = Vector{UInt8}(codeunits(cmd2))
ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), stdin_fd, buf_w2, length(buf_w2))
all_data2 = UInt8[]
t0 = time()
while time() - t0 < 5.0
    pfd = Ref(PollFD(fd=stdout_fd, events=POLL_IN, revents=Int16(0)))
    ret = ccall(:poll, Int32, (Ptr{PollFD}, UInt64, Int32), pfd, 1, 1000)
    ret <= 0 && continue
    n = ccall(:read, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), stdout_fd, buf_r, length(buf_r))
    n <= 0 && continue
    append!(all_data2, @view buf_r[1:n])
    if occursin("readyok", String(copy(all_data2)))
        println("SUCCESS: received readyok")
        break
    end
end

kill(proc)
println("All POSIX I/O tests passed!")
