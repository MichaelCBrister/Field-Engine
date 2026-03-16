# Quick test: verify POSIX poll+read works for Stockfish I/O on this platform.
# Run: julia test_posix_read.jl /path/to/stockfish

sf_path = length(ARGS) >= 1 ? ARGS[1] : "stockfish"

Base.@kwdef struct PollFD
    fd::Int32
    events::Int16
    revents::Int16
end
const POLL_IN = Int16(0x0001)

proc = open(`$sf_path`, "r+")
pipe = proc.out

# Extract raw fd
rawfd = Base._fd(pipe)
fd_int = reinterpret(Int32, rawfd)
println("pipe fd = $fd_int")

# Send "uci" command
write(proc, "uci\n")
flush(proc)
println("Sent 'uci', waiting for response...")

# Try POSIX poll + read
buf = Vector{UInt8}(undef, 4096)
linebuf = UInt8[]
deadline = time() + 10.0  # 10 second timeout
found_uciok = false

while time() < deadline
    pfd = Ref(PollFD(fd=fd_int, events=POLL_IN, revents=Int16(0)))
    ret = ccall(:poll, Int32, (Ptr{PollFD}, UInt64, Int32), pfd, 1, 1000)
    if ret < 0
        println("poll() failed: errno=$(Base.Libc.errno())")
        break
    elseif ret == 0
        println("  poll timeout (1s), retrying...")
        continue
    end

    n = ccall(:read, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), fd_int, buf, length(buf))
    if n < 0
        errno = Base.Libc.errno()
        println("  read() failed: errno=$errno (EAGAIN=11, EINTR=4)")
        if errno == 11 || errno == 4
            continue
        end
        break
    elseif n == 0
        println("  read() returned 0 (EOF)")
        break
    end

    append!(linebuf, @view buf[1:n])
    println("  read $n bytes, total=$(length(linebuf))")

    # Check for uciok
    s = String(copy(linebuf))
    if occursin("uciok", s)
        println("SUCCESS: received uciok")
        found_uciok = true
        break
    end
end

if !found_uciok
    println("FAILED: did not receive uciok within 10s")
    println("linebuf contents ($(length(linebuf)) bytes):")
    if !isempty(linebuf)
        println(String(copy(linebuf)))
    end
end

kill(proc)
