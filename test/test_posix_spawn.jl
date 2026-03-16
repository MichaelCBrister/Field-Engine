# Test POSIX I/O from inside Threads.@spawn vs main task
sf_path = length(ARGS) >= 1 ? ARGS[1] : "stockfish"

Base.@kwdef struct PollFD
    fd::Int32; events::Int16; revents::Int16
end
const POLL_IN = Int16(0x0001)

function test_posix_io(label::String, sf_path::String)
    proc = open(`$sf_path`, "r+")
    stdin_fd  = reinterpret(Int32, Base._fd(proc.in))
    stdout_fd = reinterpret(Int32, Base._fd(proc.out))
    println("[$label] stdin_fd=$stdin_fd  stdout_fd=$stdout_fd  thread=$(Threads.threadid())")

    # POSIX write
    cmd = "uci\n"
    buf_w = Vector{UInt8}(codeunits(cmd))
    n_wr = ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
                 stdin_fd, buf_w, length(buf_w))
    println("[$label] wrote $n_wr bytes")

    # POSIX poll + read
    buf_r = Vector{UInt8}(undef, 4096)
    all_data = UInt8[]
    deadline = time() + 10.0
    while time() < deadline
        pfd = Ref(PollFD(fd=stdout_fd, events=POLL_IN, revents=Int16(0)))
        ret = ccall(:poll, Int32, (Ptr{PollFD}, UInt64, Int32), pfd, 1, 1000)
        if ret <= 0
            println("[$label] poll returned $ret (errno=$(Base.Libc.errno()))")
            continue
        end
        n = ccall(:read, Cssize_t, (Cint, Ptr{UInt8}, Csize_t),
                  stdout_fd, buf_r, length(buf_r))
        if n <= 0
            println("[$label] read returned $n (errno=$(Base.Libc.errno()))")
            continue
        end
        append!(all_data, @view buf_r[1:n])
        if occursin("uciok", String(copy(all_data)))
            println("[$label] SUCCESS: uciok received ($(length(all_data)) bytes)")
            kill(proc)
            return true
        end
    end
    println("[$label] FAILED: timeout ($(length(all_data)) bytes received)")
    kill(proc)
    return false
end

# Test 1: main task
println("=== Test from main task ===")
test_posix_io("main", sf_path)

# Test 2: @spawn task
println("\n=== Test from @spawn task ===")
t = Threads.@spawn test_posix_io("spawn", $sf_path)
result = fetch(t)
println("@spawn result: $result")
