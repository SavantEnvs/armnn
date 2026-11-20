// Fuzz the external-profiling pipe server (gatord side): the bytes a profiled client sends over the
// socket are written into one end of a socketpair; BasePipeServer reads the stream metadata packet
// (endianness detection, metadata length) and then every following packet header + payload.
#include <basePipeServer/BasePipeServer.hpp>
#include <common/include/ProfilingException.hpp>
#include <cstdint>
#include <exception>
#include <sys/socket.h>
#include <unistd.h>

// The input is written in full before the server reads, so it has to fit in the socket buffer
// (a larger write would block forever with nobody reading yet).
static constexpr size_t kMaxInput = 60000;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size > kMaxInput) return -1;
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) return 0;
    int sndbuf = 1 << 20;
    setsockopt(fds[0], SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    size_t off = 0;
    while (off < size)
    {
        ssize_t n = write(fds[0], data + off, size - off);
        if (n <= 0) break;
        off += static_cast<size_t>(n);
    }
    close(fds[0]);   // peer sees EOF/POLLHUP once the data is consumed
    try
    {
        arm::pipe::BasePipeServer server(fds[1], false);   // takes ownership of fds[1]
        if (server.WaitForStreamMetaData())
        {
            for (int i = 0; i < 4096; ++i)
            {
                arm::pipe::Packet p = server.WaitForPacket(0);
                if (p.IsEmpty()) break;
            }
        }
    }
    catch (const arm::pipe::ProfilingException&) {}
    catch (const std::exception&) {}
    return 0;
}
