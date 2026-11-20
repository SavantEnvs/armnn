// Fuzz the external-profiling timeline decoders (server side): the timeline message directory
// packet, the timeline capture packets it describes, and the counter directory packet.
// Input: byte 0 selects the stream (bit 0 set: counter directory; clear: timeline directory then
// timeline capture), bytes 1-2 (LE) are the length of the first packet's body, the rest follows.
#include <timelineDecoder/DirectoryCaptureCommandHandler.hpp>
#include <timelineDecoder/TimelineCaptureCommandHandler.hpp>
#include <timelineDecoder/TimelineDecoder.hpp>
#include <timelineDecoder/TimelineDirectoryCaptureCommandHandler.hpp>
#include <common/include/Packet.hpp>
#include <common/include/ProfilingException.hpp>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>

static arm::pipe::Packet MakePacket(uint32_t family, uint32_t id, const uint8_t* d, size_t n)
{
    std::unique_ptr<unsigned char[]> buf(new unsigned char[n ? n : 1]);
    if (n) std::memcpy(buf.get(), d, n);
    uint32_t header = (family << 26) | (id << 16);
    return arm::pipe::Packet(header, static_cast<uint32_t>(n), buf);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < 3) return -1;
    const uint8_t mode = data[0];
    size_t first = static_cast<size_t>(data[1]) | (static_cast<size_t>(data[2]) << 8);
    data += 3; size -= 3;
    if (first > size) first = size;
    try
    {
        if (mode & 1)
        {
            arm::pipe::DirectoryCaptureCommandHandler dir("ArmNN", 0, 2, 1, true);
            dir(MakePacket(0, 2, data, size));
            dir.GetCounterDirectory();
        }
        else
        {
            arm::pipe::TimelineDecoder decoder;
            // The store-into-model callbacks the server installs; without them every Create* returns
            // TimelineStatus_Fail and the capture handler stops after the first record.
            decoder.SetDefaultCallbacks();
            arm::pipe::TimelineCaptureCommandHandler capture(1, 1, 0, decoder);
            arm::pipe::TimelineDirectoryCaptureCommandHandler directory(1, 0, 0, capture, true);
            directory(MakePacket(1, 0, data, first));
            capture(MakePacket(1, 1, data + first, size - first));
        }
    }
    catch (const arm::pipe::ProfilingException&) {}
    catch (const std::exception&) {}
    return 0;
}
