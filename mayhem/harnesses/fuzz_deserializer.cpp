// Fuzz the Arm NN serialized-model format: IDeserializer::CreateNetworkFromBinary (flatbuffer
// verify + every Parse* layer), then re-serialize the resulting network with ISerializer.
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnnSerializer/ISerializer.hpp>
#include <armnn/Exceptions.hpp>
#include <cstdint>
#include <exception>
#include <sstream>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    std::vector<uint8_t> model(data, data + size);
    try
    {
        auto deserializer = armnnDeserializer::IDeserializer::Create();
        armnn::INetworkPtr net = deserializer->CreateNetworkFromBinary(model);
        auto serializer = armnnSerializer::ISerializer::Create();
        serializer->Serialize(*net);
        std::ostringstream out;
        serializer->SaveSerializedToStream(out);
    }
    catch (const armnn::Exception&) {}
    catch (const std::exception&) {}
    return 0;
}
