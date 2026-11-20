// Fuzz the Arm NN ONNX parser on a text-format ModelProto (IOnnxParser::CreateNetworkFromString):
// same graph building as the binary entry, reached through protobuf TextFormat.
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <armnn/Exceptions.hpp>
#include <cstdint>
#include <exception>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    std::string text(reinterpret_cast<const char*>(data), size);
    try
    {
        auto parser = armnnOnnxParser::IOnnxParser::Create();
        armnn::INetworkPtr net = parser->CreateNetworkFromString(text);
    }
    catch (const armnn::Exception&) {}
    catch (const std::exception&) {}
    return 0;
}
