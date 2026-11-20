// Fuzz the Arm NN ONNX parser on a binary ModelProto: protobuf decode, then OnnxParser's graph
// building (every supported ONNX operator, shape inference, constant handling).
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <armnn/Exceptions.hpp>
#include <cstdint>
#include <exception>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    std::vector<uint8_t> model(data, data + size);
    try
    {
        auto parser = armnnOnnxParser::IOnnxParser::Create();
        armnn::INetworkPtr net = parser->CreateNetworkFromBinary(model);
    }
    catch (const armnn::Exception&) {}
    catch (const std::exception&) {}
    return 0;
}
