// Fuzz the Arm NN TfLite parser: a .tflite flatbuffer -> armnn::INetwork (all Parse* operators,
// tensor-shape inference, layer validation). Parse errors are reported as exceptions.
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnn/Exceptions.hpp>
#include <cstdint>
#include <exception>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    std::vector<uint8_t> model(data, data + size);
    try
    {
        auto parser = armnnTfLiteParser::ITfLiteParser::Create();
        armnn::INetworkPtr net = parser->CreateNetworkFromBinary(model);
        for (size_t sg = 0; sg < parser->GetSubgraphCount(); ++sg)
        {
            for (const auto& name : parser->GetSubgraphInputTensorNames(sg))
                parser->GetNetworkInputBindingInfo(sg, name);
            for (const auto& name : parser->GetSubgraphOutputTensorNames(sg))
                parser->GetNetworkOutputBindingInfo(sg, name);
        }
    }
    catch (const armnn::Exception&) {}
    catch (const std::exception&) {}
    return 0;
}
