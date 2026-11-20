// Fuzz the whole inference path on a .tflite model: TfLite parser -> armnn::Optimize for CpuRef ->
// IRuntime::LoadNetwork -> EnqueueWorkload (reference backend workloads, encoders/decoders).
// The runtime is created once and is read-only across inputs apart from the network it loads,
// which is unloaded again at the end of every input.
#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <cstdint>
#include <exception>
#include <vector>

static armnn::IRuntime* GetRuntime()
{
    static armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());
    return runtime.get();
}

// Bound per-tensor allocation: the harness has to allocate the model's declared I/O tensors itself,
// and a fuzzed shape can declare gigabytes.
static constexpr unsigned int kMaxTensorBytes = 1u << 22;

// Unloads the network on every exit path, including a throw from the tensor-info lookups, so no
// input leaves a network loaded in the shared runtime.
struct UnloadGuard
{
    armnn::IRuntime* runtime;
    armnn::NetworkId id;
    ~UnloadGuard() { try { runtime->UnloadNetwork(id); } catch (...) {} }
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    std::vector<uint8_t> model(data, data + size);
    armnn::IRuntime* runtime = GetRuntime();
    try
    {
        auto parser = armnnTfLiteParser::ITfLiteParser::Create();
        armnn::INetworkPtr net = parser->CreateNetworkFromBinary(model);
        if (parser->GetSubgraphCount() == 0) return 0;

        std::vector<armnnTfLiteParser::BindingPointInfo> inBind, outBind;
        for (const auto& name : parser->GetSubgraphInputTensorNames(0))
            inBind.push_back(parser->GetNetworkInputBindingInfo(0, name));
        for (const auto& name : parser->GetSubgraphOutputTensorNames(0))
            outBind.push_back(parser->GetNetworkOutputBindingInfo(0, name));
        for (auto& b : inBind)  if (b.second.GetNumBytes() > kMaxTensorBytes) return 0;
        for (auto& b : outBind) if (b.second.GetNumBytes() > kMaxTensorBytes) return 0;

        armnn::IOptimizedNetworkPtr opt = armnn::Optimize(*net, {armnn::Compute::CpuRef},
                                                          runtime->GetDeviceSpec());
        armnn::NetworkId id;
        std::string err;
        if (runtime->LoadNetwork(id, std::move(opt), err) != armnn::Status::Success) return 0;
        UnloadGuard unload{runtime, id};

        std::vector<std::vector<uint8_t>> inBufs, outBufs;
        armnn::InputTensors inputs;
        armnn::OutputTensors outputs;
        for (auto& b : inBind)
        {
            inBufs.emplace_back(b.second.GetNumBytes(), 0);
            armnn::TensorInfo info = runtime->GetInputTensorInfo(id, b.first);
            info.SetConstant(true);
            inputs.emplace_back(b.first, armnn::ConstTensor(info, inBufs.back().data()));
        }
        for (auto& b : outBind)
        {
            outBufs.emplace_back(b.second.GetNumBytes(), 0);
            outputs.emplace_back(b.first, armnn::Tensor(runtime->GetOutputTensorInfo(id, b.first),
                                                         outBufs.back().data()));
        }
        try { runtime->EnqueueWorkload(id, inputs, outputs); }
        catch (const armnn::Exception&) {}
        catch (const std::exception&) {}
    }
    catch (const armnn::Exception&) {}
    catch (const std::exception&) {}
    return 0;
}
