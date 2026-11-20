// Fuzz the whole inference path on an Arm NN serialized model (the format that can express every
// layer type and descriptor): IDeserializer -> armnn::Optimize for CpuRef -> LoadNetwork ->
// EnqueueWorkload. The runtime is created once; the network loaded for an input is unloaded again.
#include <armnn/ArmNN.hpp>
#include <armnn/Exceptions.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <cstdint>
#include <exception>
#include <vector>

static armnn::IRuntime* GetRuntime()
{
    static armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());
    return runtime.get();
}

// The harness allocates the model's declared I/O tensors itself; bound what a fuzzed shape can ask for.
static constexpr unsigned int kMaxTensorBytes = 1u << 22;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    std::vector<uint8_t> model(data, data + size);
    armnn::IRuntime* runtime = GetRuntime();
    try
    {
        auto deserializer = armnnDeserializer::IDeserializer::Create();
        armnn::INetworkPtr net = deserializer->CreateNetworkFromBinary(model);
        armnn::IOptimizedNetworkPtr opt = armnn::Optimize(*net, {armnn::Compute::CpuRef},
                                                          runtime->GetDeviceSpec());
        armnn::NetworkId id;
        std::string err;
        if (runtime->LoadNetwork(id, std::move(opt), err) != armnn::Status::Success) return 0;

        // Bindings: ask the loaded network for the binding ids the deserializer assigned (0..n-1
        // as the Serializer writes them); stop at the first id the network does not have.
        std::vector<std::vector<uint8_t>> bufs;
        armnn::InputTensors inputs;
        armnn::OutputTensors outputs;
        bool ok = true;
        for (armnn::LayerBindingId b = 0; b < 16; ++b)
        {
            armnn::TensorInfo info;
            try { info = runtime->GetInputTensorInfo(id, b); } catch (const armnn::Exception&) { break; }
            if (info.GetNumBytes() > kMaxTensorBytes) { ok = false; break; }
            bufs.emplace_back(info.GetNumBytes(), 0);
            info.SetConstant(true);
            inputs.emplace_back(b, armnn::ConstTensor(info, bufs.back().data()));
        }
        for (armnn::LayerBindingId b = 0; ok && b < 16; ++b)
        {
            armnn::TensorInfo info;
            try { info = runtime->GetOutputTensorInfo(id, b); } catch (const armnn::Exception&) { break; }
            if (info.GetNumBytes() > kMaxTensorBytes) { ok = false; break; }
            bufs.emplace_back(info.GetNumBytes(), 0);
            outputs.emplace_back(b, armnn::Tensor(info, bufs.back().data()));
        }
        if (ok)
        {
            try { runtime->EnqueueWorkload(id, inputs, outputs); }
            catch (const armnn::Exception&) {}
            catch (const std::exception&) {}
        }
        runtime->UnloadNetwork(id);
    }
    catch (const armnn::Exception&) {}
    catch (const std::exception&) {}
    return 0;
}
