// Known-answer functional checks for Arm NN (run by mayhem/test.sh): build a network through the
// public INetwork API, run it on the CpuRef backend and compare the outputs to hand-computed values;
// round-trip it through ISerializer/IDeserializer and check the same answer; parse the shipped
// .tflite and .armnn sample models and check their bindings (and, for the tflite, its output).
// Prints one "PASS <name>" / "FAIL <name>: <why>" line per check.
#include <armnn/ArmNN.hpp>
#include <armnnDeserializer/IDeserializer.hpp>
#include <armnnSerializer/ISerializer.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static std::string g_dump;   // optional argv[2]: write the serialized models there (seed generation)
static void Report(const std::string& name, bool ok, const std::string& why = "")
{
    if (ok) std::printf("PASS %s\n", name.c_str());
    else { std::printf("FAIL %s: %s\n", name.c_str(), why.c_str()); ++g_fail; }
}

static armnn::INetworkPtr BuildAddRelu()
{
    armnn::INetworkPtr net = armnn::INetwork::Create();
    armnn::TensorInfo info({1, 4}, armnn::DataType::Float32);
    auto* a = net->AddInputLayer(0, "a");
    auto* b = net->AddInputLayer(1, "b");
    auto* add = net->AddElementwiseBinaryLayer(armnn::ElementwiseBinaryDescriptor(armnn::BinaryOperation::Add), "add");
    armnn::ActivationDescriptor relu; relu.m_Function = armnn::ActivationFunction::ReLu;
    auto* act = net->AddActivationLayer(relu, "relu");
    auto* out = net->AddOutputLayer(0, "out");
    a->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    b->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(act->GetInputSlot(0));
    act->GetOutputSlot(0).Connect(out->GetInputSlot(0));
    a->GetOutputSlot(0).SetTensorInfo(info);
    b->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);
    act->GetOutputSlot(0).SetTensorInfo(info);
    return net;
}

static bool RunAddRelu(armnn::IRuntime& rt, armnn::INetwork& net, std::string& why)
{
    armnn::IOptimizedNetworkPtr opt = armnn::Optimize(net, {armnn::Compute::CpuRef}, rt.GetDeviceSpec());
    armnn::NetworkId id; std::string err;
    if (rt.LoadNetwork(id, std::move(opt), err) != armnn::Status::Success) { why = "LoadNetwork: " + err; return false; }
    std::vector<float> a = {-1.f, 2.f, -3.f, 4.f}, b = {0.5f, 0.5f, 0.5f, 0.5f}, o(4, -99.f);
    armnn::TensorInfo ia = rt.GetInputTensorInfo(id, 0); ia.SetConstant(true);
    armnn::TensorInfo ib = rt.GetInputTensorInfo(id, 1); ib.SetConstant(true);
    armnn::InputTensors in = {{0, armnn::ConstTensor(ia, a.data())}, {1, armnn::ConstTensor(ib, b.data())}};
    armnn::OutputTensors out = {{0, armnn::Tensor(rt.GetOutputTensorInfo(id, 0), o.data())}};
    rt.EnqueueWorkload(id, in, out);
    rt.UnloadNetwork(id);
    const float want[4] = {0.f, 2.5f, 0.f, 4.5f};
    for (int i = 0; i < 4; ++i)
        if (std::fabs(o[i] - want[i]) > 1e-6f) { why = "out[" + std::to_string(i) + "]=" + std::to_string(o[i]); return false; }
    return true;
}

static std::vector<uint8_t> ReadFile(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::string Shape(const armnn::TensorShape& s)
{
    std::string r;
    for (unsigned i = 0; i < s.GetNumDimensions(); ++i) r += (i ? "x" : "") + std::to_string(s[i]);
    return r;
}

static std::vector<float> RunSingle(armnn::IRuntime& rt, armnn::INetwork& net, armnn::LayerBindingId inId,
                                    armnn::LayerBindingId outId, const std::vector<float>& in, size_t outSize)
{
    armnn::IOptimizedNetworkPtr opt = armnn::Optimize(net, {armnn::Compute::CpuRef}, rt.GetDeviceSpec());
    armnn::NetworkId id; std::string err;
    if (rt.LoadNetwork(id, std::move(opt), err) != armnn::Status::Success) throw armnn::Exception(err);
    std::vector<float> out(outSize, -99.f);
    armnn::TensorInfo ii = rt.GetInputTensorInfo(id, inId); ii.SetConstant(true);
    armnn::InputTensors it = {{inId, armnn::ConstTensor(ii, in.data())}};
    armnn::OutputTensors ot = {{outId, armnn::Tensor(rt.GetOutputTensorInfo(id, outId), out.data())}};
    rt.EnqueueWorkload(id, it, ot);
    rt.UnloadNetwork(id);
    return out;
}

int main(int argc, char** argv)
{
    const std::string root = argc > 1 ? argv[1] : ".";
    if (argc > 2) g_dump = argv[2];
    armnn::IRuntimePtr rt = armnn::IRuntime::Create(armnn::IRuntime::CreationOptions());

    { std::string why; armnn::INetworkPtr net = BuildAddRelu();
      bool ok = false; try { ok = RunAddRelu(*rt, *net, why); } catch (const std::exception& e) { why = e.what(); }
      Report("cpuref_add_relu", ok, why); }

    { std::string why; bool ok = false;
      try {
        armnn::INetworkPtr net = BuildAddRelu();
        auto ser = armnnSerializer::ISerializer::Create();
        ser->Serialize(*net);
        std::stringstream ss; ser->SaveSerializedToStream(ss);
        std::string blob = ss.str();
        if (!g_dump.empty()) std::ofstream(g_dump + "/add_relu.armnn", std::ios::binary) << blob;
        std::vector<uint8_t> bytes(blob.begin(), blob.end());
        auto des = armnnDeserializer::IDeserializer::Create();
        armnn::INetworkPtr back = des->CreateNetworkFromBinary(bytes);
        auto bi = des->GetNetworkInputBindingInfo(0, "a");
        if (Shape(bi.m_TensorInfo.GetShape()) != "1x4") why = "binding a shape " + Shape(bi.m_TensorInfo.GetShape());
        else ok = RunAddRelu(*rt, *back, why);
      } catch (const std::exception& e) { why = e.what(); }
      Report("serializer_roundtrip_add_relu", ok, why); }


    { std::string why; bool ok = false;
      try {
        auto parser = armnnTfLiteParser::ITfLiteParser::Create();
        armnn::INetworkPtr net = parser->CreateNetworkFromBinary(ReadFile(root + "/tests/InterfaceTests/simple_conv2d_1_op.tflite"));
        auto inNames = parser->GetSubgraphInputTensorNames(0);
        auto outNames = parser->GetSubgraphOutputTensorNames(0);
        auto ib = parser->GetNetworkInputBindingInfo(0, inNames.at(0));
        auto ob = parser->GetNetworkOutputBindingInfo(0, outNames.at(0));
        std::printf("INFO tflite in=%s %s out=%s %s\n", inNames[0].c_str(), Shape(ib.second.GetShape()).c_str(),
                    outNames[0].c_str(), Shape(ob.second.GetShape()).c_str());
        std::vector<float> in(ib.second.GetNumElements());
        for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i % 7) - 3.f;
        std::vector<float> out = RunSingle(*rt, *net, ib.first, ob.first, in, ob.second.GetNumElements());
        double sum = 0; for (float v : out) sum += v;
        std::printf("INFO tflite out_sum=%.6f out0=%.6f\n", sum, out.empty() ? 0.0 : out[0]);
        // Golden values: 1x5x5x2 input of (i%7)-3 through the model's 3x3 conv on CpuRef.
        ok = Shape(ib.second.GetShape()) == "1x5x5x2" && Shape(ob.second.GetShape()) == "1x3x3x2" &&
             out.size() == 18 && std::fabs(sum - 14.740168) < 1e-3 && std::fabs(out[0] - (-0.581886)) < 1e-4;
        if (ok)
        {
            // The same model through ISerializer -> IDeserializer must compute the same answer.
            auto ser = armnnSerializer::ISerializer::Create();
            ser->Serialize(*net);
            std::stringstream ss; ser->SaveSerializedToStream(ss);
            std::string blob = ss.str();
            if (!g_dump.empty()) std::ofstream(g_dump + "/simple_conv2d.armnn", std::ios::binary) << blob;
            auto des = armnnDeserializer::IDeserializer::Create();
            armnn::INetworkPtr back = des->CreateNetworkFromBinary(std::vector<uint8_t>(blob.begin(), blob.end()));
            auto dib = des->GetNetworkInputBindingInfo(0, inNames[0]);
            auto dob = des->GetNetworkOutputBindingInfo(0, outNames[0]);
            std::vector<float> out2 = RunSingle(*rt, *back, dib.m_BindingId, dob.m_BindingId, in, dob.m_TensorInfo.GetNumElements());
            Report("tflite_to_armnn_roundtrip", out2 == out, "deserialized model computes a different output");
        }
        if (!ok) why = "unexpected output";
      } catch (const std::exception& e) { why = e.what(); }
      Report("tflite_simple_conv2d", ok, why); }

    return g_fail ? 1 : 0;
}
