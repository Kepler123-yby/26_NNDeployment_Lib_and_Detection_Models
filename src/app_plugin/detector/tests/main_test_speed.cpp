#include "NNDetector.hpp"
#include "AppConfig.hpp"
#include "network_performance.hpp"

#include <opencv2/videoio.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

// 性能测试：对 V8 / V5 / 大符 各连续推理 kIterations 次并输出分段耗时，
// 随后用官方 benchmark_app 跑吞吐。配置通过 cv::CommandLineParser 管理，与 detector_speed_bench 对齐。
//
// 用法：
//   main_test_speed [root] [--config=<yaml>] [--benchmark-device=CPU] [--cuda|--no-cuda]
//
// 注意：本测试的 benchmark 部分固定使用 OpenVINO 模型（benchmark_app）。

namespace fs = std::filesystem;

namespace
{
constexpr int kIterations = 2000;

enum class Task
{
    Armor,
    Rune
};

struct TestCase
{
    std::string name;
    std::string config_key;
    Task task;
    fs::path model_path;
    fs::path video_path;
    std::string infer_mode;
    std::size_t pipeline_delay;
};

std::string speedKeys()
{
    return app::commonKeys() +
           "{key k||只运行指定测试用例（armor_v8 / armor_v5 / rune_detect），不传则全部}"
           "{model||模型文件路径（覆盖配置中的 xml，也用于 benchmark）}"
           "{benchmark-device b|CPU|benchmark_app 的目标设备（如 CPU / GPU）}";
}

void runTest(const app::AppConfig &app_config, const TestCase &test,
             const std::string &benchmark_device, const std::string &model_arg)
{
    cv::VideoCapture video((app_config.root / test.video_path).string());
    cv::Mat frame;
    if (!video.isOpened() || !video.read(frame) || frame.empty())
        throw std::runtime_error("无法读取测试视频: " + (app_config.root / test.video_path).string());

    const YamlConfig config = [&]
    {
        YamlConfig c = app::makeYamlConfig(app_config, test.config_key);
        if (!model_arg.empty())
            c.model = model_arg;
        return c;
    }();
    if (test.task == Task::Armor)
    {
        ArmorDetector detector(config, test.pipeline_delay,
                               DebugConfig(false, true));
        for (int index = 0; index < kIterations; ++index)
            detector.process(frame, 2);
    }
    else
    {
        RuneDetector detector(config, test.pipeline_delay,
                              DebugConfig(false, true));
        for (int index = 0; index < kIterations; ++index)
            detector.process(frame);
    }

    const fs::path benchmark_model = model_arg.empty()
                                         ? (app_config.root / test.model_path)
                                         : fs::absolute(fs::path(model_arg));

    MPT::OfficialBenchmarkConfig benchmark;
    benchmark.perf_hint = "throughput";
    benchmark.time_seconds = 20;
    benchmark.inference_only = false;
    MPT::runOfficialBenchmark(benchmark_model.string(), benchmark_device,
                              test.infer_mode, benchmark);
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const app::CommandLine cli(argc, argv,
                                  {"config", "c", "key", "k", "model", "benchmark-device", "b"});
        cv::CommandLineParser parser(cli.argc(), cli.argv(), speedKeys());
        parser.about("NNdeployment 性能测试（默认 config/detect_openvino.yaml）");
        if (!app::handleParser(parser))
            return parser.has("help") ? 0 : 1;

        const app::AppConfig app_config =
            app::resolveAppConfig(parser, "detect_openvino.yaml");
        const std::string only_key = parser.get<std::string>("key");
        const std::string model_arg = parser.get<std::string>("model");
        const std::string benchmark_device = parser.get<std::string>("benchmark-device");

        std::cout << "配置文件: " << app_config.config_path.string()
                  << " | benchmark 设备: " << benchmark_device
                  << (only_key.empty() ? "" : (" | 测试用例: " + only_key))
                  << (model_arg.empty() ? "" : (" | 模型: " + model_arg))
                  << " | OpenCV CUDA 可用: " << (opencvCudaAvailable() ? "是" : "否")
                  << " | 预处理=" << app::describeCuda(app_config.cuda.preprocess)
                  << " 后处理=" << app::describeCuda(app_config.cuda.postprocess) << std::endl;

        const std::array<TestCase, 3> tests = {{
            {"Armor V8", "armor_v8", Task::Armor,
             "所有模型/openvino/Infantry-v8n-fp16-20260726-D1.8w-B16/Infantry-v8n-fp16-20260726-D1.8w-B16.xml",
             "测试视频/中距离陀螺.avi", "async", 1},
            {"Armor V5", "armor_v5", Task::Armor,
             "所有模型/openvino/Infantry-v5n-fp16-20250725/Infantry-v5n-Release-20260725.xml",
             "测试视频/中距离陀螺.avi", "async", 1},
            {"Rune V8", "rune_detect", Task::Rune,
             "所有模型/openvino/Rune-v8n-fp16-20260624-D14367-B16/Rune-v8n-fp16-20260624-D14367-B16.xml",
             "测试视频/符.avi", "sync", 0},
        }};

        int failures = 0;
        int ran = 0;
        for (const TestCase &test : tests)
        {
            if (!only_key.empty() && test.config_key != only_key)
                continue;
            ++ran;
            try
            {
                std::cout << "Testing " << test.name << std::endl;
                runTest(app_config, test, benchmark_device, model_arg);
            }
            catch (const std::exception &error)
            {
                ++failures;
                std::cerr << test.name << " failed: " << error.what() << std::endl;
            }
        }
        if (ran == 0)
        {
            std::cerr << "未知的 --key: " << only_key
                      << "（可用：armor_v8 / armor_v5 / rune_detect）" << std::endl;
            return 1;
        }
        return failures == 0 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main_test_speed failed: " << error.what() << std::endl;
        return 1;
    }
}
