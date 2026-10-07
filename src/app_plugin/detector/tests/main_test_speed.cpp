#include "NNDetector.hpp"
#include "network_performance.hpp"

#include <opencv2/videoio.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef NNDEPLOYMENT_PROJECT_ROOT
#define NNDEPLOYMENT_PROJECT_ROOT "."
#endif

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

void runTest(const fs::path &root, const TestCase &test,
             const std::string &benchmark_device)
{
    // 性能测试始终使用 OpenVINO 模型与配置。
    const fs::path config_path = root / "src/app_plugin/detector/config/detect_openvino.yaml";
    const fs::path model_folder = root / "所有模型";
    cv::VideoCapture video((root / test.video_path).string());
    cv::Mat frame;
    if (!video.isOpened() || !video.read(frame) || frame.empty())
        throw std::runtime_error("无法读取测试视频: " + (root / test.video_path).string());

    const YamlConfig config(config_path.string(), test.config_key,
                            model_folder.string());
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

    MPT::OfficialBenchmarkConfig benchmark;
    benchmark.perf_hint = "throughput";
    benchmark.time_seconds = 20;
    benchmark.inference_only = false;
    MPT::runOfficialBenchmark((root / test.model_path).string(), benchmark_device,
                              test.infer_mode, benchmark);
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const fs::path root = fs::absolute(
            argc > 1 ? fs::path(argv[1]) : fs::path(NNDEPLOYMENT_PROJECT_ROOT));
        // benchmark_app 的目标设备：无 Intel GPU 的机器用 CPU；可用第 2 个参数覆盖（如 GPU）。
        const std::string benchmark_device = argc > 2 ? argv[2] : "CPU";
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
        for (const TestCase &test : tests)
        {
            try
            {
                std::cout << "Testing " << test.name << std::endl;
                runTest(root, test, benchmark_device);
            }
            catch (const std::exception &error)
            {
                ++failures;
                std::cerr << test.name << " failed: " << error.what() << std::endl;
            }
        }
        return failures == 0 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main_test_speed failed: " << error.what() << std::endl;
        return 1;
    }
}
