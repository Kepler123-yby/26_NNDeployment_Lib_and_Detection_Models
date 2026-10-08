#include "NNDetector.hpp"
#include "AppConfig.hpp"

#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

// 端到端帧率对比工具：对同一模型重复推理，输出平均耗时与 FPS。
// 通过 cv::CommandLineParser 解析配置与 CUDA 覆盖。
//
// 示例：
//   detector_speed_bench --config=detect_tensorrt.yaml --key=armor_v8 --iterations=400 --cuda
//   detector_speed_bench /path/to/root --config=detect_openvino.yaml --no-cuda

namespace fs = std::filesystem;

namespace
{
std::string benchKeys()
{
    return app::commonKeys() +
           "{key k|armor_v8|模型配置节点名}"
           "{model||模型文件路径（覆盖配置中的 xml）}"
           "{iterations n|500|重复推理次数}"
           "{video v||测试视频路径（默认 测试视频/装甲板.mp4）}";
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const app::CommandLine cli(argc, argv,
                                  {"config", "c", "key", "k", "model", "iterations", "n", "video", "v"});
        cv::CommandLineParser parser(cli.argc(), cli.argv(), benchKeys());
        parser.about("NNdeployment 端到端帧率对比工具");
        if (!app::handleParser(parser))
            return parser.has("help") ? 0 : 1;

        const app::AppConfig app_config = app::resolveAppConfig(parser, "detect_openvino.yaml");
        const std::string model_key = parser.get<std::string>("key");
        const std::string model_arg = parser.get<std::string>("model");
        const int iterations = parser.get<int>("iterations");
        const std::string video_arg = parser.get<std::string>("video");
        const fs::path video_path =
            video_arg.empty() ? app_config.root / "测试视频/装甲板.mp4"
                              : fs::absolute(fs::path(video_arg));

        cv::VideoCapture video(video_path.string());
        cv::Mat frame;
        if (!video.isOpened() || !video.read(frame) || frame.empty())
            throw std::runtime_error("无法读取测试视频: " + video_path.string());

        const YamlConfig config = [&]
        {
            YamlConfig c = app::makeYamlConfig(app_config, model_key);
            if (!model_arg.empty())
                c.model = model_arg;
            return c;
        }();

        std::cout << "配置文件: " << app_config.config_path.string()
                  << " | 节点: " << model_key
                  << (model_arg.empty() ? "" : (" | 模型: " + model_arg))
                  << " | OpenCV CUDA 可用: " << (opencvCudaAvailable() ? "是" : "否")
                  << " | " << app::describeDevice(config) << std::endl;

        ArmorDetector detector(config, 0, DebugConfig(false, false));

        // 预热。
        for (int index = 0; index < 10; ++index)
            detector.process(frame, 2);

        const auto start = std::chrono::steady_clock::now();
        for (int index = 0; index < iterations; ++index)
            detector.process(frame, 2);
        const auto stop = std::chrono::steady_clock::now();

        const double total_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << "key=" << model_key
                  << " iterations=" << iterations
                  << " avg=" << total_ms / iterations << " ms"
                  << " FPS=" << iterations * 1000.0 / total_ms << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "detector_speed_bench failed: " << error.what() << std::endl;
        return 1;
    }
}
