#include "NNDetector.hpp"
#include "AppConfig.hpp"

#include <opencv2/videoio.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

// 冒烟测试：检查 V5 / V8 / 大符 的模型加载、首帧推理、空图空结果、空输入与错误配置。
// 配置通过 cv::CommandLineParser 管理，与 detector_speed_bench 对齐。
//
// 用法：
//   detector_smoke_test [root] [--config=<yaml>] [--cuda|--no-cuda|...] [-h]

namespace fs = std::filesystem;

namespace
{
constexpr std::size_t kSyncPipelineDelay = 0;

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

cv::Mat readFrame(const fs::path &path)
{
    cv::VideoCapture video(path.string());
    cv::Mat frame;
    require(video.isOpened() && video.read(frame) && !frame.empty(),
            "无法读取测试帧: " + path.string());
    return frame;
}

void testArmor(const app::AppConfig &app_config, const std::string &key,
               const cv::Mat &frame)
{
    ArmorDetector detector(app::makeYamlConfig(app_config, key), kSyncPipelineDelay);
    auto detected = detector.process(frame, 2);
    require(detected.has_value() && !detected->image.empty(),
            key + " 首帧推理未返回图像");

    const cv::Mat blank = cv::Mat::zeros(frame.size(), frame.type());
    auto empty_result = detector.process(blank, 2);
    require(empty_result.has_value() && empty_result->results.empty(),
            key + " 空白图像应返回空结果");

    bool rejected_empty_input = false;
    try
    {
        detector.process(cv::Mat{}, 2);
    }
    catch (const std::invalid_argument &)
    {
        rejected_empty_input = true;
    }
    require(rejected_empty_input, key + " 未拒绝空输入");
    std::cout << key << ": first-frame, blank-result and empty-input checks passed\n";
}

void testRune(const app::AppConfig &app_config, const cv::Mat &frame)
{
    RuneDetector detector(app::makeYamlConfig(app_config, "rune_detect"), kSyncPipelineDelay);
    auto detected = detector.process(frame);
    require(detected.has_value() && !detected->image.empty(),
            "rune_detect 首帧推理未返回图像");

    const cv::Mat blank = cv::Mat::zeros(frame.size(), frame.type());
    auto empty_result = detector.process(blank);
    require(empty_result.has_value() && empty_result->results.empty(),
            "rune_detect 空白图像应返回空结果");

    bool rejected_empty_input = false;
    try
    {
        detector.process(cv::Mat{});
    }
    catch (const std::invalid_argument &)
    {
        rejected_empty_input = true;
    }
    require(rejected_empty_input, "rune_detect 未拒绝空输入");
    std::cout << "rune_detect: first-frame, blank-result and empty-input checks passed\n";
}

void testInvalidConfig(const app::AppConfig &app_config)
{
    bool rejected = false;
    try
    {
        ArmorDetector detector(app::makeYamlConfig(app_config, "missing_model"),
                               kSyncPipelineDelay);
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    require(rejected, "错误配置未被拒绝");
    std::cout << "invalid-config check passed\n";
}

void testMismatchedTask(const app::AppConfig &app_config)
{
    bool armor_rejected = false;
    try
    {
        ArmorDetector detector(app::makeYamlConfig(app_config, "rune_detect"),
                               kSyncPipelineDelay);
    }
    catch (const std::invalid_argument &)
    {
        armor_rejected = true;
    }
    require(armor_rejected, "ArmorDetector 未拒绝大符模型");

    bool rune_rejected = false;
    try
    {
        RuneDetector detector(app::makeYamlConfig(app_config, "armor_v5"),
                              kSyncPipelineDelay);
    }
    catch (const std::invalid_argument &)
    {
        rune_rejected = true;
    }
    require(rune_rejected, "RuneDetector 未拒绝装甲板模型");
    std::cout << "mismatched-task checks passed\n";
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const app::CommandLine cli(argc, argv, {"config", "c"});
        cv::CommandLineParser parser(cli.argc(), cli.argv(), app::commonKeys());
        parser.about("NNdeployment 冒烟测试（默认 tests/detector_smoke.yaml）");
        if (!app::handleParser(parser))
            return parser.has("help") ? 0 : 1;

        const app::AppConfig app_config = app::resolveAppConfig(
            parser, "detector_smoke.yaml", "src/app_plugin/detector/tests");

        std::cout << "配置文件: " << app_config.config_path.string()
                  << " | OpenCV CUDA 可用: " << (opencvCudaAvailable() ? "是" : "否")
                  << " | 预处理=" << app::describeCuda(app_config.cuda.preprocess)
                  << " 后处理=" << app::describeCuda(app_config.cuda.postprocess) << std::endl;

        const fs::path root = app_config.root;
        const cv::Mat armor_frame = readFrame(root / "测试视频/装甲板.mp4");
        const cv::Mat rune_frame = readFrame(root / "测试视频/符.avi");

        testArmor(app_config, "armor_v5", armor_frame);
        testArmor(app_config, "armor_v8", armor_frame);
        testRune(app_config, rune_frame);
        testInvalidConfig(app_config);
        testMismatchedTask(app_config);
        std::cout << "detector smoke test passed" << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "detector smoke test failed: " << error.what() << std::endl;
        return 1;
    }
}
