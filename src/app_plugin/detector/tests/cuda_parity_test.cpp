#include "NNDetector.hpp"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// CPU 与 OpenCV CUDA 两条后处理路径的一致性测试。
// 前提：使用带 OpenCV CUDA 的构建（-DNNDEPLOYMENT_ENABLE_OPENCV_CUDA=ON）。

#ifndef NNDEPLOYMENT_PROJECT_ROOT
#define NNDEPLOYMENT_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

namespace
{
constexpr std::size_t kDelay = 0; // 同步推理，结果与输入帧一一对应。
constexpr int kFrames = 20;       // 每个视频抽检的帧数。

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

cv::VideoCapture openVideo(const fs::path &path)
{
    cv::VideoCapture video(path.string());
    require(video.isOpened(), "无法打开测试视频: " + path.string());
    return video;
}

struct ArmorRecord
{
    int armor_id = -1;
    int color_id = -1;
    double score = 0.0;
    double coords[8] = {};
    bool used = false;
};

std::vector<ArmorRecord> toRecords(const std::vector<NetArmorResult> &results)
{
    std::vector<ArmorRecord> records;
    records.reserve(results.size());
    for (const NetArmorResult &result : results)
    {
        if (result.points.size() != 4)
            continue;
        ArmorRecord record;
        record.armor_id = result.armor_id;
        record.color_id = result.color_id;
        record.score = result.score;
        for (std::size_t index = 0; index < 4; ++index)
        {
            record.coords[index * 2 + 0] = result.points[index].x;
            record.coords[index * 2 + 1] = result.points[index].y;
        }
        records.push_back(record);
    }
    return records;
}

// 在参考结果中为每个待匹配结果寻找类别一致、坐标与分数接近的候选。
bool matchRecords(const std::vector<ArmorRecord> &reference,
                  const std::vector<ArmorRecord> &candidates,
                  double point_tolerance, double score_tolerance,
                  std::string &reason)
{
    if (reference.size() != candidates.size())
    {
        reason = "结果数量不一致: cpu=" + std::to_string(reference.size()) +
                 " cuda=" + std::to_string(candidates.size());
        return false;
    }

    std::vector<ArmorRecord> pool = candidates;
    for (const ArmorRecord &expected : reference)
    {
        bool matched = false;
        for (ArmorRecord &actual : pool)
        {
            if (actual.used || actual.armor_id != expected.armor_id ||
                actual.color_id != expected.color_id)
                continue;

            double max_distance = 0.0;
            for (int index = 0; index < 8; ++index)
                max_distance = std::max(max_distance, std::abs(actual.coords[index] - expected.coords[index]));

            if (max_distance <= point_tolerance &&
                std::abs(actual.score - expected.score) <= score_tolerance)
            {
                actual.used = true;
                matched = true;
                break;
            }
        }
        if (!matched)
        {
            reason = "存在无法匹配的装甲板结果 (id=" + std::to_string(expected.armor_id) +
                     " color=" + std::to_string(expected.color_id) +
                     " score=" + std::to_string(expected.score) + ")";
            return false;
        }
    }
    return true;
}

void compareArmorVideo(const fs::path &root, const std::string &key,
                       const fs::path &video_path, const std::string &cuda_config_name,
                       double point_tolerance, double score_tolerance)
{
    const fs::path models = root / "所有模型";
    const fs::path cpu_config = root / "src/app_plugin/detector/tests/cuda_parity_cpu.yaml";
    const fs::path cuda_config = root / "src/app_plugin/detector/tests" / cuda_config_name;

    ArmorDetector cpu(YamlConfig{cpu_config.string(), key, models.string()}, kDelay);
    ArmorDetector cuda(YamlConfig{cuda_config.string(), key, models.string()}, kDelay);

    cv::VideoCapture video = openVideo(video_path);
    cv::Mat frame;
    int compared = 0;
    while (compared < kFrames && video.read(frame) && !frame.empty())
    {
        auto cpu_output = cpu.process(frame, 2);
        auto cuda_output = cuda.process(frame, 2);
        require(cpu_output.has_value() && cuda_output.has_value(), key + " 推理未返回结果");

        std::vector<ArmorRecord> reference = toRecords(cpu_output->results);
        std::vector<ArmorRecord> candidate = toRecords(cuda_output->results);
        std::string reason;
        require(matchRecords(reference, candidate, point_tolerance, score_tolerance, reason),
                key + " 第" + std::to_string(compared) + "帧不一致: " + reason);
        ++compared;
    }
    std::cout << key << ": CUDA 与 CPU 后处理一致性通过，共比较 " << compared << " 帧" << std::endl;
}

void compareRuneVideo(const fs::path &root, const fs::path &video_path,
                      const std::string &cuda_config_name, double point_tolerance)
{
    const fs::path models = root / "所有模型";
    const fs::path cpu_config = root / "src/app_plugin/detector/tests/cuda_parity_cpu.yaml";
    const fs::path cuda_config = root / "src/app_plugin/detector/tests" / cuda_config_name;

    RuneDetector cpu(YamlConfig{cpu_config.string(), "rune_detect", models.string()}, kDelay);
    RuneDetector cuda(YamlConfig{cuda_config.string(), "rune_detect", models.string()}, kDelay);

    cv::VideoCapture video = openVideo(video_path);
    cv::Mat frame;
    int compared = 0;
    while (compared < kFrames && video.read(frame) && !frame.empty())
    {
        auto cpu_output = cpu.process(frame);
        auto cuda_output = cuda.process(frame);
        require(cpu_output.has_value() && cuda_output.has_value(), "rune_detect 推理未返回结果");
        require(cpu_output->results.size() == cuda_output->results.size(),
                "rune_detect 第" + std::to_string(compared) + "帧结果数量不一致");

        // NMS 对相同置信度的候选顺序可能不一致，这里按内容排序后比较。
        const auto make_keys = [point_tolerance](const std::vector<NetRuneResult> &results)
        {
            std::vector<std::string> keys;
            keys.reserve(results.size());
            for (const NetRuneResult &result : results)
            {
                std::string key = std::to_string(result.class_id);
                for (const cv::Point2d &point : result.points)
                {
                    const double x = std::round(point.x / point_tolerance) * point_tolerance;
                    const double y = std::round(point.y / point_tolerance) * point_tolerance;
                    key += "|" + std::to_string(x) + "," + std::to_string(y);
                }
                keys.push_back(std::move(key));
            }
            std::sort(keys.begin(), keys.end());
            return keys;
        };

        require(make_keys(cpu_output->results) == make_keys(cuda_output->results),
                "rune_detect 第" + std::to_string(compared) + "帧结果不一致");
        ++compared;
    }
    std::cout << "rune_detect: CUDA 与 CPU 后处理一致性通过，共比较 " << compared << " 帧" << std::endl;
}
} // namespace

int main()
{
    try
    {
        const fs::path root = fs::path(NNDEPLOYMENT_PROJECT_ROOT);

        // 仅后处理 CUDA：输入张量完全相同，要求近乎精确一致。
        compareArmorVideo(root, "armor_v5", root / "测试视频/装甲板.mp4", "cuda_parity_cuda.yaml", 0.05, 0.02);
        compareArmorVideo(root, "armor_v8", root / "测试视频/装甲板.mp4", "cuda_parity_cuda.yaml", 0.05, 0.02);
        compareRuneVideo(root, root / "测试视频/符.avi", "cuda_parity_cuda.yaml", 0.05);

        // 预处理 + 后处理均走 CUDA：缩放实现存在亚像素差异，放宽到像素级容差。
        compareArmorVideo(root, "armor_v5", root / "测试视频/装甲板.mp4", "cuda_parity_full.yaml", 2.0, 0.1);
        compareArmorVideo(root, "armor_v8", root / "测试视频/装甲板.mp4", "cuda_parity_full.yaml", 2.0, 0.1);
        compareRuneVideo(root, root / "测试视频/符.avi", "cuda_parity_full.yaml", 2.0);

        std::cout << "cuda parity test passed" << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "cuda parity test failed: " << error.what() << std::endl;
        return 1;
    }
}
