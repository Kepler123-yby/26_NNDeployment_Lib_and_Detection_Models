#include "network_deployment_lib.hpp"
#include "network_inference.hpp"
#include "network_performance.hpp"
#include "network_postprocess.hpp"

#ifndef NNDEPLOYMENT_WITH_TENSORRT
#define NNDEPLOYMENT_WITH_TENSORRT 0
#endif

#ifndef NNDEPLOYMENT_WITH_OPENVINO
#define NNDEPLOYMENT_WITH_OPENVINO 1
#endif

#ifndef NNDEPLOYMENT_WITH_OPENCV_CUDA
#define NNDEPLOYMENT_WITH_OPENCV_CUDA 0
#endif

#if NNDEPLOYMENT_WITH_OPENVINO
#include "network_deployment_openvino.hpp"
#endif

#if NNDEPLOYMENT_WITH_TENSORRT
#include "network_deployment_tensorrt.hpp"
#endif

#if NNDEPLOYMENT_WITH_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#endif

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

// 总模块：YOLOModel 构造、部署/后处理选择以及完整推理入口。

// ==================== YOLOModel 实现 ====================

// namespace 中只包含字符串解析、枚举转换、配置读取和提示输出等局部工具，如需研究原理代码，无需着重研究此部分内容。
namespace
{
using NetDeployWay = YOLOModel::NetDeployWay;
using NetInferMode = YOLOModel::NetInferMode;
using NetPostProcessMode = YOLOModel::NetPostProcessMode;
using ModelConfig = YOLOModel::ModelConfig;

// 将字符串统一转换为小写。
std::string lowerString(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

using PerformanceClock = std::chrono::steady_clock;

// 计算两个时间点之间的微秒数。
double elapsedMicroseconds(PerformanceClock::time_point start,
                           PerformanceClock::time_point stop)
{
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

// 将字符串解析为推理模式。
NetInferMode parseInferMode(const std::string &value)
{
    const std::string mode = lowerString(value);
    if (mode.empty() || mode == "async")
        return NetInferMode::async;
    if (mode == "sync")
        return NetInferMode::sync;
    if (mode == "async4")
        return NetInferMode::async4;
    throw std::runtime_error("不支持的infer_mode: " + value);
}

// 将字符串解析为部署后端。
NetDeployWay parseDeployWay(const std::string &value)
{
    const std::string way = lowerString(value);
    if (way.empty() || way == "openvino")
        return NetDeployWay::openvino;
    if (way == "tensorrt")
        return NetDeployWay::tensorrt;
    throw std::runtime_error("不支持的deploy_way: " + value);
}

// 将字符串解析为后处理模式。
NetPostProcessMode parsePostProcessMode(const std::string &value)
{
    const std::string mode = lowerString(value);
    if (mode.empty() || mode == "auto" || mode == "auto_detect")
        return NetPostProcessMode::auto_detect;
    if (mode == "v5infantry" || mode == "v5infantry_fourpoints")
        return NetPostProcessMode::v5infantry_fourpoints;
    if (mode == "v8infantry" || mode == "v8infantry_fourpoints")
        return NetPostProcessMode::v8infantry_fourpoints;
    if (mode == "v8infantry_21" || mode == "v8infantry_fourpoints_21")
        return NetPostProcessMode::v8infantry_fourpoints_21;
    if (mode == "lidar" || mode == "lidar_fourpoints")
        return NetPostProcessMode::lidar_fourpoints;
    if (mode == "rune" || mode == "rune_fivepoints")
        return NetPostProcessMode::rune_fivepoints;
    throw std::runtime_error("不支持的postprocess_mode: " + value);
}

// 将部署后端转换为可打印字符串。
std::string toString(NetDeployWay way)
{
    switch (way)
    {
    case NetDeployWay::openvino:
        return "openvino";
    case NetDeployWay::tensorrt:
        return "tensorrt";
    default:
        return "error type";
    }
}

// 将推理模式转换为可打印字符串。
std::string toString(NetInferMode way)
{
    switch (way)
    {
    case NetInferMode::sync:
        return "sync";
    case NetInferMode::async:
        return "async";
    case NetInferMode::async4:
        return "async4";
    default:
        return "error type";
    }
}

// 将后处理模式转换为可打印字符串。
std::string toString(NetPostProcessMode way)
{
    switch (way)
    {
    case NetPostProcessMode::auto_detect:
        return "auto";
    case NetPostProcessMode::v5infantry_fourpoints:
        return "v5infantry";
    case NetPostProcessMode::v8infantry_fourpoints:
        return "v8infantry";
    case NetPostProcessMode::v8infantry_fourpoints_21:
        return "v8infantry_21";
    case NetPostProcessMode::lidar_fourpoints:
        return "lidar";
    case NetPostProcessMode::rune_fivepoints:
        return "rune";
    default:
        return "error type";
    }
}

// 打印当前模型配置及非标准组合提示。
void printCurrentModelInfo(const ModelConfig &model_config)
{
    std::cout << "兵种：" << toString(model_config.postprocess_mode) << " "
              << "推理模式：" << toString(model_config.infer_mode) << " "
              << "部署方式：" << toString(model_config.deploy_way) << std::endl;
    std::cout << "预处理：" << (model_config.preprocess_cuda ? "CUDA" : "CPU")
              << " 后处理：" << (model_config.postprocess_cuda ? "CUDA" : "CPU") << std::endl;

    switch (model_config.postprocess_mode)
    {
    case NetPostProcessMode::auto_detect:
        break;
    case NetPostProcessMode::v5infantry_fourpoints:
        break;
    case NetPostProcessMode::v8infantry_fourpoints:
    case NetPostProcessMode::v8infantry_fourpoints_21:
        break;
    case NetPostProcessMode::lidar_fourpoints:
        if (model_config.infer_mode != NetInferMode::sync || model_config.deploy_way != NetDeployWay::tensorrt)
            std::cout << "注：标准情况下雷达应该使用" << toString(NetDeployWay::tensorrt) << "部署，" << toString(NetInferMode::sync) << "推理" << std::endl;
        break;
    case NetPostProcessMode::rune_fivepoints:
        if (model_config.infer_mode != NetInferMode::sync || model_config.deploy_way != NetDeployWay::openvino)
            std::cout << "注：标准情况下打符应该使用" << toString(NetDeployWay::openvino) << "部署，" << toString(NetInferMode::sync) << "推理" << std::endl;
        break;
    }
    std::cout << std::endl;
}

// 解析模型文件路径。
// 模型仓库根目录（"所有模型"）下按后端分目录（openvino/、onnx/、tensorrt/），
// model_folder 推荐直接传仓库根（"所有模型"），xml 里带上后端子目录（如 "onnx/xxx.onnx"）；
// 同时兼容旧写法（model_folder 传某个后端子目录，xml 只写文件名）。
// 为避免把查找根目录写死，依次尝试：
//   1) model_folder / name；
//   2) model_folder 的父目录（模型仓库根）/ name，如 name="onnx/xxx.onnx"；
//   3) 在模型仓库根目录下按文件名递归查找，如 name="xxx.onnx"。
std::string resolveModelPath(const std::string &model_folder, const std::string &name)
{
    const std::filesystem::path folder(model_folder);
    const std::filesystem::path name_path(name);
    const std::vector<std::filesystem::path> candidates = {
        folder / name_path, folder.parent_path() / name_path};

    for (const std::filesystem::path &candidate : candidates)
    {
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec))
            return std::filesystem::weakly_canonical(candidate, ec).string();
    }

    // 在模型仓库根目录（model_folder 的父目录）下按文件名递归查找。
    const std::filesystem::path repository_root = folder.parent_path();
    std::error_code ec;
    if (!repository_root.empty() &&
        std::filesystem::is_directory(repository_root, ec))
    {
        for (std::filesystem::recursive_directory_iterator it(
                 repository_root,
                 std::filesystem::directory_options::skip_permission_denied, ec),
             end;
             it != end; it.increment(ec))
        {
            if (ec)
                break;
            std::error_code file_ec;
            if (it->is_regular_file(file_ec) &&
                it->path().filename() == name_path.filename())
                return std::filesystem::weakly_canonical(it->path(), file_ec).string();
        }
    }

    // 未找到时返回默认拼接结果，交由具体后端给出明确的加载错误。
    return (folder / name_path).lexically_normal().string();
}

// 读取 YAML 节点的可选字段，缺省时返回 fallback。
template <typename T>
T yamlValueOr(const YAML::Node &node, const char *key, const T &fallback)
{
    return node[key] ? node[key].as<T>() : fallback;
}

// 从 YAML 的指定节点读取内部模型配置。
ModelConfig loadModelConfigFromYaml(const YamlConfig &yaml_config)
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile(yaml_config.yaml_path);
    }
    catch (const YAML::Exception &error)
    {
        throw std::runtime_error("无法打开模型配置文件: " + yaml_config.yaml_path + " (" + error.what() + ")");
    }

    const YAML::Node node = root[yaml_config.model_key];
    if (!node)
        throw std::runtime_error("模型配置中不存在键: " + yaml_config.model_key);

    if (yaml_config.model_folder.empty())
        throw std::runtime_error("传入的模型文件夹路径为空: " + yaml_config.model_key);

    if (!node["xml"])
        throw std::runtime_error("模型配置缺少xml文件名: " + yaml_config.model_key);
    const std::string xml_name = node["xml"].as<std::string>();
    if (xml_name.empty())
        throw std::runtime_error("模型配置缺少xml文件名: " + yaml_config.model_key);

    const std::string model_path =
        resolveModelPath(yaml_config.model_folder, xml_name);

    const NetInferMode infer_mode = parseInferMode(yamlValueOr<std::string>(node, "infer_mode", ""));
    const NetDeployWay deploy_way = parseDeployWay(yamlValueOr<std::string>(node, "deploy_way", ""));
    const NetPostProcessMode postprocess_mode = parsePostProcessMode(yamlValueOr<std::string>(node, "postprocess_mode", ""));

    std::string device = yamlValueOr<std::string>(node, "device", "");
    if (device.empty())
        device = "GPU";

    float confidence_threshold = 0.5f;
    if (node["score_threshold"])
        confidence_threshold = node["score_threshold"].as<float>();
    else if (node["config_thresh"])                        // 兼容旧版本配置文件
        confidence_threshold = node["config_thresh"].as<float>();

    // 可选：OpenCV CUDA 加速开关。未填写时保持 CPU。
    // 支持 preprocess_cuda / postprocess_cuda 单独开关，也支持 cuda 一键开关。
    const bool cuda_all = node["cuda"] && node["cuda"].as<bool>();
    bool preprocess_cuda = cuda_all;
    bool postprocess_cuda = cuda_all;
    if (node["preprocess_cuda"])
        preprocess_cuda = node["preprocess_cuda"].as<bool>();
    if (node["postprocess_cuda"])
        postprocess_cuda = node["postprocess_cuda"].as<bool>();

    // YamlConfig 中的覆盖值优先于配置文件（供调用方在运行时管理 CUDA）。
    if (yaml_config.preprocess_cuda.has_value())
        preprocess_cuda = *yaml_config.preprocess_cuda;
    if (yaml_config.postprocess_cuda.has_value())
        postprocess_cuda = *yaml_config.postprocess_cuda;

    return ModelConfig(model_path, infer_mode, deploy_way, device, confidence_threshold,
                       postprocess_mode, preprocess_cuda, postprocess_cuda);
}
} // namespace

YOLOModel::InferenceEngine::InferenceEngine(const ModelConfig &model_config, const DebugConfig &debug_config) : m_model_config(model_config), m_debug_config(debug_config) {}

const float *YOLOModel::InferenceEngine::infer(const cv::Mat &pre_processed_image)
{
    switch (m_model_config.infer_mode)
    {
    case NetInferMode::sync:
        return syncInfer(pre_processed_image);
    case NetInferMode::async:
        return asyncInfer(pre_processed_image);
    case NetInferMode::async4:
        return asyncInfer4(pre_processed_image);
    }
    throw std::runtime_error("不支持的推理模式");
}

bool YOLOModel::cudaAvailable()
{
#if NNDEPLOYMENT_WITH_OPENCV_CUDA
    try
    {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    }
    catch (const std::exception &)
    {
        return false;
    }
#else
    return false;
#endif
}

float YOLOModel::defaultNmsThreshold(NetPostProcessMode mode)
{
    switch (mode)
    {
    case NetPostProcessMode::rune_fivepoints:
        return 100.f;   // 符模型 NMS 阈值为像素距离
    case NetPostProcessMode::v5infantry_fourpoints:
    case NetPostProcessMode::v8infantry_fourpoints:
    case NetPostProcessMode::v8infantry_fourpoints_21:
    case NetPostProcessMode::lidar_fourpoints:
    case NetPostProcessMode::auto_detect:
    default:
        return 0.2f;    // armor模型 NMS 阈值为 IOU 比例
    }
}

YOLOModel::YOLOModel(const std::string &model_path)
    : YOLOModel(model_path,
                "async",
                "openvino",
                "GPU",
                0.5f,
                "auto_detect",
                DebugConfig())
{}

YOLOModel::YOLOModel(const std::string &model_path,
                     const std::string &infer_mode,
                     const std::string &deploy_way,
                     const std::string &device,
                     float confidence_threshold,
                     const std::string &postprocess_mode,
                     const DebugConfig &debug_config,
                     bool preprocess_cuda,
                     bool postprocess_cuda)
    : YOLOModel(ModelConfig(model_path,
                            parseInferMode(infer_mode),
                            parseDeployWay(deploy_way),
                            device,
                            confidence_threshold,
                            parsePostProcessMode(postprocess_mode),
                            preprocess_cuda,
                            postprocess_cuda),
                debug_config)
{}

YOLOModel::YOLOModel(const YamlConfig &yaml_config,
                     const DebugConfig &debug_config)
    : YOLOModel(loadModelConfigFromYaml(yaml_config), debug_config)
{}

YOLOModel::~YOLOModel() noexcept = default;

YOLOModel::YOLOModel(const ModelConfig &model_config, const DebugConfig &debug_config) : m_model_config(model_config), m_debug_config(debug_config)
{
    if (m_debug_config.calculate_speed_info)
        m_speed_stats = std::make_unique<MPT::SpeedStats>();

    // 配置请求了 OpenCV CUDA 但在当前构建/设备上不可用时，回退到 CPU 并给出提示。
    if (m_model_config.preprocess_cuda || m_model_config.postprocess_cuda)
    {
        if (!cudaAvailable())
        {
            std::cout << "注：配置请求了 OpenCV CUDA 加速，但当前构建或设备不支持，已回退到 CPU。" << std::endl;
            m_model_config.preprocess_cuda = false;
            m_model_config.postprocess_cuda = false;
        }
    }

    // 根据部署方式选择openvino/tensorrt，目前仅雷达会使用tensorrt
    switch (m_model_config.deploy_way)
    {
    case NetDeployWay::openvino:
#if NNDEPLOYMENT_WITH_OPENVINO
        m_inference_engine = std::make_unique<OpenVINOEngine>(m_model_config, m_debug_config);
#else
        throw std::runtime_error("OpenVINO support is disabled at build time");
#endif
        break;
    case NetDeployWay::tensorrt:
#if NNDEPLOYMENT_WITH_TENSORRT
        m_inference_engine = std::make_unique<TensorRTEngine>(m_model_config, m_debug_config);
#else
        throw std::runtime_error("TensorRT support is disabled at build time");
#endif
        break;
    }

    // 根据后处理模式决定后处理模块，大部分情况下使用auto即可，除非模型结构改了
    switch (m_model_config.postprocess_mode)
    {
    case NetPostProcessMode::auto_detect:
        {
            const int rows = m_inference_engine->m_infer_param.out_tensor_rows;
            const int cols = m_inference_engine->m_infer_param.out_tensor_cols;
            if (rows == 25200 && cols == 22)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::v5infantry_fourpoints;
                m_postprocessor = std::make_unique<V5InfantryPostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else if (rows == 25 && cols == 6300)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::v8infantry_fourpoints;
                m_postprocessor = std::make_unique<V8InfantryPostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else if (rows == 21 && cols == 6300)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::v8infantry_fourpoints_21;
                m_postprocessor = std::make_unique<V8_21InfantryPostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else if (rows == 18 && cols == 6300)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::rune_fivepoints;
                m_postprocessor = std::make_unique<RunePostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else
            {
                throw std::runtime_error("无法自动识别模型输出格式: [" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
            }
        }
        break;
    case NetPostProcessMode::v5infantry_fourpoints:
        m_postprocessor = std::make_unique<V5InfantryPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::v8infantry_fourpoints:
        m_postprocessor = std::make_unique<V8InfantryPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::v8infantry_fourpoints_21:
        m_postprocessor = std::make_unique<V8_21InfantryPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::lidar_fourpoints:
        m_postprocessor = std::make_unique<LidarPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::rune_fivepoints:
        m_postprocessor = std::make_unique<RunePostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    }

    // 调用namespace中的函数打印使用的模型信息，这部分不需要debug开启也打印
    printCurrentModelInfo(m_model_config);
}

//  完整的网络识别框架——主要流程：传入图片->预处理->推理->后处理->输出结果
std::vector<NetArmorResult> YOLOModel::netProcess(const cv::Mat &input_image, const int &my_color)
{
    // debug 的测速选项开启时进行各部分的测速。
    const bool measure = m_speed_stats != nullptr;
    PerformanceClock::time_point phase_start{};
    double preprocess_us = 0.0;
    double inference_us = 0.0;
    double postprocess_us = 0.0;

    // 预处理
    if (measure)
        phase_start = PerformanceClock::now();
    cv::Mat pre_processed_mat = m_inference_engine->preProcessImage(input_image);
    if (measure)
        preprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());
    if (pre_processed_mat.empty())
        return {};

    // 推理
    if (measure)
        phase_start = PerformanceClock::now();
    const float *infer_output = m_inference_engine->infer(pre_processed_mat);
    if (measure)
        inference_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 后处理
    if (measure)
        phase_start = PerformanceClock::now();
    std::vector<NetArmorResult> results = m_postprocessor->postProcessArmorMat(
        infer_output, m_inference_engine->m_infer_param, my_color);
    if (measure)
        postprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 输出计时结果
    if (measure)
    {
        m_speed_stats->update(preprocess_us, inference_us, postprocess_us);
        m_speed_stats->printCurrentStats();
    }
    return results;
}

std::vector<NetRuneResult> YOLOModel::netProcess(const cv::Mat &input_image)
{
    const bool measure = m_speed_stats != nullptr;
    PerformanceClock::time_point phase_start{};
    double preprocess_us = 0.0;
    double inference_us = 0.0;
    double postprocess_us = 0.0;

    // 预处理
    if (measure)
        phase_start = PerformanceClock::now();
    cv::Mat pre_processed_mat = m_inference_engine->preProcessImage(input_image);
    if (measure)
        preprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());
    if (pre_processed_mat.empty())
        return {};

    // 推理
    if (measure)
        phase_start = PerformanceClock::now();
    const float *infer_output = m_inference_engine->infer(pre_processed_mat);
    if (measure)
        inference_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 后处理
    if (measure)
        phase_start = PerformanceClock::now();
    std::vector<NetRuneResult> results = m_postprocessor->postProcessRuneMat(
        infer_output, m_inference_engine->m_infer_param);
    if (measure)
        postprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 输出计时结果
    if (measure)
    {
        m_speed_stats->update(preprocess_us, inference_us, postprocess_us);
        m_speed_stats->printCurrentStats();
    }
    return results;
}

bool YOLOModel::supportsArmor() const
{
    return m_model_config.postprocess_mode == NetPostProcessMode::v5infantry_fourpoints ||
           m_model_config.postprocess_mode == NetPostProcessMode::v8infantry_fourpoints ||
           m_model_config.postprocess_mode == NetPostProcessMode::v8infantry_fourpoints_21 ||
           m_model_config.postprocess_mode == NetPostProcessMode::lidar_fourpoints;
}

bool YOLOModel::supportsRune() const
{
    return m_model_config.postprocess_mode == NetPostProcessMode::rune_fivepoints;
}
