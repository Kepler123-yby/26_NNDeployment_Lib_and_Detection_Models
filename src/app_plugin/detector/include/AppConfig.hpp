#pragma once

#include <opencv2/core/utility.hpp>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// 应用级配置管理：统一使用 cv::CommandLineParser 解析命令行，
// 负责定位项目根目录、选择 YAML 配置文件，并解析 CUDA 覆盖开关。

#ifndef NNDEPLOYMENT_PROJECT_ROOT
#define NNDEPLOYMENT_PROJECT_ROOT "."
#endif

namespace app
{
// 把 `--config value` / `-c value` 归一化为 `--config=value`。
// cv::CommandLineParser 只识别 `--key=value`，这里兼容常见的空格形式。
inline std::vector<std::string> normalizeCommandLine(
    int argc, const char *const argv[], const std::vector<std::string> &value_keys)
{
    std::vector<std::string> normalized;
    normalized.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index)
    {
        std::string arg = argv[index];
        if (index > 0 && arg.size() > 1 && arg[0] == '-' &&
            arg.find('=') == std::string::npos)
        {
            std::string bare = arg;
            while (!bare.empty() && bare.front() == '-')
                bare.erase(bare.begin());
            for (const std::string &key : value_keys)
            {
                if (bare == key && index + 1 < argc)
                {
                    arg += "=";
                    arg += argv[++index];
                    break;
                }
            }
        }
        normalized.push_back(std::move(arg));
    }
    return normalized;
}

// 持有归一化后的命令行存储，保证传给 cv::CommandLineParser 的指针有效。
class CommandLine
{
public:
    CommandLine(int argc, const char *const argv[],
                const std::vector<std::string> &value_keys)
        : m_storage(normalizeCommandLine(argc, argv, value_keys))
    {
        m_argv.reserve(m_storage.size());
        for (const std::string &arg : m_storage)
            m_argv.push_back(arg.c_str());
    }

    int argc() const { return static_cast<int>(m_argv.size()); }
    const char *const *argv() const { return m_argv.data(); }

private:
    std::vector<std::string> m_storage;
    std::vector<const char *> m_argv;
};

// 公共命令行键：帮助、配置文件、CUDA 开关与位置参数（项目根目录）。
inline std::string commonKeys()
{
    return "{help h usage ?||显示本帮助}"
           "{config c||配置文件路径（默认 detect_openvino.yaml）}"
           "{cuda||同时启用 CUDA 预处理和后处理}"
           "{no-cuda||强制 CPU 预处理和后处理}"
           "{cuda-pre||仅启用 CUDA 预处理}"
           "{no-cuda-pre||禁用 CUDA 预处理}"
           "{cuda-post||仅启用 CUDA 后处理}"
           "{no-cuda-post||禁用 CUDA 后处理}"
           "{@root||项目根目录（默认取构建时记录的工程根）}";
}

// CUDA 运行时覆盖：未设置时使用 YAML 配置中的取值。
struct CudaOptions
{
    std::optional<bool> preprocess;
    std::optional<bool> postprocess;
};

// 从解析结果中提取 CUDA 开关，后出现的开关覆盖先出现的。
inline CudaOptions resolveCuda(const cv::CommandLineParser &parser)
{
    CudaOptions cuda;
    if (parser.has("cuda"))
    {
        cuda.preprocess = true;
        cuda.postprocess = true;
    }
    if (parser.has("no-cuda"))
    {
        cuda.preprocess = false;
        cuda.postprocess = false;
    }
    if (parser.has("cuda-pre"))
        cuda.preprocess = true;
    if (parser.has("no-cuda-pre"))
        cuda.preprocess = false;
    if (parser.has("cuda-post"))
        cuda.postprocess = true;
    if (parser.has("no-cuda-post"))
        cuda.postprocess = false;
    return cuda;
}

inline std::string describeCuda(const std::optional<bool> &value)
{
    if (!value.has_value())
        return "跟随配置";
    return *value ? "启用" : "禁用";
}

// 应用级配置：项目根目录、配置文件路径与 CUDA 覆盖。
struct AppConfig
{
    std::filesystem::path root;
    std::filesystem::path config_path;
    CudaOptions cuda;
};

// 解析位置参数 root、可选 --config 以及 CUDA 开关。
inline AppConfig resolveAppConfig(const cv::CommandLineParser &parser,
                                  const std::string &default_config_name)
{
    AppConfig config;

    const std::string root_arg = parser.get<std::string>(0);
    config.root = std::filesystem::absolute(
        root_arg.empty() ? std::filesystem::path(NNDEPLOYMENT_PROJECT_ROOT)
                         : std::filesystem::path(root_arg));

    const std::filesystem::path config_dir =
        config.root / "src/app_plugin/detector/config";
    const std::string config_arg = parser.get<std::string>("config");
    if (config_arg.empty())
    {
        config.config_path = config_dir / default_config_name;
    }
    else
    {
        const std::filesystem::path given(config_arg);
        std::error_code ec;
        if (given.is_absolute())
            config.config_path = given;
        else if (std::filesystem::exists(given, ec))
            config.config_path = std::filesystem::absolute(given); // 相对于当前工作目录
        else
            config.config_path = config_dir / given; // 只给文件名时到配置目录下查找
    }

    config.cuda = resolveCuda(parser);

    std::error_code ec;
    if (!std::filesystem::is_regular_file(config.config_path, ec))
        throw std::runtime_error("找不到配置文件: " + config.config_path.string() +
                                 "（可用 --config=<路径> 或 --config <路径> 指定）");
    return config;
}

// 统一处理 -h/--help 与解析错误；返回 false 表示调用方应直接退出。
inline bool handleParser(const cv::CommandLineParser &parser)
{
    if (parser.has("help"))
    {
        parser.printMessage();
        return false;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return false;
    }
    return true;
}
} // namespace app
