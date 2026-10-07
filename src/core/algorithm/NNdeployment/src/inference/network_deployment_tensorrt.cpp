#include "network_deployment_tensorrt.hpp"

#if NNDEPLOYMENT_WITH_TENSORRT

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#if NNDEPLOYMENT_WITH_OPENCV_CUDA
#include "network_preprocess_cuda.hpp"
#endif

namespace
{
void printTensorInfo(const nvinfer1::ICudaEngine *engine)
{
    if (!engine)
    {
        std::cerr << "错误: 引擎未初始化" << std::endl;
        return;
    }

    std::cout << "=== 模型张量信息 ===" << '\n';
    for (int i = 0; i < 2; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType dtype = engine->getTensorDataType(name);

        std::cout << "张量 " << i << ":" << '\n';
        std::cout << "  名称: " << name << '\n';
        std::cout << "  类型: " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "输入" : "输出") << '\n';
        std::cout << "  形状: [";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1)
                std::cout << ", ";
        }
        std::cout << "]" << '\n';

        std::cout << "  数据类型: ";
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            std::cout << "float32";
            break;
        case nvinfer1::DataType::kHALF:
            std::cout << "float16";
            break;
        case nvinfer1::DataType::kINT8:
            std::cout << "int8";
            break;
        case nvinfer1::DataType::kINT32:
            std::cout << "int32";
            break;
        case nvinfer1::DataType::kBOOL:
            std::cout << "bool";
            break;
        default:
            std::cout << "unknown";
        }
        std::cout << '\n' << '\n';
    }
    std::cout.flush();
}

// 读取二进制文件到内存。
std::vector<char> readBinaryFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("找不到 TensorRT 模型: " + path);

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<std::size_t>(size));
    if (size > 0 && !file.read(data.data(), size))
        throw std::runtime_error("读取 TensorRT 模型失败: " + path);
    return data;
}

// 将序列化 engine 写入磁盘缓存；写失败时静默跳过（不影响本次推理）。
void writeBinaryFile(const std::string &path, const void *data, std::size_t size)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
        return;
    file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
}

// 用 TensorRT 自带的 ONNX Parser 解析 .onnx 并现场构建序列化 engine。
std::vector<char> buildEngineFromOnnx(const std::string &onnx_path, nvinfer1::ILogger &logger)
{
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
        throw std::runtime_error("无法创建 TensorRT Builder");

    const std::uint32_t explicit_batch =
        1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(explicit_batch));
    if (!network)
        throw std::runtime_error("无法创建 TensorRT Network");

    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser)
        throw std::runtime_error("无法创建 TensorRT ONNX Parser");

    if (!parser->parseFromFile(
            onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("解析 ONNX 模型失败: " + onnx_path);

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
        throw std::runtime_error("无法创建 TensorRT BuilderConfig");

    // 本仓库模型为 fp16，允许 TensorRT 使用半精度，与 trtexec --fp16 行为一致。
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
        throw std::runtime_error("构建 TensorRT engine 失败: " + onnx_path);

    const char *begin = static_cast<const char *>(serialized->data());
    return std::vector<char>(begin, begin + serialized->size());
}
} // namespace

TensorRTEngine::TensorRTEngine(const YOLOModel::ModelConfig &model_config, const DebugConfig &debug_config) : InferenceEngine(model_config, debug_config)
{
    // ==================== 初始化推理引擎 ====================
    // .onnx 直接走 TensorRT ONNX Parser 现场构建；其它文件按序列化 engine 加载。
    const std::filesystem::path model_path(m_model_config.model_path);
    const std::string extension = model_path.extension().string();
    const bool is_onnx = extension == ".onnx" || extension == ".ONNX";

    std::vector<char> engine_data;
    if (is_onnx)
    {
        // 同名 .engine 作为构建缓存：存在且不早于 onnx 时直接复用。
        std::filesystem::path cache_path = model_path;
        cache_path.replace_extension(".engine");

        std::error_code ec;
        const bool cache_valid =
            std::filesystem::is_regular_file(cache_path, ec) &&
            std::filesystem::last_write_time(cache_path, ec) >=
                std::filesystem::last_write_time(model_path, ec);

        if (cache_valid)
        {
            engine_data = readBinaryFile(cache_path.string());
        }
        else
        {
            if (m_debug_config.print_debug_info)
                std::cout << "从 ONNX 构建 TensorRT engine: " << model_path << std::endl;
            engine_data = buildEngineFromOnnx(model_path.string(), m_logger);
            writeBinaryFile(cache_path.string(), engine_data.data(), engine_data.size());
        }
    }
    else
    {
        engine_data = readBinaryFile(m_model_config.model_path);
    }

    const size_t file_size = engine_data.size();

    // 创建TensorRT运行时对象
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime)
        throw std::runtime_error("无法创建 TensorRT 运行时");

    // 反序列化引擎数据,生成推理引擎
    m_engine.reset(m_runtime->deserializeCudaEngine(engine_data.data(), file_size));
    if (!m_engine)
        throw std::runtime_error("无法反序列化 TensorRT 模型: " + m_model_config.model_path);

    // 创建执行上下文
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context)
        throw std::runtime_error("无法创建 TensorRT 执行上下文");

    // 写入输入输出张量大小
    m_target_width = m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).d[3];
    m_target_height = m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).d[2];

    m_output_anchors = m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[2];

    m_infer_param.out_tensor_rows = m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[1];
    m_infer_param.out_tensor_cols = m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[2];

    m_input_volume = 1 * sizeof(float);
    for (int i = 0; i < m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).nbDims; i++)
    {
        m_input_volume *= m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).d[i];
    }
    m_output_volume = 1 * sizeof(float);
    for (int i = 0; i < m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).nbDims; i++)
    {
        m_output_volume *= m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[i];
    }

    // ==================== 分配推理缓冲区 ====================
    // 获取输入张量的形状并设置
    nvinfer1::Dims inputDims = m_engine->getTensorShape("images");
    m_context->setInputShape("images", inputDims);

    // 分配GPU显存用于输入输出
    cudaMalloc(&m_buffers[m_input_index], m_input_volume);
    cudaMalloc(&m_buffers[m_output_index], m_output_volume);

    // 绑定显存地址到TensorRT上下文
    m_context->setTensorAddress("images", m_buffers[m_input_index]);
    m_context->setTensorAddress("output0", m_buffers[m_output_index]);

    // 分配页锁定内存用于存储输入输出数据（提高传输效率）
    cudaMallocHost(reinterpret_cast<void **>(&m_rst), m_output_volume);
    cudaMallocHost(&m_blob_pinned, m_input_volume);

    // 创建CUDA流用于异步操作
    cudaStreamCreate(&m_cuda_stream);

    if (m_debug_config.print_debug_info)
        printTensorInfo(m_engine.get());
}

int TensorRTEngine::inputWidth() const
{
    return m_target_width;
}

int TensorRTEngine::inputHeight() const
{
    return m_target_height;
}

// 启用 preprocess_cuda 时，在 GPU 上融合完成 letterbox + BGR→RGB + /255 + HWC→CHW，
// 直接写入 TensorRT 输入显存并回填坐标参数；否则使用通用 CPU 预处理。
cv::Mat TensorRTEngine::preProcessImage(const cv::Mat &origin_image)
{
#if NNDEPLOYMENT_WITH_OPENCV_CUDA
    if (m_model_config.preprocess_cuda)
    {
        if (MPT::cudaFuseTensorRTInput(origin_image, m_target_width, m_target_height,
                                       static_cast<float *>(m_buffers[m_input_index]),
                                       m_cuda_stream, m_infer_param))
        {
            m_input_on_device = true;
            // 占位返回：infer 会直接使用设备端输入，不会读取该返回值。
            return origin_image;
        }
    }
#endif
    m_input_on_device = false;
    return YOLOModel::InferenceEngine::preProcessImage(origin_image);
}

const float *TensorRTEngine::syncInfer(const cv::Mat &pre_processed_image)
{
    if (!m_input_on_device)
    {
        cv::dnn::blobFromImage(pre_processed_image, m_blob, 1 / 255.0,
                               cv::Size(m_target_width, m_target_height),
                               cv::Scalar(0, 0, 0), true, false, CV_32F);

        memcpy(m_blob_pinned, m_blob.data, m_blob.total() * m_blob.elemSize());

        cudaMemcpyAsync(m_buffers[m_input_index], m_blob_pinned,
                        m_input_volume, cudaMemcpyHostToDevice, m_cuda_stream);
    }

    m_context->enqueueV3(m_cuda_stream);

    cudaMemcpyAsync(m_rst, m_buffers[m_output_index],
                    m_output_volume, cudaMemcpyDeviceToHost, m_cuda_stream);

    cudaStreamSynchronize(m_cuda_stream);

    m_input_on_device = false;
    return m_rst;
}

const float *TensorRTEngine::asyncInfer(const cv::Mat &pre_processed_image)
{
    if (!m_input_on_device)
    {
        cv::dnn::blobFromImage(pre_processed_image, m_blob, 1 / 255.0,
                               cv::Size(m_target_width, m_target_height),
                               cv::Scalar(0, 0, 0), true, false, CV_32F);

        memcpy(m_blob_pinned, m_blob.data, m_blob.total() * m_blob.elemSize());

        cudaMemcpyAsync(m_buffers[m_input_index], m_blob_pinned,
                        m_input_volume, cudaMemcpyHostToDevice, m_cuda_stream);
    }

    m_context->enqueueV3(m_cuda_stream);

    cudaMemcpyAsync(m_rst, m_buffers[m_output_index],
                    m_output_volume, cudaMemcpyDeviceToHost, m_cuda_stream);

    cudaStreamSynchronize(m_cuda_stream);

    m_input_on_device = false;
    return m_rst;
}

TensorRTEngine::~TensorRTEngine()
{
    if (m_cuda_stream)
        cudaStreamDestroy(m_cuda_stream);
    if (m_blob_pinned)
        cudaFreeHost(m_blob_pinned);
    if (m_rst)
        cudaFreeHost(m_rst);
    if (m_buffers[m_input_index])
        cudaFree(m_buffers[m_input_index]);
    if (m_buffers[m_output_index])
        cudaFree(m_buffers[m_output_index]);
}

#endif
