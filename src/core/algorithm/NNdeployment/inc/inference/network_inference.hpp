#pragma once

#include "network_deployment_lib.hpp"

#include <opencv2/core.hpp>

#ifndef NNDEPLOYMENT_WITH_OPENCV_CUDA
#define NNDEPLOYMENT_WITH_OPENCV_CUDA 0
#endif

class YOLOModel::InferenceEngine
{
public:
    InferenceEngine(const ModelConfig &model_config, const DebugConfig &debug_config);
    virtual ~InferenceEngine() = default;

    virtual int inputWidth() const = 0;
    virtual int inputHeight() const = 0;
    // 返回供当前推理请求写入的预处理缓冲区。
    virtual cv::Mat acquirePreprocessBuffer() { return {}; }
    // 使用当前后端的输入尺寸、坐标参数和请求缓冲区预处理图像。
    // 声明为虚函数，后端（如 TensorRT + OpenCV CUDA）可覆盖为设备端融合预处理。
    virtual cv::Mat preProcessImage(const cv::Mat &origin_image);
    const float *infer(const cv::Mat &pre_processed_image);
    virtual const float *syncInfer(const cv::Mat &pre_processed_image) = 0;
    virtual const float *asyncInfer(const cv::Mat &pre_processed_image) = 0;
    virtual const float *asyncInfer4(const cv::Mat &pre_processed_image) { throw std::logic_error("async4 infer not supported"); }

    InferParam m_infer_param;

protected:
#if NNDEPLOYMENT_WITH_OPENCV_CUDA
    // 使用 OpenCV CUDA 在 GPU 上完成缩放与 padding，写入 output_buffer。
    // 成功返回 true；CUDA 不可用时返回 false，由调用方回退到 CPU 路径。
    static bool cudaResizeAndPad(const cv::Mat &origin_image,
                                 int target_width, int target_height,
                                 int new_width, int new_height,
                                 int pad_x, int pad_y,
                                 cv::Mat &output_buffer);
#endif

    ModelConfig m_model_config;
    DebugConfig m_debug_config;
};
