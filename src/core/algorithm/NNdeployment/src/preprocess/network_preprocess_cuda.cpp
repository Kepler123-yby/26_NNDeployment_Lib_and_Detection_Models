#include "network_inference.hpp"

#if NNDEPLOYMENT_WITH_OPENCV_CUDA

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <iostream>

// ==================== 预处理模块（OpenCV CUDA 实现） ====================
// 使用 OpenCV CUDA 在 GPU 上完成缩放与 padding，再下载回 host 侧的预处理缓冲区。
// 该文件仅在构建时启用 NNDEPLOYMENT_WITH_OPENCV_CUDA 时参与编译。

bool YOLOModel::InferenceEngine::cudaResizeAndPad(const cv::Mat &origin_image,
                                                  int target_width, int target_height,
                                                  int new_width, int new_height,
                                                  int pad_x, int pad_y,
                                                  cv::Mat &output_buffer)
{
    if (origin_image.empty() || target_width <= 0 || target_height <= 0)
        return false;
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        return false;

    try
    {
        cv::cuda::GpuMat gpu_origin;
        gpu_origin.upload(origin_image);

        cv::cuda::GpuMat gpu_output;
        if (new_height == target_height && new_width == target_width)
        {
            // 长宽比一致时直接缩放，无需 padding。
            cv::cuda::resize(gpu_origin, gpu_output,
                             cv::Size(target_width, target_height),
                             0, 0, cv::INTER_LINEAR);
        }
        else
        {
            cv::cuda::GpuMat gpu_resized;
            cv::cuda::resize(gpu_origin, gpu_resized,
                             cv::Size(new_width, new_height),
                             0, 0, cv::INTER_LINEAR);
            cv::cuda::copyMakeBorder(gpu_resized, gpu_output,
                                     pad_y, target_height - new_height - pad_y,
                                     pad_x, target_width - new_width - pad_x,
                                     cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));
        }

        gpu_output.download(output_buffer);
        return !output_buffer.empty();
    }
    catch (const cv::Exception &error)
    {
        std::cerr << "OpenCV CUDA 预处理失败，已回退到 CPU 路径：" << error.what() << std::endl;
        return false;
    }
}

#endif
