# NNdeployment 使用示例

调用方只需包含 `network_deployment_interface.hpp`。装甲板和大符使用不同类型，构建后不会暴露另一种结果接口。

## 构造参数

- `model_path`：OpenVINO 使用 `.xml`，TensorRT 使用 `.trt`/`.engine`。
- `infer_mode`：`"sync"` / `"async"` / `"async4"`。
- `deploy_way`：`"openvino"` / `"tensorrt"`。
- `device`：OpenVINO 设备，如 `"GPU"` / `"CPU"` / `"NPU"`。
- `confidence_threshold`：置信度阈值。
- `postprocess_mode`：通常传 `"auto_detect"`。

## 直接传参构造

两个模型都支持只传模型路径：

```cpp
#include "network_deployment_interface.hpp"

ArmorModel armor_model("armor_model.xml");
RuneModel rune_model("rune_model.xml");
```

也可以直接指定部署参数；未提供的尾部参数使用接口默认值：

```cpp
ArmorModel armor_model(
    "armor_model.xml",
    "async",
    "openvino",
    "GPU",
    0.5f,
    "v8infantry_21");

RuneModel rune_model(
    "rune_model.xml",
    "sync",
    "openvino");
```

## 配置文件构造

YAML 构造使用公开的 `YamlConfig`，三个字段依次为 YAML 文件路径、配置节点名和模型仓库根目录。节点里的 `xml` 字段带后端子目录（如 `openvino/xxx.xml`、`onnx/xxx.onnx`、`tensorrt/xxx.engine`），内部将其与 `model_folder` 拼接为完整路径。为兼容旧写法（`model_folder` 传某个后端子目录、`xml` 只写文件名），当直接拼接不存在时会依次回退到“模型文件夹的父目录”以及“在其父目录下按文件名递归查找”：

```cpp
ArmorModel armor_model(YamlConfig{
    "config/detect_openvino.yaml",
    "armor_detect_aim",
    "config/openvino"});

RuneModel rune_model(YamlConfig{
    "config/detect_openvino.yaml",
    "rune_detect",
    "config/openvino"});
```

YAML 构造和直接传参构造是两个独立入口，不再根据字符串后缀自动判断参数用途。

## 获取结果

```cpp
cv::Mat frame;

std::vector<NetArmorResult> armors = armor_model.netProcess(frame, 2);
for (const NetArmorResult &armor : armors)
{
    const std::vector<cv::Point2d> &points = armor.points;
    int armor_id = armor.armor_id;
    int color_id = armor.color_id;
    double score = armor.score;
}

std::vector<NetRuneResult> runes = rune_model.netProcess(frame);
for (const NetRuneResult &rune : runes)
{
    const std::vector<cv::Point2d> &points = rune.points;
    int class_id = rune.class_id;
    double score = rune.score;
}
```

模型任务或输出形状无效时，`netProcess()` 返回空 vector，不要求调用方处理任务类型异常。如果出现空vector，可能是内部出错，也有可能是模型未检测到目标，此时建议打开debug模式输出debug信息。

## 独立性能测试

OpenVINO 官方 `benchmark_app` 不属于正常推理接口。需要时单独包含内部性能头并调用：

```cpp
#include "network_performance.hpp"

MPT::runOfficialBenchmark("armor_model.xml", "GPU", "async");
```

使用该接口的目标需单独链接 `NNdeployment_mpt_lib`。

## OpenCV CUDA 加速（可选）

CMake 会自动检测可用的后端并全部编译进同一个 `NNdeployment_lib`，运行时通过 YAML 的 `deploy_way` 选择 OpenVINO / TensorRT。将 `OpenCV_DIR` 指向带 CUDA 的 OpenCV 即会自动编译 CUDA 版本的预处理/后处理：

- 预处理：使用 `cv::cuda::resize` / `cv::cuda::copyMakeBorder` 在 GPU 上完成缩放与 padding；
- 后处理：使用自定义 CUDA kernel 并行解码候选框，NMS 与结果组装仍在 CPU 完成，保持与 CPU 路径一致的语义。

是否启用由 `ModelConfig` 的 `preprocess_cuda` / `postprocess_cuda` 控制：YAML 构造读取节点中的 `preprocess_cuda` / `postprocess_cuda`（也可用 `cuda` 一键开关），默认 CPU；直接传参构造在末尾追加这两个布尔值：

```cpp
ArmorModel armor_model(
    "armor_model.xml", "async", "openvino", "GPU", 0.5f, "v8infantry_21",
    DebugConfig(), /*preprocess_cuda=*/true, /*postprocess_cuda=*/true);
```

在未以该选项构建或当前没有可用 CUDA 设备时，库会打印提示并自动回退到 CPU。

YAML 构造还支持在运行时覆盖：`YamlConfig` 的 `preprocess_cuda` / `postprocess_cuda` 为 `std::optional<bool>`，设置后优先于配置文件；`opencvCudaAvailable()` 返回当前构建/设备是否支持 CUDA。

```cpp
YamlConfig config{"config/detect_openvino.yaml", "armor_v8", "所有模型"};
config.preprocess_cuda = true;   // 覆盖 YAML 中的取值
config.postprocess_cuda = false;
ArmorModel armor_model(config, DebugConfig());
```

> 注意：当前实现仍需要 host↔device 往返，端到端帧率不会提升（实测 TensorRT 后端略降）。该开关默认关闭，仅作为可选能力保留；真正的提速需要零拷贝改造。
