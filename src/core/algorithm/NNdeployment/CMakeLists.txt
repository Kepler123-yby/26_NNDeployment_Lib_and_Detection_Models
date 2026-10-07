cmake_minimum_required(VERSION 3.22.0)
set(lib_name NNdeployment)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# 依赖路径配置（参考 qianli_vision 的直接查找方式）
#
# 每个依赖都“检测到即编译”：同一次构建会把可用的后端全部编进同一个库，
# 运行时由 YAML 配置的 deploy_way / preprocess_cuda / postprocess_cuda 选择，
# 因此切换后端或 CUDA 只需改配置文件，不需要重新编译，也不用为不同机器维护
# 多份构建脚本。路径可通过 -D<VAR>=<path> 覆盖，找不到的依赖会自动跳过。
#
#   -DOpenCV_DIR=<带 CUDA 的 OpenCV 构建目录>    # 启用 OpenCV CUDA 预处理/后处理
#   -DOPENVINO_DIR=<OpenVINO runtime/cmake>      # 也可用环境变量
#   -DTENSORRT_ROOT=<TensorRT 安装根目录>
# ============================================================================
set(OPENVINO_DIR "${OPENVINO_DIR}" CACHE PATH "OpenVINO runtime 的 cmake 目录（留空则用系统/环境变量）")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 安装根目录（含 include/ 与 lib/）")
option(NNDEPLOYMENT_ENABLE_OPENVINO "检测到 OpenVINO 时编译该后端" ON)
option(NNDEPLOYMENT_ENABLE_TENSORRT "检测到 TensorRT 时编译该后端" ON)
option(NNDEPLOYMENT_ENABLE_OPENCV_CUDA "检测到 OpenCV CUDA 模块时启用预处理/后处理加速" ON)

# OpenCV 为必需依赖；若 OpenCV_DIR 指向带 CUDA 模块的安装，会自动启用加速。
find_package(OpenCV REQUIRED)
# 模型配置文件使用 YAML（与 qianli_vision 一致）。
find_package(yaml-cpp REQUIRED)

# CUDA Toolkit（TensorRT 与 OpenCV CUDA 都可能需要）。
find_package(CUDAToolkit QUIET)

# ---------------------------------------------------------------------------
# OpenVINO 后端
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_OPENVINO OFF)
if(NNDEPLOYMENT_ENABLE_OPENVINO)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(OpenVINO_FOUND)
        set(NNDEPLOYMENT_WITH_OPENVINO ON)
        message(STATUS "NNdeployment: 启用 OpenVINO 后端 (${OpenVINO_DIR})")
    else()
        message(STATUS "NNdeployment: 未找到 OpenVINO，跳过 OpenVINO 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# TensorRT 后端
# 优先使用 TensorRT 官方/第三方 CMake package；找不到时按 TENSORRT_ROOT 与
# 常见安装路径用 find_path/find_library 直接查找。
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_TENSORRT OFF)
if(NNDEPLOYMENT_ENABLE_TENSORRT AND CUDAToolkit_FOUND)
    find_package(TensorRT QUIET)
    if(NOT TensorRT_FOUND)
        find_path(TENSORRT_INCLUDE_DIR
            NAMES NvInfer.h
            PATHS ${TENSORRT_ROOT}/include
                  /usr/local/tensorrt/include
                  /usr/src/tensorrt/include
                  /usr/local/include
                  /usr/include/x86_64-linux-gnu
        )
        find_library(TENSORRT_NVINFER_LIB
            NAMES nvinfer
            PATHS ${TENSORRT_ROOT}/lib
                  ${TENSORRT_ROOT}/lib64
                  /usr/local/tensorrt/lib
                  /usr/src/tensorrt/lib
                  /usr/local/lib
                  /usr/lib/x86_64-linux-gnu
        )
        get_filename_component(_tensorrt_lib_dir "${TENSORRT_NVINFER_LIB}" DIRECTORY)
        find_library(TENSORRT_NVONNXPARSER_LIB
            NAMES nvonnxparser
            PATHS "${_tensorrt_lib_dir}"
                  ${TENSORRT_ROOT}/lib
                  ${TENSORRT_ROOT}/lib64
                  /usr/local/tensorrt/lib
                  /usr/src/tensorrt/lib
                  /usr/local/lib
                  /usr/lib/x86_64-linux-gnu
        )
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB)
            set(TensorRT_FOUND TRUE)
            set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB})
        endif()
    endif()
    if(TensorRT_FOUND)
        set(NNDEPLOYMENT_WITH_TENSORRT ON)
        message(STATUS "NNdeployment: 启用 TensorRT 后端 (${TENSORRT_INCLUDE_DIR})")
    else()
        message(STATUS "NNdeployment: 未找到 TensorRT，跳过 TensorRT 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenCV CUDA 加速（预处理/后处理）
# 通过检查 OpenCV 是否提供 CUDA 模块目标来判断，无需单独传编译开关。
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_OPENCV_CUDA OFF)
if(NNDEPLOYMENT_ENABLE_OPENCV_CUDA AND CUDAToolkit_FOUND AND
   TARGET opencv_cudawarping AND TARGET opencv_cudaarithm)
    enable_language(CUDA)
    set(NNDEPLOYMENT_WITH_OPENCV_CUDA ON)
    message(STATUS "NNdeployment: 启用 OpenCV CUDA 预处理/后处理")
else()
    message(STATUS "NNdeployment: 未启用 OpenCV CUDA（未检测到带 CUDA 的 OpenCV）")
endif()

if(NOT NNDEPLOYMENT_WITH_OPENVINO AND NOT NNDEPLOYMENT_WITH_TENSORRT)
    message(FATAL_ERROR "未找到任何推理后端（OpenVINO 或 TensorRT），无法构建 NNdeployment_lib")
endif()

# 供 app_plugin 等子目录判断库中实际编译了哪些能力。
set(NNDEPLOYMENT_WITH_OPENVINO "${NNDEPLOYMENT_WITH_OPENVINO}" CACHE INTERNAL "NNdeployment: OpenVINO 后端已编译")
set(NNDEPLOYMENT_WITH_TENSORRT "${NNDEPLOYMENT_WITH_TENSORRT}" CACHE INTERNAL "NNdeployment: TensorRT 后端已编译")
set(NNDEPLOYMENT_WITH_OPENCV_CUDA "${NNDEPLOYMENT_WITH_OPENCV_CUDA}" CACHE INTERNAL "NNdeployment: OpenCV CUDA 已编译")

# ---------------------------------------------------------------------------
# 源文件：公共部分 + 检测到的后端
# ---------------------------------------------------------------------------
set(NNdeployment_src
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/network_deployment_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_deployment_lib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess.cpp
)

if(NNDEPLOYMENT_WITH_OPENVINO)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_openvino.cpp
    )
endif()

if(NNDEPLOYMENT_WITH_TENSORRT)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_tensorrt.cpp
    )
endif()

if(NNDEPLOYMENT_WITH_OPENCV_CUDA)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess_cuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess_cuda.cu
    )
endif()

add_library(${lib_name}_lib SHARED ${NNdeployment_src})

add_library(${lib_name}_mpt_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/performance/network_performance.cpp
)
set_target_properties(${lib_name}_mpt_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${lib_name}_mpt_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

# 头文件路径
target_include_directories(${lib_name}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(${lib_name}_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

# 链接库
target_link_libraries(${lib_name}_lib PUBLIC
    ${OpenCV_LIBS}
)
target_link_libraries(${lib_name}_lib PRIVATE ${lib_name}_mpt_lib)
target_link_libraries(${lib_name}_lib PRIVATE yaml-cpp)

if(NNDEPLOYMENT_WITH_OPENVINO)
    target_link_libraries(${lib_name}_lib PRIVATE openvino::runtime)
endif()

if(NNDEPLOYMENT_WITH_TENSORRT)
    target_include_directories(${lib_name}_lib PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIRS}
    )
    target_link_libraries(${lib_name}_lib PRIVATE
        ${TENSORRT_LIBRARIES}
        CUDA::cudart
    )
endif()

if(NNDEPLOYMENT_WITH_OPENCV_CUDA)
    target_link_libraries(${lib_name}_lib PRIVATE CUDA::cudart)
endif()

# 把检测结果传给源码：1 表示该能力可用。
target_compile_definitions(${lib_name}_lib PRIVATE
    NNDEPLOYMENT_WITH_OPENVINO=$<BOOL:${NNDEPLOYMENT_WITH_OPENVINO}>
    NNDEPLOYMENT_WITH_TENSORRT=$<BOOL:${NNDEPLOYMENT_WITH_TENSORRT}>
    NNDEPLOYMENT_WITH_OPENCV_CUDA=$<BOOL:${NNDEPLOYMENT_WITH_OPENCV_CUDA}>
)

message(STATUS "NNdeployment 后端汇总: OpenVINO=${NNDEPLOYMENT_WITH_OPENVINO} TensorRT=${NNDEPLOYMENT_WITH_TENSORRT} OpenCV-CUDA=${NNDEPLOYMENT_WITH_OPENCV_CUDA}")
