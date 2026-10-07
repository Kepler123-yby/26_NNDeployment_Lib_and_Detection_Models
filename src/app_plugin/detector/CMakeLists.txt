cmake_minimum_required(VERSION 3.22)

set(lib_name NNDetector)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

add_library(${lib_name}_lib SHARED
    src/NNDetector.cpp
)
target_compile_features(${lib_name}_lib PUBLIC cxx_std_20)
target_include_directories(${lib_name}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${lib_name}_lib
    PUBLIC
        NNdeployment_lib
    PRIVATE
        opencv_core
)

add_executable(main examples/main.cpp)
target_compile_definitions(main PRIVATE
    NNDEPLOYMENT_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(main PRIVATE
    ${lib_name}_lib
    opencv_core
    opencv_imgproc
    opencv_videoio
)

add_executable(main_test_speed tests/main_test_speed.cpp)
target_compile_definitions(main_test_speed PRIVATE
    NNDEPLOYMENT_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(main_test_speed PRIVATE
    ${lib_name}_lib
    NNdeployment_mpt_lib
    opencv_core
    opencv_videoio
)

add_executable(detector_speed_bench tests/detector_speed_bench.cpp)
target_compile_definitions(detector_speed_bench PRIVATE
    NNDEPLOYMENT_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(detector_speed_bench PRIVATE
    ${lib_name}_lib
    opencv_core
    opencv_videoio
)

if(BUILD_TESTING)
    add_executable(detector_smoke_test tests/detector_smoke_test.cpp)
    target_compile_definitions(detector_smoke_test PRIVATE
        NNDEPLOYMENT_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(detector_smoke_test PRIVATE
        ${lib_name}_lib
        opencv_core
        opencv_videoio
    )
    add_test(NAME detector_smoke COMMAND detector_smoke_test)
    set_tests_properties(detector_smoke PROPERTIES TIMEOUT 180)
endif()

# OpenCV CUDA 后处理的 CPU/CUDA 一致性测试。
# 测试使用 OpenVINO 模型，仅在库中实际编译了 OpenCV CUDA + OpenVINO 时构建与注册。
if(NNDEPLOYMENT_WITH_OPENCV_CUDA AND NNDEPLOYMENT_WITH_OPENVINO)
    add_executable(cuda_parity_test tests/cuda_parity_test.cpp)
    target_compile_definitions(cuda_parity_test PRIVATE
        NNDEPLOYMENT_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(cuda_parity_test PRIVATE
        ${lib_name}_lib
        opencv_core
        opencv_videoio
    )
    if(BUILD_TESTING)
        add_test(NAME cuda_parity COMMAND cuda_parity_test)
        set_tests_properties(cuda_parity PROPERTIES TIMEOUT 300)
    endif()
endif()
