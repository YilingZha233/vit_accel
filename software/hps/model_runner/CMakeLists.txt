cmake_minimum_required(VERSION 3.10)
project(tinyvit_cpu_runner LANGUAGES CXX)
find_package(ncnn REQUIRED)
add_executable(model_runner main.cpp)
target_compile_features(model_runner PRIVATE cxx_std_11)
target_compile_options(model_runner PRIVATE -Wall -Wextra)
target_link_libraries(model_runner PRIVATE ncnn)

# Keep the benchmark executable unchanged; camera use has no repeated warmups/load.
add_executable(camera_runner camera_runner.cpp)
target_compile_features(camera_runner PRIVATE cxx_std_11)
target_compile_options(camera_runner PRIVATE -Wall -Wextra)
target_link_libraries(camera_runner PRIVATE ncnn)
