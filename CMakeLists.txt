cmake_minimum_required(VERSION 2.8)

project(Calculate_map_freespace)

add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} "src/calculate_map_freespace.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
