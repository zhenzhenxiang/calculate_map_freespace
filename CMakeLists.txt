cmake_minimum_required(VERSION 2.8)

project(Calculate_map_freespace)

add_compile_options(-std=c++11)
find_package(OpenCV 3.4 REQUIRED)

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} "src/calculate_map_freespace.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

add_executable(${PROJECT_NAME}_map_beautify "src/map_beautify.cpp")
target_link_libraries(${PROJECT_NAME}_map_beautify ${OpenCV_LIBRARIES})
