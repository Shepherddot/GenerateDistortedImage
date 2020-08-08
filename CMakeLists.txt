cmake_minimum_required(VERSION 3.16)
project(DistortImage)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(
${OpenCV_INCLUDE_DIR}
)

add_executable(DistortImage
main.cpp
)

target_link_libraries(DistortImage
${OpenCV_LIBS}
)