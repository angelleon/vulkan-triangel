cmake_minimum_required(VERSION 3.16)

project("vulkan triangle")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories($ENV{VULKAN_INCLUDE})
link_directories($ENV{VULKAN_LIB})
find_package(glfw3 3.3 REQUIRED)


add_executable(triangle main.cpp)
target_link_libraries(triangle vulkan glfw)