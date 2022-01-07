project(CalibrationTool)
cmake_minimum_required(VERSION 2.6)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

include(FindEigen3)
include(FindCeres)
include(FindGLOG)
include(FindOpenCV)
INCLUDE(FindPCL)
INCLUDE(FindBoost)
INCLUDE(FindFlann)

#添加utils模块
add_subdirectory(./utils)
include_directories(./utils)
include_directories(${PROJECT_BINARY_DIR}/utils)

#添加calibration模块
add_subdirectory(./calibration)
include_directories(./calibration/include)
include_directories(${PROJECT_BINARY_DIR}/calibration/include)

#添加实例模块
add_subdirectory(./example)