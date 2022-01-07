
link_directories(${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${FLANN_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${OpenCV4_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_SOURCE_DIR}/calibration/include)

add_executable(ASCCT ./AreaScanCameraCaliTool.cpp)
target_link_libraries(ASCCT CaliTool Utils)
target_link_libraries(ASCCT debug ${GLOG_LIBRARIES} optimized ${GLOG_LIBRARIES})
target_link_libraries(ASCCT debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})
target_link_libraries(ASCCT debug ${CERES_DEBUG_LIB} optimized ${CERES_RELEASE_LIB})
foreach(_var ${GFLAGS_LIBRARIES})
    target_link_libraries(ASCCT debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
	optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
endforeach()

foreach(_var ${FLANN_LIBRARIES})
    	target_link_libraries(ASCCT debug ${FLANN_LIB_DIR}/${_var}.lib
		optimized ${FLANN_LIB_DIR}/${_var}-gd.lib)
endforeach()

FOREACH(_var ${Boost_LIBRARIES})
	 TARGET_LINK_LIBRARIES(ASCCT debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
	optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
ENDFOREACH()

FOREACH(_var ${PCL_LIBRARIES})
	TARGET_LINK_LIBRARIES(ASCCT debug ${PCL_LIB_DIR}/${_var}debug.lib 
		optimized ${PCL_LIB_DIR}/${_var}release.lib)
ENDFOREACH()

add_executable(LSCCT ./LineScanCameraCaliTool.cpp)
target_link_libraries(LSCCT CaliTool Utils)
target_link_libraries(LSCCT debug ${GLOG_LIBRARIES} optimized ${GLOG_LIBRARIES})
target_link_libraries(LSCCT debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})
target_link_libraries(LSCCT debug ${CERES_DEBUG_LIB} optimized ${CERES_RELEASE_LIB})
foreach(_var ${GFLAGS_LIBRARIES})
    target_link_libraries(LSCCT debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
	optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
endforeach()

foreach(_var ${FLANN_LIBRARIES})
    	target_link_libraries(LSCCT debug ${FLANN_LIB_DIR}/${_var}.lib
		optimized ${FLANN_LIB_DIR}/${_var}-gd.lib)
endforeach()

FOREACH(_var ${Boost_LIBRARIES})
	 TARGET_LINK_LIBRARIES(LSCCT debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
	optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
ENDFOREACH()

FOREACH(_var ${PCL_LIBRARIES})
	TARGET_LINK_LIBRARIES(LSCCT debug ${PCL_LIB_DIR}/${_var}debug.lib 
		optimized ${PCL_LIB_DIR}/${_var}release.lib)
ENDFOREACH()

add_executable(Test ./ModuleTest.cpp)
target_link_libraries(Test CaliTool Utils)
target_link_libraries(Test debug ${GLOG_LIBRARIES} optimized ${GLOG_LIBRARIES})
target_link_libraries(Test debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})
target_link_libraries(Test debug ${CERES_DEBUG_LIB} optimized ${CERES_RELEASE_LIB})
foreach(_var ${GFLAGS_LIBRARIES})
    target_link_libraries(Test debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
	optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
endforeach()

foreach(_var ${FLANN_LIBRARIES})
    	target_link_libraries(Test debug ${FLANN_LIB_DIR}/${_var}.lib
		optimized ${FLANN_LIB_DIR}/${_var}-gd.lib)
endforeach()

FOREACH(_var ${Boost_LIBRARIES})
	 TARGET_LINK_LIBRARIES(Test debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
	optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
ENDFOREACH()

FOREACH(_var ${PCL_LIBRARIES})
	TARGET_LINK_LIBRARIES(Test debug ${PCL_LIB_DIR}/${_var}debug.lib 
		optimized ${PCL_LIB_DIR}/${_var}release.lib)
ENDFOREACH()
