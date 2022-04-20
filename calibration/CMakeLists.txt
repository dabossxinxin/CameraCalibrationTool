
link_directories(${PROJECT_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB CaliHdrs "./include/*h")
file(GLOB CaliSrcs "./source/*.cpp")

source_group("Source Files" FILES ${CaliSrcs})
source_group("Header Files" FILES ${CaliHdrs})

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${FLANN_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${OpenCV4_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/calibration/include)

add_library(CaliTool STATIC ${CaliSrcs} ${CaliHdrs})

target_link_libraries(CaliTool debug ${GLOG_DEBUG_LIB} 		optimized ${GLOG_RELEASE_LIB})
target_link_libraries(CaliTool debug ${OpenCV4_DEBUG_LIB} 	optimized ${OpenCV4_RELEASE_LIB})
target_link_libraries(CaliTool debug ${CERES_DEBUG_LIB} 	optimized ${CERES_RELEASE_LIB})
foreach(_var ${GFLAGS_LIBRARIES})
    target_link_libraries(CaliTool debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
	optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
endforeach()

foreach(_var ${FLANN_LIBRARIES})
    	target_link_libraries(CaliTool debug ${FLANN_LIB_DIR}/${_var}.lib
		optimized ${FLANN_LIB_DIR}/${_var}-gd.lib)
endforeach()

FOREACH(_var ${Boost_LIBRARIES})
	 TARGET_LINK_LIBRARIES(CaliTool debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
	optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
ENDFOREACH()

FOREACH(_var ${PCL_LIBRARIES})
	TARGET_LINK_LIBRARIES(CaliTool debug ${PCL_LIB_DIR}/${_var}debug.lib 
		optimized ${PCL_LIB_DIR}/${_var}release.lib)
ENDFOREACH()