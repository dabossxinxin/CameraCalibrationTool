
add_compile_options(-bigobj)
link_directories(${PROJECT_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB UtilsHdrs "./*.h")
file(GLOB UtilsSrcs "./*.cpp")

source_group("Source Files" FILES ${UtilsSrcs})
source_group("Header Files" FILES ${UtilsHdrs})

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${FLANN_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${OpenCV4_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/utils)

add_library(Utils STATIC ${UtilsHdrs} ${UtilsSrcs})

target_link_libraries(Utils debug ${GLOG_LIBRARIES} optimized ${GLOG_LIBRARIES})
target_link_libraries(Utils debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})
target_link_libraries(Utils debug ${CERES_DEBUG_LIB} optimized ${CERES_RELEASE_LIB})
foreach(_var ${GFLAGS_LIBRARIES})
    target_link_libraries(Utils debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
	optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
endforeach()

foreach(_var ${FLANN_LIBRARIES})
    	target_link_libraries(Utils debug ${FLANN_LIB_DIR}/${_var}.lib
		optimized ${FLANN_LIB_DIR}/${_var}-gd.lib)
endforeach()

FOREACH(_var ${Boost_LIBRARIES})
	 TARGET_LINK_LIBRARIES(Utils debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
	optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
ENDFOREACH()

FOREACH(_var ${PCL_LIBRARIES})
	TARGET_LINK_LIBRARIES(Utils debug ${PCL_LIB_DIR}/${_var}debug.lib 
		optimized ${PCL_LIB_DIR}/${_var}release.lib)
ENDFOREACH()