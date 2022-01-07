
FIND_PATH(OpenCV4_INCLUDE_DIR NAMES opencv2 HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv-4.4.0/include)
FIND_FILE(OpenCV4_DEBUG_LIB opencv_world440d.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv-4.4.0/lib)
FIND_FILE(OpenCV4_RELEASE_LIB opencv_world440.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv-4.4.0/lib)
