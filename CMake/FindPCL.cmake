
FIND_PATH(PCL_INCLUDE_DIR NAMES pcl-1.9 HINTS ${CMAKE_SOURCE_DIR}/../../SDK/PCL-1.9.1/include)
FIND_PATH(PCL_LIB_DIR NAMES pcl_common_release.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/PCL-1.9.1/lib)

SET(PCL_LIBRARIES
	pcl_common_
	pcl_features_
	pcl_filters_
	pcl_io_ply_
	pcl_io_
	pcl_kdtree_
	pcl_keypoints_
	pcl_ml_
	pcl_octree_
	pcl_outofcore_
	pcl_people_
	pcl_recognition_
	pcl_registration_
	pcl_sample_consensus_
	pcl_search_
	pcl_segmentation_
	pcl_stereo_
	pcl_surface_
	pcl_tracking_
	pcl_visualization_
	)
	
	

 SET(PCL_INCLUDE_DIRS ${PCL_INCLUDE_DIR}/pcl-1.9)

