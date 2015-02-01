set(the_description "Structure from Motion (SfM)")
set(OPENCV_SFM_REQUIRED_DEPS opencv_core opencv_imgproc opencv_features2d
    opencv_calib3d opencv_highgui opencv_flann opencv_imgcodecs)
ocv_define_module(sfm ${OPENCV_SFM_REQUIRED_DEPS})

ocv_include_directories("${OpenCV_SOURCE_DIR}/include")#for opencv.hpp
ocv_include_modules(${OPENCV_SFM_REQUIRED_DEPS})

if (HAVE_opencv_xfeatures2d)
  ocv_include_directories("${OPENCV_MODULE_opencv_xfeatures2d_LOCATION}/include")
endif()

if (HAVE_opencv_ocl)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/ocl/include")
endif()


ocv_include_directories("${OPENCV_MODULE_opencv_sfm_LOCATION}/include")

add_executable(example-sfm samples/sfm.cpp)
target_link_libraries(example-sfm opencv_sfm)
