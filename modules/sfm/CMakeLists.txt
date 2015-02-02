set(the_description "Structure from Motion (SfM)")
set(OPENCV_SFM_REQUIRED_DEPS opencv_core opencv_imgproc opencv_features2d
    opencv_calib3d opencv_highgui opencv_flann opencv_imgcodecs
    OPTIONAL opencv_xfeatures2d opencv_cvv opencv_ocl)
ocv_define_module(sfm ${OPENCV_SFM_REQUIRED_DEPS})

ocv_include_directories("${OpenCV_SOURCE_DIR}/include")#for opencv.hpp
ocv_include_modules(${OPENCV_SFM_REQUIRED_DEPS})

if (HAVE_opencv_xfeatures2d)
  ocv_include_directories("${OPENCV_MODULE_opencv_xfeatures2d_LOCATION}/include")
endif()

if (HAVE_opencv_cvv)
  ocv_include_directories("${OPENCV_MODULE_opencv_cvv_LOCATION}/include")
  # we need C++11 and want warnings:
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic")
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow -Wmissing-declarations)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DCVVISUAL_DEBUGMODE")
endif()

if (HAVE_opencv_ocl)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/ocl/include")
endif()


ocv_include_directories("${OPENCV_MODULE_opencv_sfm_LOCATION}/include")

add_executable(example-sfm samples/sfm.cpp)
target_link_libraries(example-sfm opencv_sfm)
