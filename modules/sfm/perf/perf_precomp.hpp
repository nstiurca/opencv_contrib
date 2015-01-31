#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_SFM_PRECOMP_HPP__
#define __OPENCV_SFM_PRECOMP_HPP__

#include "opencv2/ts.hpp"
//#include "opencv2/imgproc.hpp"
#include "opencv2/sfm.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/videoio.hpp"

#ifdef GTEST_CREATE_SHARED_LIBRARY
#error no modules except ts should have GTEST_CREATE_SHARED_LIBRARY defined
#endif

#endif
