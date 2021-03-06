/*
 * descriptors.cpp
 *
 *  Created on: Feb 9, 2015
 *      Author: nstiurca
 */
#include <opencv2/sfm.hpp>

namespace cv
{

static const uint8_t _lut[256][8] = {
        {0,0,0,0, 0,0,0,0,},
        {1,0,0,0, 0,0,0,0,},
        {0,1,0,0, 0,0,0,0,},
        {1,1,0,0, 0,0,0,0,},
        {0,0,1,0, 0,0,0,0,},
        {1,0,1,0, 0,0,0,0,},
        {0,1,1,0, 0,0,0,0,},
        {1,1,1,0, 0,0,0,0,},
        {0,0,0,1, 0,0,0,0,},
        {1,0,0,1, 0,0,0,0,},
        {0,1,0,1, 0,0,0,0,},
        {1,1,0,1, 0,0,0,0,},
        {0,0,1,1, 0,0,0,0,},
        {1,0,1,1, 0,0,0,0,},
        {0,1,1,1, 0,0,0,0,},
        {1,1,1,1, 0,0,0,0,},
        {0,0,0,0, 1,0,0,0,},
        {1,0,0,0, 1,0,0,0,},
        {0,1,0,0, 1,0,0,0,},
        {1,1,0,0, 1,0,0,0,},
        {0,0,1,0, 1,0,0,0,},
        {1,0,1,0, 1,0,0,0,},
        {0,1,1,0, 1,0,0,0,},
        {1,1,1,0, 1,0,0,0,},
        {0,0,0,1, 1,0,0,0,},
        {1,0,0,1, 1,0,0,0,},
        {0,1,0,1, 1,0,0,0,},
        {1,1,0,1, 1,0,0,0,},
        {0,0,1,1, 1,0,0,0,},
        {1,0,1,1, 1,0,0,0,},
        {0,1,1,1, 1,0,0,0,},
        {1,1,1,1, 1,0,0,0,},
        {0,0,0,0, 0,1,0,0,},
        {1,0,0,0, 0,1,0,0,},
        {0,1,0,0, 0,1,0,0,},
        {1,1,0,0, 0,1,0,0,},
        {0,0,1,0, 0,1,0,0,},
        {1,0,1,0, 0,1,0,0,},
        {0,1,1,0, 0,1,0,0,},
        {1,1,1,0, 0,1,0,0,},
        {0,0,0,1, 0,1,0,0,},
        {1,0,0,1, 0,1,0,0,},
        {0,1,0,1, 0,1,0,0,},
        {1,1,0,1, 0,1,0,0,},
        {0,0,1,1, 0,1,0,0,},
        {1,0,1,1, 0,1,0,0,},
        {0,1,1,1, 0,1,0,0,},
        {1,1,1,1, 0,1,0,0,},
        {0,0,0,0, 1,1,0,0,},
        {1,0,0,0, 1,1,0,0,},
        {0,1,0,0, 1,1,0,0,},
        {1,1,0,0, 1,1,0,0,},
        {0,0,1,0, 1,1,0,0,},
        {1,0,1,0, 1,1,0,0,},
        {0,1,1,0, 1,1,0,0,},
        {1,1,1,0, 1,1,0,0,},
        {0,0,0,1, 1,1,0,0,},
        {1,0,0,1, 1,1,0,0,},
        {0,1,0,1, 1,1,0,0,},
        {1,1,0,1, 1,1,0,0,},
        {0,0,1,1, 1,1,0,0,},
        {1,0,1,1, 1,1,0,0,},
        {0,1,1,1, 1,1,0,0,},
        {1,1,1,1, 1,1,0,0,},
        {0,0,0,0, 0,0,1,0,},
        {1,0,0,0, 0,0,1,0,},
        {0,1,0,0, 0,0,1,0,},
        {1,1,0,0, 0,0,1,0,},
        {0,0,1,0, 0,0,1,0,},
        {1,0,1,0, 0,0,1,0,},
        {0,1,1,0, 0,0,1,0,},
        {1,1,1,0, 0,0,1,0,},
        {0,0,0,1, 0,0,1,0,},
        {1,0,0,1, 0,0,1,0,},
        {0,1,0,1, 0,0,1,0,},
        {1,1,0,1, 0,0,1,0,},
        {0,0,1,1, 0,0,1,0,},
        {1,0,1,1, 0,0,1,0,},
        {0,1,1,1, 0,0,1,0,},
        {1,1,1,1, 0,0,1,0,},
        {0,0,0,0, 1,0,1,0,},
        {1,0,0,0, 1,0,1,0,},
        {0,1,0,0, 1,0,1,0,},
        {1,1,0,0, 1,0,1,0,},
        {0,0,1,0, 1,0,1,0,},
        {1,0,1,0, 1,0,1,0,},
        {0,1,1,0, 1,0,1,0,},
        {1,1,1,0, 1,0,1,0,},
        {0,0,0,1, 1,0,1,0,},
        {1,0,0,1, 1,0,1,0,},
        {0,1,0,1, 1,0,1,0,},
        {1,1,0,1, 1,0,1,0,},
        {0,0,1,1, 1,0,1,0,},
        {1,0,1,1, 1,0,1,0,},
        {0,1,1,1, 1,0,1,0,},
        {1,1,1,1, 1,0,1,0,},
        {0,0,0,0, 0,1,1,0,},
        {1,0,0,0, 0,1,1,0,},
        {0,1,0,0, 0,1,1,0,},
        {1,1,0,0, 0,1,1,0,},
        {0,0,1,0, 0,1,1,0,},
        {1,0,1,0, 0,1,1,0,},
        {0,1,1,0, 0,1,1,0,},
        {1,1,1,0, 0,1,1,0,},
        {0,0,0,1, 0,1,1,0,},
        {1,0,0,1, 0,1,1,0,},
        {0,1,0,1, 0,1,1,0,},
        {1,1,0,1, 0,1,1,0,},
        {0,0,1,1, 0,1,1,0,},
        {1,0,1,1, 0,1,1,0,},
        {0,1,1,1, 0,1,1,0,},
        {1,1,1,1, 0,1,1,0,},
        {0,0,0,0, 1,1,1,0,},
        {1,0,0,0, 1,1,1,0,},
        {0,1,0,0, 1,1,1,0,},
        {1,1,0,0, 1,1,1,0,},
        {0,0,1,0, 1,1,1,0,},
        {1,0,1,0, 1,1,1,0,},
        {0,1,1,0, 1,1,1,0,},
        {1,1,1,0, 1,1,1,0,},
        {0,0,0,1, 1,1,1,0,},
        {1,0,0,1, 1,1,1,0,},
        {0,1,0,1, 1,1,1,0,},
        {1,1,0,1, 1,1,1,0,},
        {0,0,1,1, 1,1,1,0,},
        {1,0,1,1, 1,1,1,0,},
        {0,1,1,1, 1,1,1,0,},
        {1,1,1,1, 1,1,1,0,},
        {0,0,0,0, 0,0,0,1,},
        {1,0,0,0, 0,0,0,1,},
        {0,1,0,0, 0,0,0,1,},
        {1,1,0,0, 0,0,0,1,},
        {0,0,1,0, 0,0,0,1,},
        {1,0,1,0, 0,0,0,1,},
        {0,1,1,0, 0,0,0,1,},
        {1,1,1,0, 0,0,0,1,},
        {0,0,0,1, 0,0,0,1,},
        {1,0,0,1, 0,0,0,1,},
        {0,1,0,1, 0,0,0,1,},
        {1,1,0,1, 0,0,0,1,},
        {0,0,1,1, 0,0,0,1,},
        {1,0,1,1, 0,0,0,1,},
        {0,1,1,1, 0,0,0,1,},
        {1,1,1,1, 0,0,0,1,},
        {0,0,0,0, 1,0,0,1,},
        {1,0,0,0, 1,0,0,1,},
        {0,1,0,0, 1,0,0,1,},
        {1,1,0,0, 1,0,0,1,},
        {0,0,1,0, 1,0,0,1,},
        {1,0,1,0, 1,0,0,1,},
        {0,1,1,0, 1,0,0,1,},
        {1,1,1,0, 1,0,0,1,},
        {0,0,0,1, 1,0,0,1,},
        {1,0,0,1, 1,0,0,1,},
        {0,1,0,1, 1,0,0,1,},
        {1,1,0,1, 1,0,0,1,},
        {0,0,1,1, 1,0,0,1,},
        {1,0,1,1, 1,0,0,1,},
        {0,1,1,1, 1,0,0,1,},
        {1,1,1,1, 1,0,0,1,},
        {0,0,0,0, 0,1,0,1,},
        {1,0,0,0, 0,1,0,1,},
        {0,1,0,0, 0,1,0,1,},
        {1,1,0,0, 0,1,0,1,},
        {0,0,1,0, 0,1,0,1,},
        {1,0,1,0, 0,1,0,1,},
        {0,1,1,0, 0,1,0,1,},
        {1,1,1,0, 0,1,0,1,},
        {0,0,0,1, 0,1,0,1,},
        {1,0,0,1, 0,1,0,1,},
        {0,1,0,1, 0,1,0,1,},
        {1,1,0,1, 0,1,0,1,},
        {0,0,1,1, 0,1,0,1,},
        {1,0,1,1, 0,1,0,1,},
        {0,1,1,1, 0,1,0,1,},
        {1,1,1,1, 0,1,0,1,},
        {0,0,0,0, 1,1,0,1,},
        {1,0,0,0, 1,1,0,1,},
        {0,1,0,0, 1,1,0,1,},
        {1,1,0,0, 1,1,0,1,},
        {0,0,1,0, 1,1,0,1,},
        {1,0,1,0, 1,1,0,1,},
        {0,1,1,0, 1,1,0,1,},
        {1,1,1,0, 1,1,0,1,},
        {0,0,0,1, 1,1,0,1,},
        {1,0,0,1, 1,1,0,1,},
        {0,1,0,1, 1,1,0,1,},
        {1,1,0,1, 1,1,0,1,},
        {0,0,1,1, 1,1,0,1,},
        {1,0,1,1, 1,1,0,1,},
        {0,1,1,1, 1,1,0,1,},
        {1,1,1,1, 1,1,0,1,},
        {0,0,0,0, 0,0,1,1,},
        {1,0,0,0, 0,0,1,1,},
        {0,1,0,0, 0,0,1,1,},
        {1,1,0,0, 0,0,1,1,},
        {0,0,1,0, 0,0,1,1,},
        {1,0,1,0, 0,0,1,1,},
        {0,1,1,0, 0,0,1,1,},
        {1,1,1,0, 0,0,1,1,},
        {0,0,0,1, 0,0,1,1,},
        {1,0,0,1, 0,0,1,1,},
        {0,1,0,1, 0,0,1,1,},
        {1,1,0,1, 0,0,1,1,},
        {0,0,1,1, 0,0,1,1,},
        {1,0,1,1, 0,0,1,1,},
        {0,1,1,1, 0,0,1,1,},
        {1,1,1,1, 0,0,1,1,},
        {0,0,0,0, 1,0,1,1,},
        {1,0,0,0, 1,0,1,1,},
        {0,1,0,0, 1,0,1,1,},
        {1,1,0,0, 1,0,1,1,},
        {0,0,1,0, 1,0,1,1,},
        {1,0,1,0, 1,0,1,1,},
        {0,1,1,0, 1,0,1,1,},
        {1,1,1,0, 1,0,1,1,},
        {0,0,0,1, 1,0,1,1,},
        {1,0,0,1, 1,0,1,1,},
        {0,1,0,1, 1,0,1,1,},
        {1,1,0,1, 1,0,1,1,},
        {0,0,1,1, 1,0,1,1,},
        {1,0,1,1, 1,0,1,1,},
        {0,1,1,1, 1,0,1,1,},
        {1,1,1,1, 1,0,1,1,},
        {0,0,0,0, 0,1,1,1,},
        {1,0,0,0, 0,1,1,1,},
        {0,1,0,0, 0,1,1,1,},
        {1,1,0,0, 0,1,1,1,},
        {0,0,1,0, 0,1,1,1,},
        {1,0,1,0, 0,1,1,1,},
        {0,1,1,0, 0,1,1,1,},
        {1,1,1,0, 0,1,1,1,},
        {0,0,0,1, 0,1,1,1,},
        {1,0,0,1, 0,1,1,1,},
        {0,1,0,1, 0,1,1,1,},
        {1,1,0,1, 0,1,1,1,},
        {0,0,1,1, 0,1,1,1,},
        {1,0,1,1, 0,1,1,1,},
        {0,1,1,1, 0,1,1,1,},
        {1,1,1,1, 0,1,1,1,},
        {0,0,0,0, 1,1,1,1,},
        {1,0,0,0, 1,1,1,1,},
        {0,1,0,0, 1,1,1,1,},
        {1,1,0,0, 1,1,1,1,},
        {0,0,1,0, 1,1,1,1,},
        {1,0,1,0, 1,1,1,1,},
        {0,1,1,0, 1,1,1,1,},
        {1,1,1,0, 1,1,1,1,},
        {0,0,0,1, 1,1,1,1,},
        {1,0,0,1, 1,1,1,1,},
        {0,1,0,1, 1,1,1,1,},
        {1,1,0,1, 1,1,1,1,},
        {0,0,1,1, 1,1,1,1,},
        {1,0,1,1, 1,1,1,1,},
        {0,1,1,1, 1,1,1,1,},
        {1,1,1,1, 1,1,1,1,},
};

//static const uint64_t * const LUT = reinterpret_cast<const uint64_t *>(&_lut[0][0]);
typedef Vec<uint8_t, 8> Vec8b;
typedef Mat_<Vec8b> Mat8b;
typedef Mat_<uint64_t> Mat1ul;
//static const Mat1ul lut(1,256,const_cast<uint64_t *>(reinterpret_cast<const uint64_t *>(&_lut[0][0])));
static const Vec8b * const lut = reinterpret_cast<const Vec8b *>(&_lut[0][0]);

__attribute__((optimize("unroll-loops")))
static void computeAverageDescriptorImpl_8U(const Mat1b descriptors, Mat1b average)
{
    const int rows = descriptors.rows;
    const int cols = descriptors.cols;
//    Mat1ul descriptors2(rows, cols);
    Mat8b descriptors2(rows, cols);
//    Mat descriptors2;
    Mat1b descriptors3 = descriptors2.reshape(1, rows);

    // convert bit vector into vector of uint8, which can be summed over
    // this is done with a lookup table that directly converts each 8-bit pattern into
    // the correct 64-bit pattern
//    LUT(descriptors, lut, descriptors2);
//    Mat8b descriptors3 = (Mat8b) descriptors2;
    #pragma unroll
    for(int i=0; i<rows; ++i) {
        #pragma unroll
        for(int j=0; j<cols; ++j) {
            descriptors2(i,j) = lut[descriptors(i,j)];
        }
    }

    // averaging is easy
    Mat average3;
    reduce(descriptors3, average3, 0 /* column-wise -> produce single row*/, REDUCE_AVG, CV_8U);

    // sanity checks
    CV_DbgAssert(average3.rows == 1);
    CV_DbgAssert(average3.cols == descriptors.cols*8);
    CV_DbgAssert(average3.type() == CV_8U);
    CV_DbgAssert(average3.channels() == 1);

    Mat average2 = average3.reshape(8, 1);

    CV_Assert(average.rows == 1);
    CV_Assert(average.cols == cols);
    CV_Assert(average.type() == CV_8U);

    // now convert each Vec8b back to a single byte
    #pragma unroll
    for(int i=0; i<cols; ++i) {
        const Vec8b b = average2.at<Vec8b>(i);
        uint8_t c = 0;
        #pragma unroll
        for(int j=0; j<8; ++j) {
            c |= b[j] << j;
        }
        average(i) = c;
    }
}


void computeAverageDescriptor(cv::InputArrayOfArrays _descriptors, cv::OutputArray _average)
{
    const int cols = _descriptors.cols();
    const int type = _descriptors.type();
    _average.create(1, cols, type);

    if(type == CV_8U) {
        if(_descriptors.isMat() && _average.isMat()) {
            computeAverageDescriptorImpl_8U((Mat1b)_descriptors.getMat(), (Mat1b)_average.getMat());
        } else {
            CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
        }
    } else if(type == CV_32F || type == CV_64F) {
        reduce(_descriptors, _average, 0, REDUCE_AVG);
    } else {
        CV_Error(Error::StsNotImplemented, "Unknown/unsupported data type");
    }
}


} // namespace cv
