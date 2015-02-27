/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __OPENCV_SFM_LENLEN_HPP__
#define __OPENCV_SFM_LENLEN_HPP__

#include <set>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

/** @defgroup sfm Structure-from-Motion (SfM) API

Structure-from-Motion (SfM) API
-------------------------------

*/

namespace cv
{
/////////////////////////////////////////////////
// Module initialization
/////////////////////////////////////////////////
CV_EXPORTS bool initModule_sfm(void);

/////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////
typedef std::vector<Mat> vMat;
typedef std::vector<vMat> vvMat;
typedef std::vector<UMat> vUMat;
typedef std::vector<vUMat> vvUMat;
typedef std::vector<DMatch> vDMatch;
typedef std::vector<vDMatch> vvDMatch;
typedef std::vector<Point> vPoint;
typedef std::vector<KeyPoint> vKeyPoint;
typedef std::vector<vKeyPoint> vvKeyPoint;


/////////////////////////////////////////////////
// Match pruning
/////////////////////////////////////////////////
CV_EXPORTS vDMatch pruneUnmatched(const vvDMatch &allMatches, const double maxDistanceRatio);
CV_EXPORTS void pruneAsymmetric(vDMatch &matches_i, vDMatch &matches_j);
CV_EXPORTS void getSymmetricMatches(const Ptr<DescriptorMatcher> &matcher,
		InputArray descriptors1, InputArray descriptors2,
		CV_OUT vDMatch &matches_12, CV_OUT vDMatch &matches_21, const double maxDistanceRatio);
CV_EXPORTS void getSymmetricMatches(const Ptr<DescriptorMatcher> &matcher1, const Ptr<DescriptorMatcher> &matcher2,
        InputArray descriptors1, InputArray descriptors2,
        CV_OUT vDMatch &matches_12, CV_OUT vDMatch &matches_21, const double maxDistanceRatio);

/////////////////////////////////////////////////
// Descriptor averaging
/////////////////////////////////////////////////
CV_EXPORTS void computeAverageDescriptor(cv::InputArrayOfArrays descriptors, cv::OutputArray average);


/////////////////////////////////////////////////
// Track building
/////////////////////////////////////////////////

struct CV_EXPORTS_W ID
{
    short frameID;
    short pointID;

    CV_WRAP bool operator==(const ID &that) const throw() { return frameID == that.frameID && pointID == that.pointID; }
    CV_WRAP bool operator<(const ID &that)  const throw() { return frameID == that.frameID ? pointID < that.pointID : frameID < that.frameID; }
    CV_WRAP bool operator!=(const ID &that) const throw() { return !(*this == that); }
};
static inline
std::ostream& operator << (std::ostream &out, const ID &id)
{
    return out << '<' << id.frameID << ", " << id.pointID << '>';
}
typedef std::set<ID> sID;
typedef std::vector<ID> vID;
typedef std::vector<vID> vvID;

class CV_EXPORTS_W TrackBuilder : public virtual Algorithm
{
public:
    CV_WRAP virtual void detectAndComputeTracks(InputArrayOfArrays images,
            OutputArrayOfArrays allDescriptors, CV_OUT vvKeyPoint &allKeypoints,
            CV_OUT vvID &tracks) = 0;

};

class CV_EXPORTS_W SimpleTrackBuilder : public TrackBuilder
{
public:
    CV_WRAP static Ptr<SimpleTrackBuilder> create(double ratio_threshold = 0.7,
            Ptr<FeatureDetector> detector = ORB::create(),
//            Ptr<DescriptorExtractor> extractor = Ptr<DescriptorExtractor>(),
            Ptr<DescriptorMatcher> matcher = Ptr<DescriptorMatcher>());

    CV_WRAP virtual void setRatioTestThreshold(double ratio_threshold) = 0;
    CV_WRAP virtual double getRatioTestThreshold() const = 0;

    CV_WRAP virtual void setDetector(Ptr<FeatureDetector> detector) = 0;
    CV_WRAP virtual const Ptr<FeatureDetector> getDetector() const = 0;

//    CV_WRAP virtual void setExtractor(Ptr<DescriptorExtractor> extractor) = 0;
//    CV_WRAP virtual const Ptr<DescriptorExtractor> getExtractor() const = 0;

    CV_WRAP virtual void setMatcher(Ptr<DescriptorMatcher> macher) = 0;
    CV_WRAP virtual const Ptr<DescriptorMatcher> getMatcher() const = 0;
};

class CV_EXPORTS_W Tracks // TODO: make it an Algorithm
{

public:
    virtual ~Tracks() {}

    CV_WRAP virtual bool isInSomeTrack(const ID id) const = 0;
    CV_WRAP virtual ID rootID(const ID id) = 0;
    CV_WRAP virtual vID& track(const ID id) = 0;

    CV_WRAP virtual void makeNewTrack(const ID f1, const ID f2) = 0;
    CV_WRAP virtual void addToTrack(const ID newPoint, const ID parent) = 0;
    CV_WRAP virtual bool tryMerge(const ID a, const ID b) = 0;

    CV_WRAP virtual void getTracks(vvID &tracks) const = 0;

    CV_WRAP static Ptr<Tracks> create();
};

} // namespace cv

#endif //__OPENCV_SFM_LENLEN_HPP__
