//////////////////////////////////////////////////
// INCLUDES
//////////////////////////////////////////////////

#include <algorithm>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#ifdef HAVE_xfeatures2d
#include <opencv2/xfeatures2d.hpp>
#endif
#ifdef HAVE_cvv
#include <opencv2/cvv.hpp>
#endif
#include <opencv2/sfm.hpp>
#include "logging.h"
using namespace std;
using namespace cv;

/////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////
typedef vector<String> vString;
typedef Vec<double, 5> Vec5d;
typedef Vec<double, 8> Vec8d;
typedef Matx<double, 6, 6> Matx66d;
typedef Matx66d Cov6d;
typedef Vec<double, 4> Vec4d;
typedef Vec4d Quatd;
typedef vector<Ptr<DescriptorMatcher>> vDescriptorMatcher;

/////////////////////////////////////////////////
// Custom types
/////////////////////////////////////////////////

struct CameraInfo
{
    Matx33d K;
    Vec8d k;
    int rows;
    int cols;
};

struct Time
{
    uint64_t sec, nsec;
};

struct PoseWithVel
{
    Time stamp;
    Vec3d pos;
    Quatd ori;

    Cov6d pose_cov;

    Vec3d vel_lin;
    Vec3d vel_ang;

    Cov6d vel_cov;
};

struct Options
{
    // hold on to the opened YAML file because otherwise FileNodes below will dangle
    FileStorage fs;

    // Where is the data?
    string data_dir;
    string imgs_glob;
    string reconstruction_dir;
    int begin_idx;
    int end_idx;

    // camera calibration
    CameraInfo ci;

    // Interest point detection and matching
    String detector_name;
    FileNode detector_options;
    String extractor_name;
    FileNode extractor_options;
    bool detector_same_as_extractor;
    String matcher_name;
    FileNode matcher_options;
    double match_ratio;
    bool prune_outliers;
    double min_disparity;
    double min_keypoint_distance;
    int min_track_length;

    static Options create(const String &optionsFname);
};

//namespace std
//{
//extern template<>
//struct hash<::cv::ID>;
//}
//
//class FeatureTrack
//{
//public:
//	struct Node
//	{
//		ID parent;
//		short rank;
//
//		explicit Node(ID id) : parent(id), rank(0) {}
//	};
//
//	void makeSet(const ID id);
//	void merge(const ID x, const ID y);
//	void link(const ID x, const ID y);
//	ID findSet(const ID x);
//
//	void clear() { tracks.clear(); }
//
//	typedef unordered_map<ID, vector<ID>> roots_t;
//	roots_t getRootTracks(bool limitSingleFeaturePerFrame);
//
//private:
//	unordered_map<ID, Node> tracks;
//}; // class FeatureTrack

typedef Ptr<DescriptorMatcher> pDM;
typedef vector<pDM> vpDM;
struct SfMMatcher
{
    CameraInfo ci;

    typedef vector<KeyPoint*> vKPp;
    typedef vector<vKPp> vvKPp;
    vector<vvKPp> grid;
    double min_disparity;
    double min_keypoint_distance;
    int min_track_length;

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Ptr<Feature2D> feature2d;
    pDM matcherTemplate;
    vpDM matchers;

    vvKeyPoint allKeypoints;
    vector<Mat2f> distortedKPcoords;
    vector<Mat2f> undistortedKPcoords;
    vUMat allDescriptors;
    vDescriptorMatcher allMatchers;

    vector<vector<vDMatch>> pairwiseMatches;

    typedef map<ID, set<ID>> match_adjacency_lists_t;
    match_adjacency_lists_t match_adjacency_lists;

//	FeatureTrack tracks;
    Ptr<Tracks> tracks;

    void detectAndComputeKeypoints(const vUMat &images);
    void pruneDuplicateKeypoints(vKeyPoint &keypoints);
    void cvvPlotKeypoints(InputArrayOfArrays _imgs, const Scalar &color = Scalar::all(-1),
            int flags = DrawMatchesFlags::DRAW_RICH_KEYPOINTS) const;
    void trainMatchers();
    void computePairwiseSymmetricMatches(const double match_ratio, bool prune_outliers, const double min_disparity);
    void cvvVisualizePairwiseMatches(InputArrayOfArrays _imgs) const;
//    void buildAdjacencyListsAndFeatureTracks_old();
    void buildAdjacencyListsAndFeatureTracks();
    void getTracks(vvID &tracks);

    static SfMMatcher create(const Options &opts);
};

//////////////////////////////////////////////////
// Prototypes
//////////////////////////////////////////////////
static void usage(int argc, char **argv);
static void printRow(ostream &out, InputArray descriptor, const int rowIdx);

static ostream& operator<<(ostream &out, const PoseWithVel &odo);
static istream& operator>>(istream &in, PoseWithVel &odo);
static ostream& operator<<(ostream &out, const Options &o);

static void printPointLoc(ostream &out, const SfMMatcher &matcher, const ID id);
static void printPointLocAndDesc(ostream &out, const SfMMatcher &matcher, const ID id,
        const set<ID> &ids);

//////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // args?
    if(2 > argc) {
        usage(argc, argv);
        exit(-1);
    }

    // parse the options from the YAML file
    Options opts = Options::create(argv[1]);
    INFO(opts);

    ocl::Device oclDev = ocl::Device::getDefault();
    ocl::setUseOpenCL(ocl::haveOpenCL());
    INFO(ocl::haveOpenCL());
    INFO(ocl::useOpenCL());
    INFO(oclDev.name());

    // get names of images
    vString imgNames;
    glob(opts.data_dir + "/" + opts.imgs_glob, imgNames);
    if(opts.end_idx > 0) {
        imgNames.resize(opts.end_idx);
    }
    if(opts.begin_idx > 0) {
        vString trunc(imgNames.begin() + opts.begin_idx, imgNames.end());
        swap(trunc, imgNames);
    }
    const int N = imgNames.size();

    // load images
    INFO(N);
    vUMat imgsC(N), imgsG(N);
    INFO_STR("Loading images: ");
    for(int i = 0; i < N; ++i) {
        imgsC[i] = imread(imgNames[i], IMREAD_COLOR).getUMat(USAGE_ALLOCATE_DEVICE_MEMORY);
        imgsG[i] = imread(imgNames[i], IMREAD_GRAYSCALE).getUMat(USAGE_ALLOCATE_DEVICE_MEMORY);

        CV_Assert(imgsC[i].rows == opts.ci.rows);
        CV_Assert(imgsC[i].cols == opts.ci.cols);

#ifdef HAVE_cvv
        cvv::showImage(imgsC[i], CVVISUAL_LOCATION, imgNames[i].c_str());
#endif
    }

    // create SfM matcher
    SfMMatcher matcher = SfMMatcher::create(opts);

    // get keypoints
    matcher.detectAndComputeKeypoints(imgsG);
    matcher.cvvPlotKeypoints(imgsC);
    matcher.trainMatchers();

    // do pairwise, symmetric matching over all image pairs
    matcher.computePairwiseSymmetricMatches(opts.match_ratio, opts.prune_outliers, opts.min_disparity);
    matcher.cvvVisualizePairwiseMatches(imgsC);

    // build feature tracks
    matcher.buildAdjacencyListsAndFeatureTracks();
    vvID tracks;
    matcher.getTracks(tracks);
    INFO(tracks.size());
//	FeatureTrack::roots_t rootTracks = matcher.tracks.getRootTracks(true);
//	INFO(rootTracks.size());

// write pairwise matches to output files
    char fname[256] = { 0 };
    const string fnameFmt = opts.reconstruction_dir
            + "/measurement/static_measurement_desc%07d.txt";
//  boost::filesystem::create_directory(opts.reconstruction_dir);
    int i = -1;
    Mat3b srcImg;
    ofstream ofs;
    INFO(matcher.match_adjacency_lists.size());
    for(const auto &it : matcher.match_adjacency_lists) {
        const ID srcID = it.first;
        const auto &dstIDs = it.second;

        CV_DbgAssert(srcID.frameID >= i);
        if(srcID.frameID > i) {
            // got to a new frame, so prepare the next file for output;
            i = srcID.frameID;
            srcImg = (Mat3b)imgsC[i].getMat(ACCESS_READ);
            snprintf(fname, sizeof fname, fnameFmt.c_str(), i);
//            INFO(fname);
            CV_Assert(ofs);     // check that the previous file was OK
            ofs.close();        // close previous file
            ofs.open(fname);    // TODO: check ofs is ready for opening?
            if(!ofs) ERROR(fname);
            CV_Assert(ofs);     // check that new file is OK
        }

        // number of points (+1 for srcID)
        // the 0 is the featureID in HyunSoo's Matching program which is always set to 0 in file
        // SIFT_LOWES_Fisheye_FLANN.cpp:273
        ofs << dstIDs.size() + 1 << " 0 ";

        // output RGB
        const KeyPoint &srcKP = matcher.allKeypoints[srcID.frameID][srcID.pointID];
        Vec3b bgr = srcImg(cvRound(srcKP.pt.y), cvRound(srcKP.pt.x));
        ofs << +bgr[2] << ' ' << +bgr[1] << ' ' << +bgr[0]; // unary + promotes char to int so it is printed numerically

        // output source point coordinates + descriptor
        printPointLocAndDesc(ofs, matcher, srcID, dstIDs);

        // finish the line
        ofs << endl;
    }
    CV_Assert(ofs);
    ofs.close();

    // write tracks (including average descriptors) to output file
    ofs.open(opts.reconstruction_dir + "/stitchedmeasurement_static.txt");
    CV_Assert(ofs);
    i = 0;
    int smallTracks = 0;
    int bigTracks = 0;
    INFO(tracks.size());
    for(const auto &track : tracks) {
//        DEBUG(track.size());
//        DEBUG(track);
        if(track.size() < opts.min_track_length) {
            ++smallTracks;
            WARN(track.size());
            continue;
        } else if((int)track.size() > N) {
            ++bigTracks;
            WARN(track.size());
            continue;
        }
        CV_DbgAssert((int)track.size() >= 2);
        CV_DbgAssert((int)track.size() <= N);

        // output track size and ID
        ofs << track.size() << ' ' << i++;

        // output RGB
        const KeyPoint &srcKP = matcher.allKeypoints[track[0].frameID][track[0].pointID];
        Mat3b img = (Mat3b)imgsC[track[0].frameID].getMat(ACCESS_READ);
        Vec3b bgr = img(cvRound(srcKP.pt.y), cvRound(srcKP.pt.x));
        ofs << ' ' << +bgr[2] << ' ' << +bgr[1] << ' ' << +bgr[0]; // unary + promotes char to int so it is printed numerically

        // output locations of points
        for(const ID id : track) {
            printPointLoc(ofs, matcher, id);
        }

        // end the track
        ofs << endl;
    }
    CV_Assert(ofs);
    ofs.close();
    INFO(smallTracks);
    INFO(bigTracks);

    // output the descriptors for each track
    ofs.open(opts.reconstruction_dir + "/descriptors.txt");
    CV_Assert(ofs);
    i = 0;
    const int descType = (matcher.feature2d ? matcher.feature2d : matcher.extractor)->descriptorType();
    const int descSize = (matcher.feature2d ? matcher.feature2d : matcher.extractor)->descriptorSize();
    Mat averageDescriptor(1, descSize, descType);
    for(const auto &track : tracks) {
        // output the track ID
        ofs << i++ << " 1";     // not sure what the 1 is for, but Hyun Soo's code outputs it

        // compute average descriptor for track
        const int n = track.size();
        Mat trackDescriptors(n, descSize, descType);
        for(int j = 0; j < n; ++j) {
            matcher.allDescriptors[track[j].frameID].row(track[j].pointID).copyTo(trackDescriptors.row(j));
        }
        computeAverageDescriptor(trackDescriptors, averageDescriptor);
//        DEBUG(averageDescriptor);
        printRow(ofs, averageDescriptor, 0);

        ofs << endl;

        if(i % 30000 == 0) {
            printf("Finished writing descriptor % 6i / % 6i (%.1f%%)\n", i, (int)tracks.size(),
                    i * 100.0 / tracks.size());
        }
    }
    CV_Assert(ofs);
    ofs.close();

    // display tracks
    {
        Mat stitched(opts.ci.rows * N, opts.ci.cols, imgsC[0].type());
        vMat imgTracks(N);
        for(int i = 0; i < N; ++i) {
            imgTracks[i] = stitched.rowRange(i * opts.ci.rows, (i + 1) * opts.ci.rows);
            CV_DbgAssert(imgTracks[i].rows == imgsC[i].rows);CV_DbgAssert(imgTracks[i].cols == imgsC[i].cols);CV_DbgAssert(imgTracks[i].type() == imgsC[i].type());
            imgsC[i].copyTo(imgTracks[i]);
        }

        for(auto &track : tracks) {
            if((int)track.size() < N)
                continue;
//			INFO(track.first.frameID);
//			INFO(track.first.pointID);
            sort(track.begin(), track.end());

            vPoint pts(N);
            Matx41d c = Scalar::randu(0, 255);
            Scalar color(c.val[0], c.val[1], c.val[2]);
            for(int j = 0; j < N; ++j) {
                const ID &id = track[j];
                pts[j] = matcher.allKeypoints[id.frameID][id.pointID].pt;
//				INFO(pts[j]);
                pts[j].y += j * opts.ci.rows;
                circle(stitched, pts[j], 4, color);
                if(j > 0) {
                    arrowedLine(stitched, pts[j - 1], pts[j], color);
                }
            }
        }


        // display some statistics about the track lengths
        vector<double> vTrack_sizes, vTrack_mean_size, vTrack_Stddev_size;
        map<int, vector<int> > tracksIdxOfLengthHist;
        map<int, int> trackLengthsHist;

        for(size_t i=0; i<tracks.size(); ++i) {
            const auto &t = tracks[i];
            vTrack_sizes.push_back(t.size());
            tracksIdxOfLengthHist[t.size()].push_back(i);
            ++trackLengthsHist[t.size()];
        }
        INFO(trackLengthsHist);
        cv::meanStdDev(vTrack_sizes, vTrack_mean_size, vTrack_Stddev_size);
        INFO(vTrack_mean_size);
        INFO(vTrack_Stddev_size);

        // longest track(s)
        const int maxTrackSize = tracksIdxOfLengthHist.rbegin()->first;
        const vector<int> &vLongestTracksIdx = tracksIdxOfLengthHist.rbegin()->second;
        INFO(maxTrackSize);
        INFO(vLongestTracksIdx);
        INFO(vLongestTracksIdx.size());

#ifdef HAVE_cvv
        cvv::showImage(stitched, CVVISUAL_LOCATION, "stitched tracks");
#endif
    }

#ifdef HAVE_cvv
    // cvv needs this
    cvv::finalShow();
#endif

    return 0;
}

//////////////////////////////////////////////////
// USAGE
//////////////////////////////////////////////////
void usage(int agrc, char** argv)
{
    cout << "Usage:" << endl
        << '\t' << argv[0] << " <options>.yaml" << endl << endl;
}

//////////////////////////////////////////////////
// output points
//////////////////////////////////////////////////
static void printRow(ostream &out, InputArray descriptor, const int rowIdx)
{
    if(descriptor.type() == CV_8UC1) {
        Mat1b row = (Mat1b)descriptor.getMat(rowIdx);
        for(int i = 0; i < row.cols; ++i) {
            out << ' ' << +row(i); // unary + promotes char to int so it is displayed numerically
        }
    } else if(descriptor.type() == CV_32FC1) {
        Mat1f row = (Mat1f)descriptor.getMat(rowIdx);
        for(int i = 0; i < row.cols; ++i) {
            out << ' ' << row(i);
        }
    } else {
        CV_Error(cv::Error::StsNotImplemented, "this descriptor type is not supported");
    }
}

static void printPointLoc(ostream &out, const SfMMatcher &matcher, const ID id)
{
    const int camID = 0;
    out << ' ' << camID << ' ' << id.frameID
        << ' ' << matcher.undistortedKPcoords[id.frameID](id.pointID)[0]
        << ' ' << matcher.undistortedKPcoords[id.frameID](id.pointID)[1]
        << ' ' << matcher.  distortedKPcoords[id.frameID](id.pointID)[0]
        << ' ' << matcher.  distortedKPcoords[id.frameID](id.pointID)[1];
}

static void printPointLocAndDesc(ostream &out, const SfMMatcher &matcher, const ID id,
        const set<ID> &ids)
{
    // copy distorted coordinates of source point and destination points
    printPointLoc(out, matcher, id);
    printRow(out, matcher.allDescriptors[id.frameID], id.pointID);
    for(const auto id : ids) {
        printPointLoc(out, matcher, id);
        printRow(out, matcher.allDescriptors[id.frameID], id.pointID);
    }
}

//////////////////////////////////////////////////
// OPTIONS
//////////////////////////////////////////////////
Options Options::create(const String &optionsFname)
{
    INFO(optionsFname);

    Options ret;

    FileStorage &fs = ret.fs;
    fs.open(optionsFname, FileStorage::READ);
    if(!fs.isOpened()) {
        cout << "could not open file \"" << optionsFname << "\" for reading.";
        exit(-2);
    }

    ret.data_dir = (string)fs["data_dir"];
    ret.imgs_glob = (string)fs["imgs_glob"];
    ret.begin_idx = (int)fs["begin_idx"];
    ret.end_idx = (int)fs["end_idx"];

    string calib_fname = ret.data_dir + (string)fs["calib_fname"];
    INFO(calib_fname);
    FileStorage calibFS(calib_fname, FileStorage::READ | FileStorage::FORMAT_YAML);
    ret.ci.K <<
                (double)calibFS["Fx"], 0.0, (double)calibFS["Px"],
                0.0, (double)calibFS["Fy"], (double)calibFS["Py"],
                0.0, 0.0, 1.0;
    ret.ci.k <<
                                           (double)calibFS["k1"],
                                           (double)calibFS["k2"],
                                           (double)calibFS["k3"],
                                           (double)calibFS["k4"],
                                           (double)calibFS["k5"],
            calibFS["k6"].isNone() ? 0.0 : (double)calibFS["k6"],
            calibFS["k7"].isNone() ? 0.0 : (double)calibFS["k7"],
            calibFS["k8"].isNone() ? 0.0 : (double)calibFS["k8"];
    ret.ci.rows = (int)calibFS["ImageH"];
    ret.ci.cols = (int)calibFS["ImageW"];

    ret.detector_options = fs["detector"];
    ret.detector_name = (String)ret.detector_options["name"];
    ret.reconstruction_dir = ret.data_dir + "/reconstruction-" + ret.detector_name;

    ret.extractor_options = fs["extractor"];
    if(ret.extractor_options.isMap()) {
        ret.extractor_name = (String)ret.extractor_options["name"];
        ret.detector_same_as_extractor = false;
    } else if(ret.extractor_options.isString()
            && (String)ret.extractor_options == "DETECTOR_SAME_AS_EXTRACTOR") {
        ret.extractor_name = ret.detector_name;
        ret.extractor_options = ret.detector_options;
        ret.detector_same_as_extractor = true;
    } else {
        CV_Error(Error::StsParseError,
                "Expected \"extractor\" to be a map similar to \"detector\", "
                        "or the string \"DETECTOR_SAME_AS_EXTRACTOR\".");
    }

    ret.matcher_name = (String)fs["matcher"]["name"];
    ret.matcher_options = fs["matcher"]["options"];

    ret.match_ratio = (double)fs["match_ratio"];
    ret.prune_outliers = fs["prune_outliers"].isNone()
        ? false : (int)fs["prune_outliers"];

    ret.min_disparity = fs["min_disparity"].isNone()
        ? 0.0 : (double)fs["min_disparity"];

    ret.min_keypoint_distance = fs["min_keypoint_distance"].isNone()
        ? 0.0 : (double)fs["min_keypoint_distance"];

    ret.min_track_length = fs["min_track_length"].isNone()
        ? 0 : (int)fs["min_track_length"];

    return ret;
}

//////////////////////////////////////////////////
// MATCHER
//////////////////////////////////////////////////
SfMMatcher SfMMatcher::create(const Options &opts)
{
    CV_Assert(opts.fs.isOpened());

    SfMMatcher ret;

    ret.ci = opts.ci;
    ret.min_disparity = opts.min_disparity;
    ret.min_keypoint_distance = opts.min_keypoint_distance;
    ret.min_track_length = opts.min_track_length;

    if(ret.min_keypoint_distance > 0.0) {
        const int M = ret.ci.rows / ret.min_keypoint_distance;
        const int N = ret.ci.cols / ret.min_keypoint_distance;
        ret.grid.resize(M + 2, vvKPp(N + 2));
    }

    if(opts.detector_same_as_extractor) {
        INFO(opts.detector_name);
#ifdef HAVE_xfeatures2d
        ret.feature2d = xfeatures2d::Feature2D_create(opts.detector_name);
#else
        ret.feature2d = Feature2D::create(opts.detector_name);
#endif
        ret.feature2d->read(opts.detector_options);
        if(opts.min_keypoint_distance > 0.0) {
            ret.detector = ret.extractor = ret.feature2d;
            ret.feature2d.release();
        }
    } else {
        INFO(opts.detector_name);
        INFO(opts.extractor_name);
#ifdef HAVE_xfeatures2d
        ret.detector = xfeatures2d::Feature2D_create(opts.detector_name);
        ret.extractor = xfeatures2d::Feature2D_create(opts.detector_name);
#else
        ret.detector = FeatureDetector::create(opts.detector_name);
        ret.extractor = DescriptorExtractor::create(opts.extractor_name);
#endif
        ret.detector->read(opts.detector_options);
        ret.extractor->read(opts.extractor_options);
    }

    INFO(opts.matcher_name);
    ret.matcherTemplate = DescriptorMatcher::create(opts.matcher_name);
    ret.matcherTemplate->read(opts.matcher_options);

    DEBUG(ret.detector.get());
    DEBUG(ret.extractor.get());
    DEBUG(ret.feature2d.get());
    DEBUG(ret.matcherTemplate.get());

    CV_Assert((ret.detector.empty() && ret.extractor.empty()) ^ ret.feature2d.empty());
    {
        ORB *orb = dynamic_cast<ORB*>(ret.feature2d.get());
        if(orb) {
//			orb->setMaxFeatures(5000);
            DEBUG(orb->getMaxFeatures());
            DEBUG(orb->getScaleFactor());
            DEBUG(orb->getNLevels());
            DEBUG(orb->getEdgeThreshold());
            DEBUG(orb->getFirstLevel());
            DEBUG(orb->getWTA_K());
            DEBUG(orb->getScoreType());
            DEBUG(orb->getPatchSize());
            DEBUG(orb->getFastThreshold());
        }
    }

    CV_Assert(!ret.matcherTemplate.empty());

    return ret;
}

void SfMMatcher::pruneDuplicateKeypoints(vKeyPoint &keypoints)
{
    CV_Assert(min_keypoint_distance > 0.0);
    const int M = ci.rows / max(min_keypoint_distance, 2.0);
    const int N = ci.cols / max(min_keypoint_distance, 2.0);
    CV_DbgAssert((int)grid.size() == M+2);
    for(auto &row : grid) {
        CV_DbgAssert((int)row.size() == N+2);
        for(auto &cell : row) {
            cell.clear();
        }
    }

    enum { SUPPRESSED = -10 };

    for(KeyPoint &kp : keypoints) {
        // compute cell location
        const int gx = kp.pt.x / min_keypoint_distance + 1;
        const int gy = kp.pt.y / min_keypoint_distance + 1;
        CV_DbgAssert(gx > 0);
        CV_DbgAssert(gx <= N);
        CV_DbgAssert(gy > 0);
        CV_DbgAssert(gy <= M);

        // compute bounding box around kp
        Rect2f r(kp.pt.x - min_keypoint_distance, kp.pt.y - min_keypoint_distance,
                min_keypoint_distance * 2.0, min_keypoint_distance * 2.0);

        // check surrounding cells for suppressant, or suppress all
        for(int y = gy - 1; y <= gy + 1; ++y) {
            CV_DbgAssert(y >= 0);CV_DbgAssert(y <= M+1);
            for(int x = gx - 1; x <= gx + 1; ++x) {
                CV_DbgAssert(x >= 0);CV_DbgAssert(x <= N+1);
                for(KeyPoint *p : grid[y][x]) {
                    if(p->pt.inside(r)) {
                        // kp overlaps with *p, so suppress one
                        if(kp.response > p->response) {
                            p->class_id = SUPPRESSED;
                        } else {
                            kp.class_id = SUPPRESSED;
                        }
                    }
                }
            }
        }

        // add to cell
        grid[gy][gx].push_back(&kp);
    }

    // remove suppressed keypoints
    auto newEnd = remove_if(keypoints.begin(), keypoints.end(),
            [](const KeyPoint &kp) {return kp.class_id == SUPPRESSED;});
    DEBUG(keypoints.size());
    keypoints.resize(newEnd - keypoints.begin());
    DEBUG(keypoints.size());
}

void SfMMatcher::detectAndComputeKeypoints(const vUMat &images)
{
    const int N = images.size();
    allKeypoints.resize(N);
    allDescriptors.resize(N);
    distortedKPcoords.resize(N);
    undistortedKPcoords.resize(N);

    InputArray emptymask = noArray();

    const bool combined = !!feature2d;

    for(int i = 0; i < N; ++i) {
        if(combined) {
            CV_DbgAssert(!min_keypoint_distance>0.0);
            feature2d->detectAndCompute(images[i], emptymask, allKeypoints[i], allDescriptors[i]);
        } else {
            detector->detect(images[i], allKeypoints[i], emptymask);
            if(min_keypoint_distance > 0.0)
                pruneDuplicateKeypoints(allKeypoints[i]);
            extractor->compute(images[i], allKeypoints[i], allDescriptors[i]);
        }
        const int n = allKeypoints[i].size();
        if(0 == n) {
            WARN(i);
            WARN_STR("no keypoints found");
            continue;
        }
        distortedKPcoords[i] = Mat2f(n, 1, reinterpret_cast<Vec2f*>(&allKeypoints[i][0].pt),
                sizeof(KeyPoint));
        undistortedKPcoords[i].create(n, 1);
        undistortPoints(distortedKPcoords[i].clone(), undistortedKPcoords[i], ci.K, ci.k, noArray(),
                ci.K);
        printf("Image % 4i / %i (%.1f%%). Found %i descriptors\n", i + 1, N, 100.0 * (i + 1) / N,
                n);
    }
}

void SfMMatcher::trainMatchers()
{
    const int N = allDescriptors.size();
    matchers.resize(N);
    for(int i = 0; i < N; ++i) {
        if(i%10 == 0) {
            printf("Train % 4i / %i (%.1f%%).\n", i + 1, N, 100.0 * (i + 1) / N);
        }
        matchers[i] = matcherTemplate->clone(true);
        if(allKeypoints[i].size()) {
	    matchers[i]->add(allDescriptors[i]);
            matchers[i]->train();
        }
    }
}

void SfMMatcher::computePairwiseSymmetricMatches(const double match_ratio, bool prune_outliers,
        const double min_disparity)
{
    // do pairwise, symmetric matching over all image pairs
    const int N = allDescriptors.size();
    INFO(N);
    pairwiseMatches.resize(N, vector<vDMatch>(N));

    struct ParLoopBody : public ParallelLoopBody
    {
        SfMMatcher &m;
        const double match_ratio;
        const double min_disparity2;
        const bool prune_outliers;
        vector<pair<int,int> > vij;

        ParLoopBody(SfMMatcher &m, const double match_ratio, const bool prune_outliers,
            const double min_disparity, const int N)
            : m(m), match_ratio(match_ratio), min_disparity2(min_disparity*min_disparity), prune_outliers(prune_outliers)
        {
            vij.reserve(N*(N-1)/2);
            for(int i=0; i<N; ++i) {
                for(int j=i+1; j<N; ++j) {
                    vij.emplace_back(i,j);
                }
            }
            DEBUG(vij.size());
            DEBUG(N*(N-1)/2);
            CV_DbgAssert((int)vij.size() == N*(N-1)/2);
        }

        void operator() (const Range &range) const
        {
            Mat2f ptsi, ptsj;
            //vector<uint8_t> inliers_mask;
            //cv::Mat_<uint8_t> inliers_mask;
            cv::Mat inliers_mask;
            for(int idx=range.start; idx<range.end; ++idx) {
                int i = vij[idx].first;
                int j = vij[idx].second;
                vDMatch &mij = m.pairwiseMatches[i][j];
                vDMatch &mji = m.pairwiseMatches[j][i];
                getSymmetricMatches(m.matchers[i], m.matchers[j], m.allDescriptors[i], m.allDescriptors[j],
                        mij, mji, match_ratio);

                if(min_disparity2 > 0.0) {
                    int readIdx = 0;
                    int writeIdx = 0;
                    while(readIdx < mij.size()) {
                        const auto &pi = m.undistortedKPcoords[i](mij[readIdx].queryIdx);
                        const auto &pj = m.undistortedKPcoords[j](mij[readIdx].trainIdx);
                        const double disparity2 = pi.dot(pj);
                        if( disparity2 >= min_disparity2 ) {
                            if(writeIdx != readIdx) {
                                mij[writeIdx] = mij[readIdx];
                                mji[writeIdx] = mji[readIdx];
                            }
                            ++writeIdx;
                        }
                        ++readIdx;
                    }
                    mij.resize(writeIdx);
                    mji.resize(writeIdx);
                }
                const int nMatchesMax = mij.size();
                if(nMatchesMax < 20) {
                    mij.clear();
                    mji.clear();
                    continue;
                }
                if(prune_outliers) {
                    ptsi.create(nMatchesMax, 1);
                    ptsj.create(nMatchesMax, 1);
                    for(int k=0; k<nMatchesMax; ++k) {
                        ptsi(k) = m.undistortedKPcoords[i](mij[k].queryIdx);
                        ptsj(k) = m.undistortedKPcoords[j](mij[k].trainIdx);
                    }
                    cv::findFundamentalMat(ptsi, ptsj, inliers_mask);
                    int readIdx = 0;
                    int writeIdx = 0;
                    while(readIdx < nMatchesMax) {
                        if(inliers_mask.at<uint8_t>(readIdx)) {
                            if(writeIdx != readIdx) {
                                mij[writeIdx] = mij[readIdx];
                                mji[writeIdx] = mji[readIdx];
                            }
                            ++writeIdx;
                        }
                        ++readIdx;
                    }
                    mij.resize(writeIdx);
                    mji.resize(writeIdx);
                }
            }
            vector<char> progress_dots(range.end-range.start+2, '.');
            progress_dots[progress_dots.size()-2] = '\n';
            progress_dots[progress_dots.size()-1] = 0;      // null terminated
            cout << progress_dots.data(); cout.flush();
        }
    };

    ParLoopBody parLoopBody(*this, match_ratio, prune_outliers, min_disparity, N);
    parallel_for_(Range(0, parLoopBody.vij.size()), parLoopBody, N);
}

#ifdef HAVE_cvv
template<typename MAT>
static void cvvVisualizePairwiseMatchesImpl(const vector<MAT> &imgs, const vvKeyPoint &allKeypoints,
        const vector<vector<vDMatch>> &pairwiseMatches)
{
    const int N = imgs.size();
    char description[100];
    for(int i = 0; i < N; ++i) {
        const int jMin = max(i - 5, 0);
        const int jMax = min(i + 5, i);
        for(int j = jMin; j < jMax; ++j) {
            snprintf(description, sizeof(description), "symmetric matches %i  %i", i, j);
            cvv::debugDMatch(imgs[i], allKeypoints[i], imgs[j], allKeypoints[j],
                    pairwiseMatches[i][j],
                    CVVISUAL_LOCATION, description);
        }
    }
}
#endif // HAVE_cvv

void SfMMatcher::cvvVisualizePairwiseMatches(InputArrayOfArrays _imgs) const
{
#ifdef HAVE_cvv
    if(_imgs.isMatVector()) {
        vMat imgs;
        _imgs.getMatVector(imgs);
        cvvVisualizePairwiseMatchesImpl(imgs, allKeypoints, pairwiseMatches);
    } else if(_imgs.isUMatVector()) {
        vUMat imgs;
        _imgs.getUMatVector(imgs);
        cvvVisualizePairwiseMatchesImpl(imgs, allKeypoints, pairwiseMatches);
    } else {
        CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    }
#else   // !defined(HAVE_cvv)
    WARN_STR("STUB: CVV is not available");
#endif  // ifdef HAVE_cvv
}

//void SfMMatcher::buildAdjacencyListsAndFeatureTracks_old()
//{
//    // build feature tracks
//    const int N = pairwiseMatches.size();
//    tracks.clear();
//    match_adjacency_lists.clear();
//    for (int i = 0; i < N; ++i) {
//        for (int j = i+1; j < N; ++j) {
//            for (const DMatch &m : pairwiseMatches[i][j]) {
//                const ID f1 = { i, m.queryIdx };
//                const ID f2 = { j, m.trainIdx };
//                tracks.makeSet(f1);
//                tracks.makeSet(f2);
//                tracks.merge(f1, f2);
//                match_adjacency_lists[f1].insert(f2);
//            }
//        }
//    }
//    INFO(match_adjacency_lists.size());
//}

void SfMMatcher::buildAdjacencyListsAndFeatureTracks()
{
    const int N = pairwiseMatches.size();

    tracks = Tracks::create();
    Tracks &t = *tracks;
    match_adjacency_lists.clear();

    int canMerge = 0;
    int cantMerge = 0;
    for(int i = 0; i < N; ++i) {
        for(int j = i + 1; j < N; ++j) {
            for(const DMatch &m : pairwiseMatches[i][j]) {
                const ID f1 = { i, m.queryIdx };
                const ID f2 = { j, m.trainIdx };
                match_adjacency_lists[f1].insert(f2);
                //match_adjacency_lists[f2].insert(f1);
                if(t.isInSomeTrack(f2)) {
                    if(t.isInSomeTrack(f1)) {
                        if(t.rootID(f1) == t.rootID(f2)) {
                            // same track already; do nothing
                        } else {
                            // f1 and f2 already in different tracks
                            // TODO: merge tracks?!?!?
                            if(t.tryMerge(t.rootID(f1), t.rootID(f2))) {
                                ++canMerge;
                            } else {
                                ++cantMerge;
                                DEBUG_STR("STUB");
                                DEBUG(i);
                                DEBUG(j);
                                DEBUG(f1);
                                DEBUG(f2);
                                DEBUG(t.track(f1));
                                DEBUG(t.track(f2));
                            }
                        }
                    } else {
                        // add f1 to f2's track
                        t.addToTrack(f1, f2);
                    }
                } else {
                    // f2 has no parent/root, so add to a (possibly new) track
                    if(t.isInSomeTrack(f1)) {
                        // add f2 to f1's track
                        t.addToTrack(f2, f1);
                    } else {
                        // new track of f1 and f2
                        t.makeNewTrack(f1, f2);
                    }
                } // if(isInSomeTrack(f2) {} else {}
            } // for(const DMatch &m : pairwiseMatches[i][j])
        } // for(int j=i+1; j<N; ++j)
    } // for(int i=0; i<N; ++i)
    INFO(match_adjacency_lists.size());
    INFO(canMerge);
    INFO(cantMerge);
} // void SfMMatcher::buildAdjacencyListsAndFeatureTracks()

static inline bool frameLess(const int a, const ID b)
{
    return a < b.frameID;
}

void SfMMatcher::getTracks(vvID &tracks_)
{
    tracks->getTracks(tracks_);

    // first, prune elements from tracks that have multiple keypoints
    // from the same image
    for(auto &track : tracks_) {
        CV_DbgAssert(!track.empty());
//        INFO(track.size());
        auto r = track.begin();         // read iterator
        auto w = r;                     // write iterator
        const auto e = track.end();     // end iterator
        CV_DbgAssert(is_sorted(r, e));
        while (r != e) {
            const short i = r->frameID;
            auto r2 = std::find_if_not(r + 1, e, [=](const ID id) {return id.frameID == i;});
            CV_DbgAssert(r2 - r > 0);
            if(r2 - r > 1) {
                // TODO: the elements [r, r2) have the same frameID. pick only 1 of them, discard the rest
                *w++ = *std::max_element(r, r2, [&](const ID largest, const ID first) {
                    CV_DbgAssert(largest.frameID == first.frameID);
                    return allKeypoints[largest.frameID][largest.pointID].response
                         < allKeypoints[largest.frameID][  first.pointID].response; });
                r = r2;
            } else {
                CV_DbgAssert(r2 - r == 1);
                // no other element has the same frameID, so just copy it;
                *w++ = *r++;
            }
        } // while(r != e)
        // done identifying duplicate points, but still need to discard them
        CV_DbgAssert(w <= e);
//        DEBUG((track.size() - (w - track.begin())));
        track.resize(w - track.begin());
    } // for( track : tracks)

    // finally, prune short tracks
    auto new_end = std::remove_if(tracks_.begin(), tracks_.end(),
                [=](const vID &vid) { return (int)vid.size() < min_track_length; });
    tracks_.resize(new_end - tracks_.begin());
} // void SfMMatcher::getTracks(vvID &tracks_)

#ifdef HAVE_cvv
template<typename MAT>
static void cvvPlotKeypointsImpl(const vector<MAT> &imgs, const vvKeyPoint &allKeypoints,
        const Scalar &color, int flags)
{
    if(!cvv::debugMode())
        return;

    CV_Assert(allKeypoints.size() == imgs.size());
    const int N = imgs.size();
    MAT m;
    char buf[80];
    for(int i = 0; i < N; ++i) {
        drawKeypoints(imgs[i], allKeypoints[i], m, color, flags);
        snprintf(buf, sizeof buf, "keypoints for image %i", i);
        cvv::showImage(m, CVVISUAL_LOCATION, buf);
    }
}
#endif // HAVE_cvv

void SfMMatcher::cvvPlotKeypoints(InputArrayOfArrays _imgs, const Scalar &color, int flags) const
{
#ifdef HAVE_cvv
    if(!cvv::debugMode())
        return;

    if(_imgs.isMatVector()) {
        vMat imgs;
        _imgs.getMatVector(imgs);
        cvvPlotKeypointsImpl(imgs, allKeypoints, color, flags);
    } else if(_imgs.isUMatVector()) {
        vUMat imgs;
        _imgs.getUMatVector(imgs);
        cvvPlotKeypointsImpl(imgs, allKeypoints, color, flags);
    } else {
        CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
    }
#else
    WARN_STR("STUB: CVV is not available");
#endif
}

////////////////////////////////////////////////////
//// FEATURE TRACKS
////////////////////////////////////////////////////
//void FeatureTrack::makeSet(const ID id)
//{
//	auto it = tracks.find(id);
//	if(tracks.end() != it) return;
//
//	tracks.emplace(piecewise_construct, make_tuple(id), make_tuple(id));
////	tracks.emplace(make_pair(id, Node{id, 0}));
//}
//
//void FeatureTrack::merge(const ID x, const ID y)
//{
//	link(findSet(x), findSet(y));
//}
//
//void FeatureTrack::link(const ID x, const ID y)
//{
//	auto itX = tracks.find(x);
//	auto itY = tracks.find(y);
//
//	CV_Assert(tracks.end() != itX);
//	CV_Assert(tracks.end() != itY);
//
//	if(itX->second.rank > itY->second.rank) {
//		itY->second.parent = x;
//	} else {
//		itX->second.parent = y;
//		if(itX->second.rank == itY->second.rank) {
//			++itY->second.rank;
//		}
//	}
//}
//
//ID FeatureTrack::findSet(const ID x)
//{
//	auto it = tracks.find(x);
//
//	CV_Assert(tracks.end() != it);
//
//	if(x != it->second.parent) {
//		it->second.parent = findSet(it->second.parent);
//	}
//
//	return it->second.parent;
//}
//
//FeatureTrack::roots_t FeatureTrack::getRootTracks(bool limitSingleFeaturePerFrame)
//{
//	roots_t rootSets;
//
//	for(auto &idNode : tracks) {
//	    auto &rootSet = rootSets[findSet(idNode.first)];
//	    CV_DbgAssert(find(begin(rootSet), end(rootSet), idNode.first) == end(rootSet));
//	    rootSet.push_back(idNode.first);
//	}
//
//	if(limitSingleFeaturePerFrame) {
//	    auto it = rootSets.begin();
//	    while(it != rootSets.end()) {
//	        auto b = begin(it->second);
//	        auto e = end(it->second);
//	        sort(b, e);
//	        auto duplicateFrameIt = adjacent_find(b, e,
//	                [](const ID a, const ID b)
//	                {
//	                    return a.frameID == b.frameID;
//	                });
//	        auto itOld = it++;
//	        if(duplicateFrameIt != e) {
//	            rootSets.erase(itOld);
//	        }
//	    }
//	}
//
//	return rootSets;
//}

//////////////////////////////////////////////////
// I/O
//////////////////////////////////////////////////

template<typename MATX>
ostream& operator<=(ostream &out, const MATX &m)
{
    for(int i = 0; i < MATX::channels; ++i) {
        out << ' ' << m.val[i];
    }
    return out;
}

template<typename MATX>
istream& operator>=(istream &in, MATX &m)
{
    for(int i = 0; i < MATX::channels; ++i) {
        in >> m.val[i];
    }
    return in;
}

ostream& operator<<(ostream &out, const PoseWithVel &odo)
{
    streamsize oldWidth = out.width();
    char oldFill = out.fill();
    return (out << odo.stamp.sec << '.' << setw(9) << setfill('0') << odo.stamp.nsec
            << setw(oldWidth) << setfill(oldFill))
            <= odo.pos     <= odo.ori     <= odo.pose_cov
            <= odo.vel_lin <= odo.vel_ang <= odo.vel_cov;
}

istream& operator>>(istream &in, PoseWithVel &odo)
{
    char dummy;
    return (in >> odo.stamp.sec >> dummy >> odo.stamp.nsec)
            >= odo.pos    >= odo.ori     >= odo.pose_cov
            >= odo.vel_lin >= odo.vel_ang >= odo.vel_cov;
}

::std::ostream& operator<<( ::std::ostream &out, const Options &o)
{
	return out << NV(o.data_dir) << endl
			<< NV(o.imgs_glob) << endl
			<< NV(o.reconstruction_dir) << endl
			<< NV(o.begin_idx) << NV(o.end_idx) << endl
			<< NV(o.ci.K) << endl
			<< NV(o.ci.k) << endl
			<< NV(o.ci.rows) << NV(o.ci.cols) << endl
			<< NV(o.detector_name) << endl
			// TODO: output detector options
			<< NV(o.extractor_name) << endl
			// TODO: output extractor options
			<< NV(o.matcher_name) << endl
			// TODO: output matcher options
			<< NV(o.match_ratio) << endl
            << NV(o.prune_outliers) << endl
            << NV(o.min_disparity) << endl
			<< NV(o.min_keypoint_distance) << endl
            << NV(o.min_track_length);
}

//::std::ostream& operator<<( ::std::ostream &out, const ::cv::ID &id)
