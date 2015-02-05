//////////////////////////////////////////////////
// INCLUDES
//////////////////////////////////////////////////

#include <forward_list>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cvv.hpp>
#include <opencv2/sfm.hpp>
#include "logging.h"
using namespace std;
using namespace cv;

/////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////
typedef vector<String> vString;
typedef Vec<double, 5> Vec5d;
typedef Matx<double, 6,6> Matx66d;
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
	Vec5d k;
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
	// Where is the data?
	string data_dir;
	string imgs_glob;
	int begin_idx;
	int end_idx;

	// camera calibration
	CameraInfo ci;

	// Interest point detection and matching
	String detector;
	String descriptor;
	String matcher;
	double match_ratio;

	static Options create(const String &optionsFname);
};

class FeatureTrack
{
public:
	struct ID
	{
		short frameID;
		short pointID;
	};

	struct IDHash
	{
		size_t operator()(const FeatureTrack::ID &f) const
		{
//			hash<uint32_t> h;
//			return h(f.frameID) ^ h(-f.pointID);
			return hash<uint32_t>()((((uint32_t)f.frameID) << 16) | (uint32_t)f.pointID);
		}
	};

	struct Node
	{
		ID parent;
		short rank;

		explicit Node(ID id) : parent(id), rank(0) {}
	};

	void makeSet(const ID id);
	void merge(const ID x, const ID y);
	void link(const ID x, const ID y);
	ID findSet(const ID x);

	void clear() { tracks.clear(); }

	typedef unordered_map<FeatureTrack::ID, vector<FeatureTrack::ID>, IDHash> roots_t;
	roots_t getRootTracks();

private:
	unordered_map<ID, Node, IDHash> tracks;
}; // class FeatureTrack

bool operator==(const FeatureTrack::ID &a, const FeatureTrack::ID &b)
		{ return a.frameID == b.frameID && a.pointID == b.pointID; }
bool operator!=(const FeatureTrack::ID &a, const FeatureTrack::ID &b)
		{ return !(a == b); }

struct SfMMatcher
{
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<Feature2D> feature2d;
	Ptr<DescriptorMatcher> matcher;

	vvKeyPoint allKeypoints;
	vUMat allDescriptors;
	vDescriptorMatcher allMatchers;

	vector<vector<vDMatch>> pairwiseMatches;

	FeatureTrack tracks;

	void detectAndComputeKeypoints( const vUMat &images );
	void computePairwiseSymmetricMatches(const double match_ratio);
	void cvvVisualizePairwiseMatches(InputArrayOfArrays _imgs) const;
	void buildFeatureTracks();

	static SfMMatcher create(const Options &opts);
};

//////////////////////////////////////////////////
// Prototypes
//////////////////////////////////////////////////
void usage(int argc, char **argv);

ostream& operator<<(ostream &out, const PoseWithVel &odo);
istream& operator>>(istream &in, PoseWithVel &odo);
ostream& operator<<(ostream &out, const Options &o);


//////////////////////////////////////////////////
// Init modules
//////////////////////////////////////////////////
bool haveFeatures2d = initModule_features2d();
//bool havexFeatures2d = initModule_xfeatures2d();
bool haveSfm = initModule_sfm();

//////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////
int main(int argc, char **argv)
{
	// args?
	if(2 != argc) {
		usage(argc, argv);
		exit(-1);
	}

	// silly check of which algorithms are available
	{
		vString algorithms;
		Algorithm::getList(algorithms);
		DEBUG(algorithms);
	}

	// parse the options from the YAML file
	Options opts = Options::create(argv[1]);
	INFO(opts);

	// get names of images
	vString imgNames;
	glob(opts.data_dir + "/" + opts.imgs_glob, imgNames);
	if(opts.end_idx>0) {
		imgNames.resize(opts.end_idx);
	}
	if(opts.begin_idx>0) {
		vString trunc(imgNames.begin() + opts.begin_idx, imgNames.end());
		swap(trunc, imgNames);
	}
	const int N = imgNames.size();

	// load images
	INFO(N);
	vUMat imgsC(N), imgsG(N);
	INFO_STR("Loading images: ");
	for(int i=0; i<N; ++i) {
		imgsC[i] = imread(imgNames[i], IMREAD_COLOR).getUMat(USAGE_ALLOCATE_DEVICE_MEMORY);
		imgsG[i] = imread(imgNames[i], IMREAD_GRAYSCALE).getUMat(USAGE_ALLOCATE_DEVICE_MEMORY);

		CV_Assert(imgsC[i].rows == opts.ci.rows);
		CV_Assert(imgsC[i].cols == opts.ci.cols);

		DEBUG(imgsC[i].size());

		cvv::showImage(imgsC[i], CVVISUAL_LOCATION, imgNames[i].c_str());
	}

	// create SfM matcher
	SfMMatcher matcher = SfMMatcher::create(opts);

	// get keypoints
	matcher.detectAndComputeKeypoints(imgsG);

	// do pairwise, symmetric matching over all image pairs
	matcher.computePairwiseSymmetricMatches(opts.match_ratio);
	matcher.cvvVisualizePairwiseMatches(imgsC);

	// build feature tracks
	matcher.buildFeatureTracks();
	FeatureTrack::roots_t rootTracks = matcher.tracks.getRootTracks();
	INFO(rootTracks.size());

	// display tracks
	{
		Mat stitched(opts.ci.rows * N, opts.ci.cols, imgsC[0].type());
		Mat imgTracks[N];
		for(int i=0; i<N; ++i) {
			imgTracks[i] = stitched.rowRange(i*opts.ci.rows, (i+1)*opts.ci.rows);
			CV_Assert(imgTracks[i].rows == imgsC[i].rows);
			CV_Assert(imgTracks[i].cols == imgsC[i].cols);
			CV_Assert(imgTracks[i].type() == imgsC[i].type());
			imgsC[i].copyTo(imgTracks[i]);
		}

		for(auto &track : rootTracks) {
			if(track.second.size() < N) continue;
			INFO(track.first.frameID);
			INFO(track.first.pointID);
			sort(track.second.begin(), track.second.end(), [&](const FeatureTrack::ID &a, const FeatureTrack::ID &b)
					{ return a.frameID < b.frameID; });

			Point pts[N];
			Matx41d c = Scalar::randu(0, 255);
			Scalar color(c.val[0], c.val[1], c.val[2]);
			for(int j=0; j<N; ++j) {
				const FeatureTrack::ID &id = track.second[j];
				pts[j] = matcher.allKeypoints[id.frameID][id.pointID].pt;
				INFO(pts[j]);
				pts[j].y += j*opts.ci.rows;
				circle(stitched, pts[j], 4, color);
				if(j>0) {
					arrowedLine(stitched, pts[j-1], pts[j], color);
				}
			}
			const Point * const pts_ = pts;
		}

		cvv::showImage(stitched, CVVISUAL_LOCATION, "stitched tracks");
	}

	// cvv needs this
	cvv::finalShow();

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
// OPTIONS
//////////////////////////////////////////////////
Options Options::create(const String &optionsFname)
{
	INFO(optionsFname);

	FileStorage fs(optionsFname, FileStorage::READ);
	if(!fs.isOpened()) {
		cout << "could not open file \"" << optionsFname
			<< "\" for reading.";
		exit(-2);
	}

	Options ret;

	ret.data_dir = (string)fs["data_dir"];
	ret.imgs_glob = (string)fs["imgs_glob"];
	ret.begin_idx = (int)fs["begin_idx"];
	ret.end_idx = (int)fs["end_idx"];

	ret.ci.K  <<
			(double)fs["Fx"], 0.0, (double)fs["Px"],
			0.0, (double)fs["Fy"], (double)fs["Py"],
			0.0, 0.0, 1.0;
	ret.ci.k  <<
			(double)fs["k1"],
			(double)fs["k2"],
			(double)fs["k3"],
			(double)fs["k4"],
			(double)fs["k5"];
	ret.ci.rows = (int)fs["ImageH"];
	ret.ci.cols = (int)fs["ImageW"];

	ret.detector = (String)fs["detector"];
	ret.descriptor = (String)fs["descriptor"];
	ret.matcher = (String)fs["matcher"];
	ret.match_ratio = (double)fs["match_ratio"];

	return ret;
}

//////////////////////////////////////////////////
// MATCHER
//////////////////////////////////////////////////
SfMMatcher SfMMatcher::create(const Options &opts)
{
	SfMMatcher ret;

	INFO(opts.detector);
	ret.detector = FeatureDetector::create<FeatureDetector>(opts.detector);
	DEBUG(ret.detector.get());
	if(opts.detector == opts.descriptor) {
		swap(ret.feature2d, ret.detector);
		DEBUG(ret.feature2d.get());
		DEBUG(ret.detector.get());
		INFO_STR("Also using same detector for descriptor extraction");
	} else {
		ret.extractor = DescriptorExtractor::create<DescriptorExtractor>(opts.descriptor);
		INFO(opts.descriptor);
		DEBUG(ret.extractor.get());
	}


	INFO(opts.matcher);
	ret.matcher = DescriptorMatcher::create(opts.matcher);


	DEBUG(ret.detector.get());
	DEBUG(ret.extractor.get());
	DEBUG(ret.feature2d.get());
	DEBUG(ret.matcher.get());

	CV_Assert((ret.detector.empty() && ret.extractor.empty())
			^ ret.feature2d.empty());
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

	CV_Assert(!ret.matcher.empty());

	return ret;
}

void SfMMatcher::detectAndComputeKeypoints( const vUMat &images )
{
	const int N = images.size();
	allKeypoints.resize(N);
	allDescriptors.resize(N);

	InputArray emptymask = noArray();

	const bool combined = !!feature2d;

	for(int i=0; i<N; ++i) {
		if(combined) {
			feature2d->detectAndCompute(images[i], emptymask, allKeypoints[i], allDescriptors[i]);
		} else {
			detector->detect(images[i], allKeypoints[i], emptymask);
			extractor->compute(images[i], allKeypoints[i], allDescriptors[i]);
		}

		cvv::debugDMatch(images[i], allKeypoints[i], images[i], allKeypoints[i], vDMatch(), CVVISUAL_LOCATION, "keypoints");
	}
}

void SfMMatcher::computePairwiseSymmetricMatches(const double match_ratio)
{
    // do pairwise, symmetric matching over all image pairs
    const int N = allDescriptors.size();
    pairwiseMatches.resize(N, vector<vDMatch>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            INFO(i);
            INFO(j);

            getSymmetricMatches(matcher, allDescriptors[i],
                    allDescriptors[j], pairwiseMatches[i][j],
                    pairwiseMatches[j][i], match_ratio);
            INFO(pairwiseMatches[i][j].size());
        }
    }
}

template <typename MAT>
static void cvvVisualizePairwiseMatchesImpl(const vector<MAT> &imgs,
        const vvKeyPoint &allKeypoints, const vector<vector<vDMatch>> &pairwiseMatches)
{
    const int N = imgs.size();
    char description[100];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            snprintf(description, sizeof(description),
                    "symmetric matches %i  %i", i, j);
            cvv::debugDMatch(imgs[i], allKeypoints[i], imgs[j],
                    allKeypoints[j], pairwiseMatches[i][j],
                    CVVISUAL_LOCATION, description);
        }
    }
}

void SfMMatcher::cvvVisualizePairwiseMatches(InputArrayOfArrays _imgs) const
{
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
}

void SfMMatcher::buildFeatureTracks()
{
    // build feature tracks
    const int N = pairwiseMatches.size();
    tracks.clear();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            for (const DMatch &m : pairwiseMatches[i][j]) {
                const FeatureTrack::ID f1 = { i, m.queryIdx };
                const FeatureTrack::ID f2 = { j, m.trainIdx };
                tracks.makeSet(f1);
                tracks.makeSet(f2);
                tracks.merge(f1, f2);
            }
        }
    }
}

//////////////////////////////////////////////////
// FEATURE TRACKS
//////////////////////////////////////////////////
void FeatureTrack::makeSet(const ID id)
{
	auto it = tracks.find(id);
	if(tracks.end() != it) return;

	tracks.emplace(piecewise_construct, make_tuple(id), make_tuple(id));
//	tracks.emplace(make_pair(id, Node{id, 0}));
}

void FeatureTrack::merge(const ID x, const ID y)
{
	link(findSet(x), findSet(y));
}

void FeatureTrack::link(const ID x, const ID y)
{
	auto itX = tracks.find(x);
	auto itY = tracks.find(y);

	CV_Assert(tracks.end() != itX);
	CV_Assert(tracks.end() != itY);

	if(itX->second.rank > itY->second.rank) {
		itY->second.parent = x;
	} else {
		itX->second.parent = y;
		if(itX->second.rank == itY->second.rank) {
			++itY->second.rank;
		}
	}
}

FeatureTrack::ID FeatureTrack::findSet(const ID x)
{
	auto it = tracks.find(x);

	CV_Assert(tracks.end() != it);

	if(x != it->second.parent) {
		it->second.parent = findSet(it->second.parent);
	}

	return it->second.parent;
}

FeatureTrack::roots_t FeatureTrack::getRootTracks()
{
	roots_t rootSets;

	for(auto &idNode : tracks) {
		rootSets[findSet(idNode.first)].push_back(idNode.first);
	}

	return rootSets;
}

//////////////////////////////////////////////////
// I/O
//////////////////////////////////////////////////

template <typename MATX>
ostream& operator<=(ostream &out, const MATX &m)
{
	for(int i=0; i<MATX::channels; ++i) {
		out << ' ' << m.val[i];
	}
	return out;
}

template <typename MATX>
istream& operator>=(istream &in, MATX &m)
{
	for(int i=0; i<MATX::channels; ++i) {
		in >> m.val[i];
	}
	return in;
}

ostream& operator<<(ostream &out, const PoseWithVel &odo)
{
	streamsize oldWidth = out.width();
	char oldFill = out.fill();
	return out << odo.stamp.sec << '.' << setw(9) << setfill('0') << odo.stamp.nsec
			<< setw(oldWidth) << setfill(oldFill)
			<= odo.pos     <= odo.ori     <= odo.pose_cov
			<= odo.vel_lin <= odo.vel_ang <= odo.vel_cov;
}

istream& operator>>(istream &in, PoseWithVel &odo)
{
	char dummy;
	return in >> odo.stamp.sec >> dummy >> odo.stamp.nsec
			>= odo.pos     >= odo.ori     >= odo.pose_cov
			>= odo.vel_lin >= odo.vel_ang >= odo.vel_cov;
}

ostream& operator<<(ostream &out, const Options &o)
{
	return out << NV(o.data_dir) << endl
			<< NV(o.imgs_glob) << endl
			<< NV(o.begin_idx) << NV(o.end_idx) << endl
			<< NV(o.ci.K) << endl
			<< NV(o.ci.k) << endl
			<< NV(o.ci.rows) << NV(o.ci.cols) << endl
			<< NV(o.detector) << endl
			<< NV(o.descriptor) << endl
			<< NV(o.matcher) << endl
			<< NV(o.match_ratio);
}
