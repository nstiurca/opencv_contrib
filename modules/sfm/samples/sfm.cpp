//////////////////////////////////////////////////
// INCLUDES
//////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/sfm.hpp>
#include "logging.h"
using namespace std;
using namespace cv;

/////////////////////////////////////////////////
// TYPEDEFS
/////////////////////////////////////////////////
typedef vector<Mat> vMat;
typedef vector<vMat> vvMat;
typedef vector<UMat> vUMat;
typedef vector<vUMat> vvUMat;
typedef vector<DMatch> vDMatch;
typedef vector<vDMatch> vvDMatch;
typedef vector<KeyPoint> vKeyPoint;
typedef vector<String> vString;
typedef Vec<double, 5> Vec5d;
typedef Matx<double, 6,6> Matx66d;
typedef Matx66d Cov6d;
typedef Vec<double, 4> Vec4d;
typedef Vec4d Quatd;

/////////////////////////////////////////////////
// Custom types
/////////////////////////////////////////////////

struct CameraInfo
{
	Matx33d K;
	Vec5d k;
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
};

ostream& operator<<(ostream &out, const PoseWithVel &odo);
istream& operator>>(istream &in, PoseWithVel &odo);
ostream& operator<<(ostream &out, const Options &o);

//////////////////////////////////////////////////
// Prototypes
//////////////////////////////////////////////////
void usage(int argc, char **argv);
Options getOptions(const string &optionsFname);

//////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////
int main(int argc, char **argv)
{
	if(2 != argc) {
		usage(argc, argv);
		exit(-1);
	}

	Options opts = getOptions(argv[1]);

	INFO(opts);

	vString imgNames;
	glob(opts.data_dir + "/" + opts.imgs_glob, imgNames);
	const int N = imgNames.size();

	INFO(N);
	vUMat imgsC(N), imgsG(N);
	for(int i=0; i<N; ++i) {
		imgsC[i] = imread(imgNames[i], IMREAD_COLOR).getUMat(USAGE_ALLOCATE_DEVICE_MEMORY);
		imgsG[i] = imread(imgNames[i], IMREAD_GRAYSCALE).getUMat(USAGE_ALLOCATE_DEVICE_MEMORY);
	}



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
Options getOptions(const string &optionsFname)
{
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

	ret.detector = (String)fs["detector"];
	ret.descriptor = (String)fs["descriptor"];
	ret.matcher = (String)fs["matcher"];
	ret.match_ratio = (double)fs["match_ratio"];

	return ret;
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
			<< NV(o.begin_idx) << endl
			<< NV(o.end_idx) << endl
			<< NV(o.ci.K) << endl
			<< NV(o.ci.k) << endl
			<< NV(o.detector) << endl
			<< NV(o.descriptor) << endl
			<< NV(o.matcher) << endl
			<< NV(o.match_ratio);
}
