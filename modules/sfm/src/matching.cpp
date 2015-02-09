#include <opencv2/sfm.hpp>

namespace cv
{

void pruneUnmatched(const vvDMatch &allMatches, const double maxDistanceRatio, vDMatch &ret)
{
	ret.clear();
	ret.reserve(allMatches.size());

	for (const vDMatch &m : allMatches) {
		CV_Assert(m.size() == 2);
		if (m[0].distance / m[1].distance < maxDistanceRatio)
			ret.push_back(m[0]);
	}
}

vDMatch pruneUnmatched(const vvDMatch &allMatches, const double maxDistanceRatio)
{
	vDMatch ret;

	pruneUnmatched(allMatches, maxDistanceRatio, ret);

	return ret;
}

void pruneAsymmetric(vDMatch &matches_i, vDMatch &matches_j)
{
	vDMatch ret_i, ret_j;
	ret_i.reserve(matches_i.size());
	ret_j.reserve(matches_j.size());

	for (size_t i = 0; i < matches_i.size(); ++i) {
		for (size_t j = 0; j < matches_j.size(); ++j) {
			const DMatch &mi = matches_i[i];
			const DMatch &mj = matches_j[j];
			if (mi.queryIdx == mj.trainIdx && mi.trainIdx == mj.queryIdx) {
				CV_Assert(mi.distance == mj.distance);
				ret_i.push_back(mi);
				ret_j.push_back(mj);
			}
		}
	} //  for(size_t i=0; i<matches_i.size(); ++i)

	std::swap(ret_i, matches_i);
	std::swap(ret_j, matches_j);
}

void getSymmetricMatches(const Ptr<DescriptorMatcher> &matcher,
		InputArray descriptors1, InputArray descriptors2,
		CV_OUT vDMatch &matches_12, CV_OUT vDMatch &matches_21, double maxDistanceRatio)
{
	if(descriptors2.rows() < descriptors1.rows()) {
		getSymmetricMatches(matcher, descriptors2, descriptors1, matches_21, matches_12, maxDistanceRatio);
		return;
	}

	// get 2 best matches from 1 to 2 and vice versa
	vvDMatch matches_12_2nn; // query 1, train 2
	vvDMatch matches_21_2nn; // query 2, train 1
	matcher->knnMatch(descriptors1, descriptors2, matches_12_2nn, 2);
	matcher->knnMatch(descriptors2, descriptors1, matches_21_2nn, 2);

	// prepare output
	CV_DbgAssert(matches_12_2nn.size() <= matches_21_2nn.size());
	const int Nmax = matches_12_2nn.size();
	matches_12.clear(); matches_12.reserve(Nmax);
	matches_21.clear(); matches_21.reserve(Nmax);

	// copy good, symmetric matches
	for(int i=0; i<Nmax; ++i) {
		const vDMatch &m12 = matches_12_2nn[i];
		CV_DbgAssert(m12.size() == 2);
		CV_DbgAssert(m12[0].queryIdx == i);
		CV_DbgAssert(m12[1].queryIdx == i);

		if(m12[0].distance < m12[1].distance * maxDistanceRatio) {
			// match from 1 to 2 is good
			const int j = m12[0].trainIdx;
			const vDMatch &m21 = matches_21_2nn[j];
			CV_DbgAssert(m21.size() == 2);
			CV_DbgAssert(m21[0].queryIdx == j);
			CV_DbgAssert(m21[1].queryIdx == j);

			if(m21[0].trainIdx == i) {
				// match from j points to i
				if(m21[0].distance < m21[1].distance * maxDistanceRatio) {
					// match from j to i passes ratio test
					CV_DbgAssert(m12[0].distance == m21[0].distance);
					matches_12.push_back(m12[0]);
					matches_21.push_back(m21[0]);
				}
			}

		}
	}

}

} // namespace cv
