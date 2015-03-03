#include "precomp.hpp"

namespace cv
{

class SimpleTrackBuilder_Impl : public SimpleTrackBuilder
{
public:
    SimpleTrackBuilder_Impl(double ratio_threshold = 0.7,
            Ptr<FeatureDetector> detector = ORB::create(),
//            Ptr<DescriptorExtractor> extractor = Ptr<DescriptorExtractor>(),
            Ptr<DescriptorMatcher> matcher = Ptr<DescriptorMatcher>());

    virtual void detectAndComputeTracks(InputArrayOfArrays images,
                OutputArrayOfArrays allDescriptors, CV_OUT vvKeyPoint &allKeypoints,
                CV_OUT vvID &tracks);

    virtual void setRatioTestThreshold(double ratio_threshold) { this->ratio_threshold = ratio_threshold; }
    virtual double getRatioTestThreshold() const { return ratio_threshold; }

    virtual void setDetector(Ptr<FeatureDetector> detector) { this->detector = detector; }
    CV_WRAP virtual const Ptr<FeatureDetector> getDetector() const { return detector; }

//    CV_WRAP virtual void setExtractor(Ptr<DescriptorExtractor> extractor) { this->extractor = extractor; }
//    CV_WRAP virtual const Ptr<DescriptorExtractor> getExtractor() const { return extractor; }

    CV_WRAP virtual void setMatcher(Ptr<DescriptorMatcher> matcher) { this->matcher = matcher; }
    CV_WRAP virtual const Ptr<DescriptorMatcher> getMatcher() const { return matcher; }

    virtual AlgorithmInfo* info() const;

protected:
    virtual int getNormType() const { return detector->defaultNorm(); }

    template<typename MAT_I, typename MAT_D>
    void detectAndComputeMatches(const std::vector<MAT_I> &images,
            std::vector<MAT_D> &allDescriptors, vvKeyPoint &allKeypoints);
    void buildTracksFromPairwiseMatches(vvID &tracks);
    void mergeDuplicateKeypointsInTracks(const vvKeyPoint &allKeypoints, vvID &tracks);

    // parameters
    double ratio_threshold;
    Ptr<FeatureDetector> detector;
//    Ptr<DescriptorExtractor> extractor;
    Ptr<DescriptorMatcher> matcher;

    // temporary data
    std::vector<Ptr<DescriptorMatcher> > matchers;
//    std::vector<std::vector<vDMatch> > pairwiseMatches;
    vDMatch pairwiseMatches_ij, pairwiseMatches_ji;
    std::vector<std::vector<vID> > adjacencyLists;
//    Ptr<Tracks> tracks;
};

CV_INIT_ALGORITHM(SimpleTrackBuilder_Impl, "TrackBuilder.Simple",
    obj.info()->addParam(obj, "ratio_threshold", obj.ratio_threshold);
    obj.info()->addParam(obj, "detector", obj.detector);
    /*obj.info()->addParam(obj, "extractor", obj.extractor);*/
    obj.info()->addParam(obj, "matcher", obj.matcher))

Ptr<SimpleTrackBuilder> SimpleTrackBuilder::create(double ratio_threshold,
        Ptr<FeatureDetector> detector,
//        Ptr<DescriptorExtractor> extractor,
        Ptr<DescriptorMatcher> matcher)
{
    return makePtr<SimpleTrackBuilder_Impl>(ratio_threshold, detector, /*extractor, */matcher);
}

SimpleTrackBuilder_Impl::SimpleTrackBuilder_Impl(double ratio_threshold,
            Ptr<FeatureDetector> detector,
//            Ptr<DescriptorExtractor> extractor,
            Ptr<DescriptorMatcher> matcher)
    : ratio_threshold(ratio_threshold)
    , detector(detector)
//    , extractor(extractor)
    , matcher(matcher)
{
    CV_Assert(!!detector);

    if(!matcher) {
        const int normType = getNormType();
        switch(normType) {
        case NORM_L2:
            this->matcher = DescriptorMatcher::create("FlannBased-Autotuned");
            break;
        case NORM_HAMMING:
            this->matcher = DescriptorMatcher::create("FlannBased-LSH");
            break;
        default:
            // TODO: Flann is supported for some other norm types as well
            // try those before defaulting to Brute Force Matcher
            this->matcher = makePtr<BFMatcher>(normType);
        }
    }
}

void SimpleTrackBuilder_Impl::detectAndComputeTracks(InputArrayOfArrays images,
        OutputArrayOfArrays allDescriptors, CV_OUT vvKeyPoint &allKeypoints,
        CV_OUT vvID &tracks)
{
    // check inputs
    CV_Assert(images.isMatVector() || images.isUMatVector());
    CV_Assert(allDescriptors.isMatVector() || allDescriptors.isUMatVector());

    // compute descriptors, keypoints, and matches
    // (disatch based on Mat/UMat)
    if(images.isMatVector() && allDescriptors.isMatVector()) {
        detectAndComputeMatches(*(const vMat *) images.getObj(),
                *(vMat *) allDescriptors.getObj(), allKeypoints);
    } else if(images.isMatVector() && allDescriptors.isUMatVector()) {
        detectAndComputeMatches(*(const vMat *) images.getObj(),
                *(vUMat *) allDescriptors.getObj(), allKeypoints);
    } else if(images.isUMatVector() && allDescriptors.isMatVector()) {
        detectAndComputeMatches(*(const vUMat *) images.getObj(),
                *(vMat *) allDescriptors.getObj(), allKeypoints);
    } else if(images.isUMatVector() && allDescriptors.isUMatVector()) {
        detectAndComputeMatches(*(const vUMat *) images.getObj(),
                *(vUMat *) allDescriptors.getObj(), allKeypoints);
    } else {
        CV_Assert(NULL == "shouldn't get here");
    }

    // build feature tracks
    buildTracksFromPairwiseMatches(tracks);
    // merge duplicate points
    mergeDuplicateKeypointsInTracks(allKeypoints, tracks);
}

template<typename MAT_I, typename MAT_D>
void SimpleTrackBuilder_Impl::detectAndComputeMatches(const std::vector<MAT_I> &images,
        std::vector<MAT_D> &allDescriptors, vvKeyPoint &allKeypoints)
{
    const int N = images.size();
    allDescriptors.resize(N);
    allKeypoints.resize(N);
    matchers.resize(N);
    adjacencyLists.resize(N);

    // TODO: make this parallel
    for(int i=0; i<N; ++i) {
        //keypoint detection
        const bool useProvidedKeypoints = false;
        detector->detectAndCompute(images[i], noArray(), allKeypoints[i],
                allDescriptors[i], useProvidedKeypoints);
        CV_DbgAssert((int)allKeypoints[i].size() == allDescriptors[i].rows);
        const int M = allDescriptors[i].rows;

        // matcher training
        const bool emptyTrainData = true;
        matchers[i] = matcher->clone(emptyTrainData);
        matchers[i]->add(allDescriptors[i]);
        matchers[i]->train();

        adjacencyLists[i].resize(M);
        for(int i=0; i<M; ++i) {
            adjacencyLists[i].clear();
        }
    }

    // compute pairwise symmetric matches
//    pairwiseMatches.resize(N, std::vector<vDMatch>(N));
    // TODO: make this parallel
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            getSymmetricMatches(matchers[i], matchers[j], allDescriptors[i],
                    allDescriptors[j], pairwiseMatches_ij,
                    pairwiseMatches_ji, ratio_threshold);
            CV_DbgAssert(pairwiseMatches_ij.size() == pairwiseMatches_ji.size());
            for(const DMatch &m : pairwiseMatches_ij) {
                const ID f1 = { i, m.queryIdx };
                const ID f2 = { j, m.trainIdx };
                adjacencyLists[i][m.queryIdx].push_back(f2);
                adjacencyLists[j][m.trainIdx].push_back(f1);

            }
        }
    }

    // sanity check: adjacency lists should have entirely unique and sorted members
    for(const auto &al : adjacencyLists) {
        for(const auto &track : al) {
            CV_DbgAssert(is_sorted(track.begin(), track.end()));
            CV_DbgAssert(adjacent_find(track.begin(), track.end()) == track.end());
        }
    }
}

void SimpleTrackBuilder_Impl::buildTracksFromPairwiseMatches(vvID &tracks)
{
    tracks.clear();
    for(size_t i=0; i<adjacencyLists.size(); ++i) {
        DEBUG(i);
        const auto &al = adjacencyLists[i];
        for(size_t j=0; j<al.size(); ++j) {
            DEBUG(j);
            const auto &track = al[j];
            if(track.empty()) continue;
            if(track[0].frameID > (int)i) {
                tracks.push_back(vID(1, {i, j}));
                tracks.back().insert(tracks.back().end(), track.begin(), track.end());
                DEBUG(tracks.back().size());
            }
        }
    }
    DEBUG(tracks.size());

    // TODO: merge tracks?

//    const int N = pairwiseMatches.size();
//
//    Ptr<Tracks> pTracks = Tracks::create();
//    Tracks &t = *pTracks;
////    match_adjacency_lists.clear();
//
//    for(int i=0; i<N; ++i) {
//        for(int j=i+1; j<N; ++j) {
//            for(const DMatch &m : pairwiseMatches[i][j]) {
//                const ID f1 = { i, m.queryIdx };
//                const ID f2 = { j, m.trainIdx };
////                match_adjacency_lists[f1].insert(f2);
//                if(t.isInSomeTrack(f2)) {
//                    if(t.isInSomeTrack(f1)) {
//                        if(t.rootID(f1) == t.rootID(f2)) {
//                            // same track already; do nothing
//                        } else {
//                            // f1 and f2 already in different tracks
//                            // TODO: merge tracks?!?!?
//                            if(t.canMerge(t.rootID(f1), t.rootID(f2))) {
//                                t.merge(t.rootID(f1), t.rootID(f2));
//                            } else {
//                                WARN_STR("STUB");
//                                WARN(i);
//                                WARN(j);
////                                WARN(f1);
////                                WARN(f2);
////                                WARN(t.track(f1));
////                                WARN(t.track(f2));
//                            }
//                        }
//                    } else {
//                        // add f1 to f2's track
//                        t.addToTrack(f1, f2);
//                    }
//                } else {
//                    // f2 has no parent/root, so add to a (possibly new) track
//                    if(t.isInSomeTrack(f1)) {
//                        // add f2 to f1's track
//                        t.addToTrack(f2, f1);
//                    } else {
//                        // new track of f1 and f2
//                        t.makeNewTrack(f1, f2);
//                    }
//                } // if(isInSomeTrack(f2) {} else {}
//            } // for(const DMatch &m : pairwiseMatches[i][j])
//        } // for(int j=i+1; j<N; ++j)
//    } // for(int i=0; i<N; ++i)
//
//    t.getTracks(tracks);
}

void SimpleTrackBuilder_Impl::mergeDuplicateKeypointsInTracks(const vvKeyPoint &allKeypoints,
        vvID &tracks)
{
    for(auto &track : tracks) {
        CV_DbgAssert(!track.empty());
//        INFO(track.size());
        auto r = track.begin();         // read iterator
        auto w = r;                     // write iterator
        const auto e = track.end();     // end iterator
        CV_DbgAssert(is_sorted(r, e));
        while(r!=e) {
            const short i = r->frameID;
            auto r2 = std::find_if_not(r+1, e, [=](const ID id) { return id.frameID == i; });
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
}

} // namespace cv
