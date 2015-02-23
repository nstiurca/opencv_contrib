#include <opencv2/sfm.hpp>
#include <functional>
#include <unordered_map>
#include "../samples/logging.h"

namespace std
{
template<>
struct hash< ::cv::ID>
{
    constexpr size_t hashUS(unsigned short us) const noexcept
            { return static_cast<size_t>(us); }
    constexpr size_t operator()(::cv::ID f) const noexcept
    {
        return    (hashUS(f.frameID) << (sizeof(short) * 8))
                | (hashUS(f.pointID));
    }
};

static_assert(hash< ::cv::ID>()({(short)0, (short)0}) == static_cast<size_t>(0ul), "trivial ID hash");
static_assert(hash< ::cv::ID>()({(short)0, (short)10}) == static_cast<size_t>(10ul), "semi-trivial ID hash");
static_assert(hash< ::cv::ID>()({(short)10, (short)0}) == static_cast<size_t>(10ul << 16), "semi-trivial ID hash");
static_assert(hash< ::cv::ID>()({(short)0x9559, (short)0x5AA5}) == static_cast<size_t>(0x95595AA5ul), "positive ID hash");
static_assert(hash< ::cv::ID>()({(short)0xDEAD, (short)0xBEEF}) == static_cast<size_t>(0xDEADBEEFul), "negative ID hash");

} // namespace std;

namespace /* anonymous */
{
using namespace cv;

class Tracks_Impl : public Tracks
{
    std::unordered_map<ID, sID> tracks;
    std::unordered_map<ID, ID> parents;
    std::unordered_map<ID, ID> roots;

public:
    bool hasParent(const ID id) const {
        return parents.find(id) != parents.end();
    }
    bool hasRoot(const ID id) const {
        return roots.find(id) != roots.end();
    }
    bool isInSomeTrack(const ID id) const {
        CV_DbgAssert(hasParent(id) == hasRoot(id));
        return hasParent(id);
    }
    ID& parentID(const ID id) {
        CV_Assert(isInSomeTrack(id));    // maybe this shouldn't be here??
        return parents[id];
    }
    const ID& parentID(const ID id) const {
        CV_Assert(isInSomeTrack(id));
        return parents.find(id)->second;
    }
    ID& rootID(const ID id) {
        CV_Assert(isInSomeTrack(id));    // maybe this shouldn't be here??
        return roots[id];
    }
    const ID& rootID(const ID id) const {
        CV_Assert(isInSomeTrack(id));
        return roots.find(id)->second;
    }
    sID& track(const ID id) {
        CV_Assert(hasRoot(id));
        sID &t = tracks[rootID(id)];
        CV_DbgAssert(t.find(id) != t.end());
        return t;
    }
    const sID& track(const ID id) const {
        CV_Assert(hasRoot(id));
        const sID &t = tracks.find(rootID(id))->second;
        CV_DbgAssert(t.find(id) != t.end());
        return t;
    }

    void makeNewTrack(const ID f1, const ID f2) {
        if(f2 < f1) {
            INFO_STR(NV(f2.frameID) << ' ' << NV(f2.pointID));
            INFO_STR(NV(f1.frameID) << ' ' << NV(f1.pointID));
            makeNewTrack(f2, f1);
            return;
        }
        CV_Assert(f1 != f2);
        CV_Assert(!isInSomeTrack(f1));
        CV_Assert(!isInSomeTrack(f2));
        tracks[f1] = {f1, f2};
        parents[f1] = parents[f2] = f1;
        roots[f1] = roots[f2] = f1;
    }
    void addToTrack(const ID newPoint, const ID parent) {
        CV_Assert(isInSomeTrack(parent));
        CV_Assert(!isInSomeTrack(newPoint));
        // add newPoint to parent's track
        track(parent).insert(newPoint);
        parents[newPoint] = parent;
        roots[newPoint] = rootID(parent);
    }
    bool canMerge(const ID a, const ID b) const {
        CV_DbgAssert(rootID(a) == a && rootID(b) == b);
        DEBUG_STR("STUB");
        return false;
    }
    void merge(const ID a, const ID b) const {
        CV_DbgAssert(rootID(a) == a && rootID(b) == b);
        DEBUG_STR("STUB");
    }

    void getTracks(vvID &tOut) const
    {
        tOut.clear();
        tOut.reserve(tracks.size());
        for(const auto &t : tracks) {
            DEBUG(t.first);
            DEBUG(t.second.size());
            DEBUG(*t.second.begin());
            tOut.emplace_back(t.second.begin(), t.second.end());
        }
    }
}; // class Tracks_Impl
} // anonymous namespace

namespace cv
{

Ptr<Tracks> Tracks::create()
{
    return makePtr<Tracks_Impl>();
}

} // namespace cv

