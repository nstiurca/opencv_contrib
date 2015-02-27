#include "precomp.hpp"

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
    std::unordered_map<ID, vID> tracks;
    std::unordered_map<ID, ID> parents;
//    std::unordered_map<ID, ID> roots;

public:
    bool isInSomeTrack(const ID id) const {
        return parents.find(id) != parents.end();
    }
    ID rootID(const ID id) {
        CV_Assert(isInSomeTrack(id));    // maybe this shouldn't be here??
        if(parents[id] == id) return id;
        return parents[id] = rootID(parents[id]);   // recursion with path-shortening
    }
    vID& track(const ID id) {
        CV_Assert(isInSomeTrack(id));
        vID &t = tracks[rootID(id)];
        CV_DbgAssert(is_sorted(t.begin(), t.end()));
        CV_DbgAssert(binary_search(t.begin(), t.end(), id));
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
    }
    void addToTrack(const ID newPoint, const ID parent) {
        CV_Assert(isInSomeTrack(parent));
        CV_Assert(!isInSomeTrack(newPoint));
        // add newPoint to parent's track
        vID &t = track(parent);
        CV_DbgAssert(is_sorted(t.begin(), t.end()));
        t.insert(lower_bound(t.begin(), t.end(), newPoint), newPoint);
        CV_DbgAssert(is_sorted(t.begin(), t.end()));
        parents[newPoint] = rootID(parent);
    }

    bool tryMerge(const ID a, const ID b) {
        CV_Assert(rootID(a) == a && rootID(b) == b);
        CV_Assert(a != b);

        CV_DbgAssert(tracks.find(a) != tracks.end());
        CV_DbgAssert(tracks.find(b) != tracks.end());
        CV_DbgAssert(tracks.find(a) != tracks.find(b));

        auto vaIt = tracks.find(a);
        auto vbIt = tracks.find(b);

        vID &va = vaIt->second;
        vID &vb = vbIt->second;

        CV_DbgAssert(is_sorted(va.begin(), va.end()));
        CV_DbgAssert(is_sorted(vb.begin(), vb.end()));

        DEBUG(va);
        DEBUG(vb);

        // first, consider what the merged set would look like
        vID vab(va.size() + vb.size());
        vID::iterator newEnd = std::merge(va.begin(), va.end(), vb.begin(), vb.end(), vab.begin());
        vab.resize(newEnd - vab.begin());
        DEBUG(vab);
        // now look for any frames with multiple keypoints included
        vID::iterator duplicate = adjacent_find(vab.begin(), vab.end(),
                [](const ID a, const ID b) { return a.frameID == b.frameID; });

        bool canMerge = duplicate == vab.end();
        DEBUG(canMerge);

        if(canMerge) {
            // do the actual merge
            if(va.size() >= vb.size()) {
                vab.swap(va);       // replace a with ab
                tracks.erase(vbIt); // erase b
                parents[b] = a;     // point b to a
            } else {
                vab.swap(vb);       // replace b with ab
                tracks.erase(vaIt); // erase a
                parents[a] = b;     // point a to b
            }
        }

        return canMerge;
    }

    void getTracks(vvID &tOut) const
    {
        tOut.clear();
        tOut.reserve(tracks.size());
        for(const auto &t : tracks) {
//            DEBUG(t.first);
//            DEBUG(t.second.size());
//            DEBUG(*t.second.begin());
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

