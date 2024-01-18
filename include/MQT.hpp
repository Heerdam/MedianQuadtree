#pragma once

#include <vector>
#include <array>
#include <variant>
#include <memory>
#include <queue>
#include <iostream>
#include <algorithm>

namespace MQT {

    template<class, class> class MedianQuadTree;
    namespace Detail {
        template<class, class> struct Node;
        template<class, class> struct Bucket;
    }

    //----------------------------------

    using Vec2 = std::array<int32_t, 2>;
    using Vec = std::array<int32_t, 3>;

    //----------------------------------

    template<class T, class ALLOCATOR>
    inline std::ostream& operator<< (std::ostream& s, const MedianQuadTree<T, ALLOCATOR>& t);

    template<class T, class ALLOCATOR>
    inline std::ostream& operator<< (std::ostream& s, const Detail::Node<T, ALLOCATOR>& t);

    template<class T, class ALLOCATOR>
    inline std::ostream& operator<< (std::ostream& s, const Detail::Bucket<T, ALLOCATOR>& t);

    //----------------------------------

    namespace Detail {

        template<class T>
        [[nodiscard]] inline bool isEqual(const T _v1, const T _v2, const T _tol = 1.e-8) noexcept {
            return std::abs( _v1 - _v2 ) <= _tol;
        }//isEqual

        template<class T>
        [[nodiscard]] inline bool isZero(const T _v, const T _tol = 1.e-8) noexcept {
            return isEqual<T>(_v, T(0.), _tol);
        }//isZero

        //----------------------------------

        template<class T, class ALLOCATOR = std::allocator<T>>
        [[nodiscard]] inline std::tuple<int32_t, int32_t, int32_t> naive_tester(
        const std::vector<T, ALLOCATOR>& _map,
        const Vec2& _min,
        const Vec2& _max,
        const int32_t _N,
        const T _h
        ) {
            int32_t h = 0, m = 0, l = 0;
            for (int32_t n0 = std::max(0, _min[0]); n0 < std::min(_N, _max[0]); ++n0) {
                for (int32_t n1 = std::max(0, _min[1]); n1 < std::min(_N, _max[1]); ++n1) {
                    const int32_t i = n1 + n0 * _N;
                    if(i < 0 || i >= int32_t(_map.size())) continue;
                    if(isEqual(_map[i], _h)) m++;
                    if (_map[i] > _h) h++;
                    else l++;
                }
            }
            return { l, m, h };
	    };//naive_tester

        //----------------------------------

        template<class T, class ALLOCATOR = std::allocator<T>>
        struct Bucket {
            Vec2 bmin_, bmax_;
            //------------------
            const int32_t level_;
            const int32_t id_;
            const int32_t N_;
            const std::vector<T, ALLOCATOR>& map_;
            T median_, max_, min_;
            bool isFlat_ = false;
            std::vector<Vec2> l_;
            std::vector<Vec2> h_;
            //----------------
            Bucket(
                const std::vector<T, ALLOCATOR>& _map,
                const Vec2& _bmin,
                const Vec2& _bmax,
                const int32_t _N,
                int32_t _l,
                int32_t _id
            ) : bmin_(_bmin), bmax_(_bmax), level_(_l), id_(_id), N_(_N), map_(_map) {}
            //----------------
            Bucket(Bucket&&) = default;
            Bucket(const Bucket&) = delete;
            Bucket& operator=(Bucket&&) = default;
            Bucket& operator=(const Bucket&) = delete;
            //----------------
            void recompute();
            //----------------
            [[nodiscard]] bool overlap_fast(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            [[nodiscard]] std::tuple<int32_t, int32_t, int32_t> overlap(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            [[nodiscard]] int32_t overlap_border(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            friend std::ostream& MQT::operator<< (std::ostream& s, const Bucket<T, ALLOCATOR>& t);
        };//Bucket

        //----------------

        template<class T, class ALLOCATOR = std::allocator<T>>
        struct Node {
            Vec2 bmin_, bmax_;
            //------------------
            const int32_t level_;
            const int32_t id_;  
            const int32_t N_;    
            int32_t size_;
            T max_, min_;
            bool isFlat_ = false;
            std::array<std::variant<std::unique_ptr<Node>, std::unique_ptr<Bucket<T, ALLOCATOR>>>, 4> c_;
            //----------------
            Node(
                const Vec2& _bmin,
                const Vec2& _bmax,
                const int32_t _N,
                int32_t _l,
                int32_t _id
            ) : bmin_(_bmin), bmax_(_bmax), level_(_l), id_(_id), N_(_N) {}
            //----------------
            Node(Node&&) = default;
            Node(const Node&) = delete;
            Node& operator=(Node&&) = default;
            Node& operator=(const Node&) = delete;
            //----------------
            void recompute();
            //----------------
            [[nodiscard]] bool overlap_fast(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            [[nodiscard]] std::tuple<int32_t, int32_t, int32_t> overlap(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            [[nodiscard]] int32_t overlap_border(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            friend std::ostream& MQT::operator<< (std::ostream& s, const Node<T, ALLOCATOR>& t);
        };//Node

    }

    template<class T, class ALLOCATOR = std::allocator<T>>
    class MedianQuadTree {

        std::unique_ptr<Detail::Node<T, ALLOCATOR>> root_;

        const std::vector<T, ALLOCATOR>& map_;
        const int32_t n0_;
        const int32_t n1_;
        const int32_t h_;

        int32_t N_;
        const int32_t extent_;
        int32_t max_level_;

        int32_t idd = 0;

        friend Detail::Node;
        friend Detail::Bucket;

    public:
        using TYPE = T;
        using ALLOCATOR_T = ALLOCATOR;
        //----------------
        MedianQuadTree(
            const std::vector<T, ALLOCATOR>& _map,
            const int32_t _n0,
            const int32_t _n1,
            const int32_t _h,
            const int32_t _min_bucket_size = 12
        );
        //----------------
        MedianQuadTree(MedianQuadTree&&) = default;
        MedianQuadTree(const MedianQuadTree&) = delete;
        MedianQuadTree& operator=(MedianQuadTree&&) = default;
        MedianQuadTree& operator=(const MedianQuadTree&) = delete;
        ~MedianQuadTree() = default;
        //----------------
        void recompute();
        //----------------
        [[nodiscard]] bool check_fast(const Vec& _pos, const Vec2& _ext) const noexcept;
        //[min, max)
        [[nodiscard]] std::tuple<int32_t, int32_t, int32_t> check_overlap(const Vec2& _min, const Vec2& _max, const T _h) const noexcept;
        [[nodiscard]] int32_t check_border(const Vec& _pos, const Vec2& _ext) const noexcept;
        //----------------
        friend std::ostream& MQT::operator<< (std::ostream& s, const MedianQuadTree<T, ALLOCATOR>& t);
    };//MedianQuadTree

}//MQT

//--------------------------------------------------
//------------------- MedianQuadTree ---------------
//--------------------------------------------------

template<class T, class ALLOCATOR>
void MQT::Detail::Bucket<T, ALLOCATOR>::recompute() {
    // n1: x, n0: y
    max_ = -std::numeric_limits<T>::infinity();
    min_ = std::numeric_limits<T>::infinity();

    std::vector<std::pair<T, Vec2>> m;
    m.reserve((bmax_[0] - bmin_[0] + 1) * (bmax_[1] - bmin_[1] + 1));

    for (int32_t n0 = bmin_[0]; n0 < bmax_[0]; ++n0) {
        for (int32_t n1 = bmin_[1]; n1 < bmax_[1]; ++n1) {
            const int32_t i = n1 + n0 * N_;
            if (i >= int32_t(map_.size())) continue;
            m.push_back({ map_[i], { n0, n1} });
            max_ = std::max(max_, map_[i]);
            min_ = std::min(min_, map_[i]);
        }
    }

    isFlat_ = isEqual(min_, max_);

    std::sort(m.begin(), m.end(), [](const auto& _e1, const auto& _e2) {
        return std::get<0>(_e1) < std::get<0>(_e2);
    });

    if (m.size() % 2 == 0) {
        const int32_t idx = (m.size() + 1) / 2;
        median_ = std::get<0>(m[idx]);

        l_.clear();
        for (int32_t j = 0; j <= idx; ++j)
            l_.push_back(std::get<1>(m[j]));

        h_.clear();
        for (size_t j = idx + 1; j < m.size(); ++j)
            h_.push_back(std::get<1>(m[j]));

    } else {
        const int32_t idx1 = m.size() / 2;
        const int32_t idx2 = idx1 + 1;
        median_ = (std::get<0>(m[idx1]) + std::get<0>(m[idx2])) * 0.5;

        l_.clear();
        for (int32_t j = 0; j <= idx1; ++j)
            l_.push_back(std::get<1>(m[j]));

        h_.clear();
        for (size_t j = idx2; j < m.size(); ++j)
            h_.push_back(std::get<1>(m[j]));
    }


}//MQT::MedianQuadTree::Bucket::recompute

template<class T, class ALLOCATOR>
bool MQT::Detail::Bucket<T, ALLOCATOR>::overlap_fast(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Bucket::overlap_fast

template<class T, class ALLOCATOR>
std::tuple<int32_t, int32_t, int32_t> MQT::Detail::Bucket<T, ALLOCATOR>::overlap(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

    const auto contains = [&](int32_t _n0, int32_t _n1) {
        return _min[0] <= _n0 && _n0 < _max[0] && _min[1] <= _n1 && _n1 < _max[1];
    };

    const bool isPartial = _min[0] > bmin_[0] || _max[0] < bmax_[0] || _min[1] > bmin_[1] || _max[1] < bmax_[1];

    if (isFlat_ && !isPartial) {

        if(isEqual(_h, median_)) return { 0, l_.size() + h_.size(), 0 };
        if (_h >= median_) return { l_.size() + h_.size(), 0, 0 };
        else return { 0, 0, l_.size() + h_.size() };

    }

    //--------------------

    //partial overlap
    if (isPartial) {

        int32_t l = 0, m = 0, h = 0;
        // n1: x, n0: y
        for (int32_t n0 = std::max(bmin_[0], _min[0]); n0 < std::min(bmax_[0], _max[0]); ++n0) {
            for (int32_t n1 = std::max(bmin_[1], _min[1]); n1 < std::min(bmax_[1], _max[1]); ++n1) {
                const int32_t i = n1 + n0 * N_;
                if (i >= int32_t(map_.size())) continue;
                if(isEqual(map_[i], _h)) m++;
                if (map_[i] > _h) h++;
                else l++;
            }
        }

        return { l, m, h };

    }

    //--------------------
    if (_h > median_) {

        int32_t low = l_.size();
        int32_t m = 0;
        int32_t high = 0;

        for (const Vec2 pos : h_) {
            const int32_t i = pos[1] + pos[0] * N_;
            if (!contains(pos[0], pos[1])) continue;

            if(isEqual(map_[i], _h)) m++;
            if (map_[i] >= _h) high++;
            if (map_[i] < _h) low++;
        }

        return { low, m, high };

    } else {

        int32_t low = 0;
        int32_t m = 0;
        int32_t high = h_.size();

        for (const Vec2 pos : l_) {
            const int32_t i = pos[1] + pos[0] * N_;
            if (!contains(pos[0], pos[1])) continue;

            if(isEqual(map_[i], _h)) m++;
            if (map_[i] >= _h) high++;
            if (map_[i] < _h) low++;
        }

        return { low, m, high };

    }

}//MQT::MedianQuadTree::Bucket::overlap

template<class T, class ALLOCATOR>
int32_t MQT::Detail::Bucket<T, ALLOCATOR>::overlap_border(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Bucket::overlap_border

//--------------

template<class T, class ALLOCATOR>
void MQT::Detail::Node<T, ALLOCATOR>::recompute() {

    for (int32_t i = 0; i < 4; ++i){
        switch (c_[0].index()) {
            case 0:
                std::get<0>(c_[i])->recompute();
            break;
            case 1:
                std::get<1>(c_[i])->recompute();
            break;
        }
    }

    //--------------

    min_ = std::numeric_limits<T>::infinity();
    max_ = -std::numeric_limits<T>::infinity();

    switch (c_[0].index()) {
        case 0:
        {			
            for (int32_t i = 0; i < 4; ++i){
                min_ = std::min(std::get<0>(c_[i])->min_, min_);
                max_ = std::max(std::get<0>(c_[i])->max_, max_);
            }
        }
        break;
        case 1:
        {
            for (int32_t i = 0; i < 4; ++i){
                min_ = std::min(std::get<1>(c_[i])->min_, min_);
                max_ = std::max(std::get<1>(c_[i])->max_, max_);
            }
        }
        break;
    }

    isFlat_ = isEqual(min_, max_);

}//MQT::MedianQuadTree::Node::recompute

template<class T, class ALLOCATOR>
bool MQT::Detail::Node<T, ALLOCATOR>::overlap_fast(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Node::overlap_fast

template<class T, class ALLOCATOR>
std::tuple<int32_t, int32_t, int32_t> MQT::Detail::Node<T, ALLOCATOR>::overlap(
    const Vec2& _min,
    const Vec2& _max,
    T _h
) const noexcept {

    const bool isPartial = _min[0] > bmin_[0] || _max[0] < bmax_[0] || _min[1] > bmin_[1] || _max[1] < bmax_[1];
    const bool isH = _h > max_;
    const bool isM = isEqual(max_, _h);

    if (isFlat_) {

        if(isPartial){

            const int32_t min0 = std::max(_min[0], bmin_[0]);
            const int32_t min1 = std::max(_min[1], bmin_[1]);

            const int32_t max0 = std::min(_max[0], bmax_[0]);
            const int32_t max1 = std::min(_max[1], bmax_[1]);

            if (isH) return { (max0 - min0) * (max1 - min1), 0, 0 };
            if (isM) return { 0, (max0 - min0) * (max1 - min1), 0 };
            else return { 0, 0, (max0 - min0) * (max1 - min1) };

        } else {
            const int32_t r = N_ / int32_t(std::pow(2, level_ - 1));
            if(isH) return { r*r, 0, 0};
            if (isM) return { 0, r*r, 0};
            else return { 0, 0, r*r};
        }

    }

    //----------------------------

    int32_t low = 0;
    int32_t m = 0;
    int32_t high = 0;
    switch (c_[0].index()) {
        case 0:
        {
            for (int32_t i = 0; i < 4; ++i) {
                const auto r = std::get<0>(c_[i])->overlap(_min, _max, _h);
                low += std::get<0>(r);
                m += std::get<1>(r);
                high += std::get<2>(r);
            }
        }
        break;
        case 1:
        {
            for (int32_t i = 0; i < 4; ++i) {
                const auto r = std::get<1>(c_[i])->overlap(_min, _max, _h);
                low += std::get<0>(r);
                m += std::get<1>(r);
                high += std::get<2>(r);
            }
        }
        break;
    }
    return { low, m, high };

}//MQT::MedianQuadTree::Node::overlap

template<class T, class ALLOCATOR>
int32_t MQT::Detail::Node<T, ALLOCATOR>::overlap_border(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Node::overlap_border

//--------------

template<class T, class ALLOCATOR>
MQT::MedianQuadTree<T, ALLOCATOR>::MedianQuadTree(
    const std::vector<T, ALLOCATOR>& _map,
    const int32_t _n0,
    const int32_t _n1,
    const int32_t _h,
    const int32_t _min_bucket_size
) : map_(_map), n0_(_n0), n1_(_n1), h_(_h), extent_(_min_bucket_size) {
    using namespace Detail;

    const int32_t s = std::max(_n0, _n1);
    max_level_ = 1;
    N_ = _min_bucket_size;
    while (N_ < s) {
        max_level_++;
        N_ *= 2;
    }
    //--------------
    root_ = std::make_unique<Node<T, ALLOCATOR>>(Vec2{ 0, 0 }, Vec2{ N_, N_ }, N_, 1, ++idd);

    std::queue<Node<T, ALLOCATOR>*> q;
    q.push(root_.get());

    int32_t nc = 0;
    int32_t bc = 0;
    while (!q.empty()) {

        Node<T, ALLOCATOR>* n = q.front();
        q.pop();

        nc++;

        /*
        children:
            0 | 1
            2 | 3
        */
        const Vec2 min_0 = { n->bmin_[0], n->bmin_[1] + (n->bmax_[1] - n->bmin_[1]) / 2 };
        const Vec2 max_0 = { n->bmin_[0] + (n->bmax_[0] - n->bmin_[0]) / 2, n->bmax_[1] };

        const Vec2 min_1 = { n->bmin_[0] + (n->bmax_[0] - n->bmin_[0]) / 2, n->bmin_[1] + (n->bmax_[1] - n->bmin_[1]) / 2 };
        const Vec2 max_1 = n->bmax_;

        const Vec2 min_2 = n->bmin_;
        const Vec2 max_2 = { n->bmin_[0] + (n->bmax_[0] - n->bmin_[0]) / 2, n->bmin_[1] + (n->bmax_[1] - n->bmin_[1]) / 2 };

        const Vec2 min_3 = { n->bmin_[0] + (n->bmax_[0] - n->bmin_[0]) / 2, n->bmin_[1] };
        const Vec2 max_3 = { n->bmax_[0], n->bmin_[1]+ (n->bmax_[1] - n->bmin_[1]) / 2 };

        if (n->level_ == max_level_ - 1) {

            bc += 4;

            // --- 0 ---
            n->c_[0] = std::make_unique<Bucket<T, ALLOCATOR>>(map_, min_0, max_0, N_, n->level_ + 1, ++idd);
            
            // --- 1 ---
            n->c_[1] = std::make_unique<Bucket<T, ALLOCATOR>>(map_, min_1, max_1, N_, n->level_ + 1, ++idd);

            // --- 2 ---
            n->c_[2] = std::make_unique<Bucket<T, ALLOCATOR>>(map_, min_2, max_2, N_, n->level_ + 1, ++idd);

            // --- 3 ---
            n->c_[3] = std::make_unique<Bucket<T, ALLOCATOR>>(map_, min_3, max_3, N_, n->level_ + 1, ++idd);

        } else {

            n->c_[0] = std::make_unique<Node<T, ALLOCATOR>>(min_0, max_0, N_, n->level_ + 1, ++idd);
            q.push(std::get<0>(n->c_[0]).get());

            n->c_[1] = std::make_unique<Node<T, ALLOCATOR>>(min_1, max_1, N_, n->level_ + 1, ++idd);
            q.push(std::get<0>(n->c_[1]).get());

            n->c_[2] = std::make_unique<Node<T, ALLOCATOR>>(min_2, max_2, N_, n->level_ + 1, ++idd);
            q.push(std::get<0>(n->c_[2]).get());

            n->c_[3] = std::make_unique<Node<T, ALLOCATOR>>(min_3, max_3, N_, n->level_ + 1, ++idd);
            q.push(std::get<0>(n->c_[3]).get());

        }
    }

    recompute();

    //std::cout << "Max Level: " << max_level_ << std::endl;
    //std::cout << "Nodes: " << nc << std::endl;
    //std::cout << "Buckets: " << bc << std::endl;
    //std::cout << "N: " << N_ << std::endl;

}//MQT::MedianQuadTree::MedianQuadTree

template<class T, class ALLOCATOR>
void MQT::MedianQuadTree<T, ALLOCATOR>::recompute() {
    root_->recompute();
}//MQT::MedianQuadTree::recompute

template<class T, class ALLOCATOR>
std::tuple<int32_t, int32_t, int32_t> MQT::MedianQuadTree<T, ALLOCATOR>::check_overlap(const Vec2& _min, const Vec2& _max, const T _h) const noexcept {
    return root_->overlap(_min, _max, _h);
}//MQT::MedianQuadTree::check

template<class T, class ALLOCATOR>
bool MQT::MedianQuadTree<T, ALLOCATOR>::check_fast(
    const Vec& _pos, 
    const Vec2& _ext
) const noexcept {

}//MQT::MedianQuadTree::check_fast

template<class T, class ALLOCATOR>
int32_t MQT::MedianQuadTree<T, ALLOCATOR>::check_border(
    const Vec& _pos, 
    const Vec2& _ext
) const noexcept {

}//MQT::MedianQuadTree::check_border

//----------------------------------------------

template<class T, class ALLOCATOR>
inline std::ostream& MQT::operator<< (std::ostream& s, const MQT::Detail::Bucket<T, ALLOCATOR>& t) {
    std::stringstream ss;
    for(int32_t i = 0; i < t.level_; ++i)
        ss << "  ";

    s << ss.str() << "Bucket [id: " << t.id_ << "]" << std::endl;
    s << ss.str() << "Level: " << t.level_ << std::endl;
    s << ss.str() << "bounds: [" << t.bmin_[0] << ", " <<  t.bmin_[1] << "][" << t.bmax_[0] << ", " <<  t.bmax_[1] << "]" << std::endl;
    s << ss.str() << "Interval: [" << t.min_ << ", " << t.median_ << ", " << t.max_ << "]" << std::endl;
    s << ss.str() << "Flat: " << (t.isFlat_ ? "yes" : "no") << std::endl;

    return s;
}//MQT::operator<<

template<class T, class ALLOCATOR>
inline std::ostream& MQT::operator<< (std::ostream& s, const MQT::Detail::Node<T, ALLOCATOR>& t) {
    std::stringstream ss;
    for(int32_t i = 0; i < t.level_; ++i)
        ss << "  ";

    s << ss.str() << "Node [id: " << t.id_ << "]" << std::endl;
    s << ss.str() << "Level: " << t.level_ << std::endl;
    s << ss.str() << "bounds: [" << t.bmin_[0] << ", " <<  t.bmin_[1] << "][" << t.bmax_[0] << ", " <<  t.bmax_[1] << "]" << std::endl;
    s << ss.str() << "Interval: [" << t.min_ << ", " << ", " << t.max_ << "]" << std::endl;
    s << ss.str() << "Flat: " << (t.isFlat_ ? "yes" : "no") << std::endl;
    s << ss.str() << "---------------------------------" << std::endl;
    for(int32_t i = 0; i < 4; ++i){
        if(i != 0) s << ss.str() << "+++++++++++++" << std::endl;
        switch(t.c_[i].index()){
            case 0:
                s << *std::get<0>(t.c_[i]);
            break;
            case 1:
                s << *std::get<1>(t.c_[i]);
            break;
        }
    }
    s << ss.str() << "---------------------------------" << std::endl;

    return s;
}//MQT::operator<<

template<class T, class ALLOCATOR>
inline std::ostream& MQT::operator<< (std::ostream& s, const MQT::MedianQuadTree<T, ALLOCATOR>& t) {
    s << "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-" << std::endl;
    s << *t.root_;
    s << "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-" << std::endl;
    return s;
}//MQT::operator<<
