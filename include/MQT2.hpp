#pragma once

#include <vector>
#include <array>
#include <memory>
#include <queue>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <variant>
#include <iostream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <cmath>

namespace MQT2 {

    template<class T>
    using Vec2 = std::array<T, 2>;

    using Result = std::tuple<uint64_t, uint64_t, uint64_t>;

    //----------------------------------

    namespace Detail {

        template<typename T>
        using var_type_t = std::conditional_t<std::is_integral_v<T>, float, T>;

        //----------------------------------

        template<class T>
        [[nodiscard]] inline bool isEqual(const T _v1, const T _v2) noexcept {
            if constexpr(std::is_integral_v<T>) return _v1 == _v2;
            else return std::abs( _v1 - _v2 ) <= std::numeric_limits<T>::epsilon() * std::max<T>(T(1.), std::max<T>(_v1, _v2));
        }//isEqual

        template<class T>
        [[nodiscard]] inline bool isZero(const T _v) noexcept {
            return isEqual<T>(_v, T(0));
        }//isZero

        //----------------------------------

        template<class T, class B_T, class ALLOCATOR = std::allocator<T>>
        [[nodiscard]] inline Result naive_tester(
        const std::vector<T, ALLOCATOR>& _map,
        const Vec2<B_T>& _min,
        const Vec2<B_T>& _max,
        const uint32_t _N,
        const T _h
        ) noexcept {
            uint32_t h = 0, m = 0, l = 0;
            for (uint32_t n0 = std::max<uint32_t>(0, _min[0]); n0 < std::min<uint32_t>(_N, _max[0]); ++n0) {
                for (uint32_t n1 = std::max<uint32_t>(0, _min[1]); n1 < std::min<uint32_t>(_N, _max[1]); ++n1) {
                    const uint32_t i = n1 + n0 * _N;
                    const auto hh = _map[i];
                    if(Detail::isEqual(hh, _h)) m++;
                    else if(hh > _h) h++;
                    else l++;  
                }
            }
            return { l, m, h };
	    };//naive_tester

        template<class T, class B_T, class ALLOCATOR = std::allocator<T>>
        [[nodiscard]] inline Result naive_border_tester(
        const std::vector<T, ALLOCATOR>& _map,
        const Vec2<B_T>& _min,
        const Vec2<B_T>& _max,
        const uint32_t _N,
        const T _h
        ) noexcept {
            uint32_t h = 0, m = 0, l = 0;
            const uint32_t min0 = std::max<uint32_t>(0, _min[0]);
            const uint32_t max0 = std::min<uint32_t>(_N, _max[0]);
            const uint32_t min1 = std::max<uint32_t>(1, _min[1] + 1);
            const uint32_t max1 = std::min<uint32_t>(_N - 1, _max[1] - 1);
            //----------------------
            for (uint32_t n0 = min0; n0 < max0; ++n0) {
                const uint32_t i1 = (min1 - 1) + n0 * _N;
                const uint32_t i2 = (max1 + 1) + n0 * _N;
                const auto hh1 = _map[i1];
                const auto hh2 = _map[i2];
                if(Detail::isEqual(hh1, _h)) m++;
                else if(hh1 > _h) h++;
                else l++;  
                if(Detail::isEqual(hh2, _h)) m++;
                else if(hh2 > _h) h++;
                else l++; 
            }
            for (uint32_t n1 = min1; n1 < max1; ++n1) {
                const uint32_t i1 = n1 + min0 * _N;
                const uint32_t i2 = n1 + (max1 - 1) * _N;
                const auto hh1 = _map[i1];
                const auto hh2 = _map[i2];
                if(Detail::isEqual(hh1, _h)) m++;
                else if(hh1 > _h) h++;
                else l++;  
                if(Detail::isEqual(hh2, _h)) m++;
                else if(hh2 > _h) h++;
                else l++; 
            }
            return { l, m, h };
	    };//naive_border_tester

        [[nodiscard]] inline uint32_t get_power_of_2(const uint32_t _target, const uint32_t _base) noexcept {
            uint32_t t = _base;
            while (t < _target)
                t  = t << 1;
            return t;
        }//get_power_2

        /*
            0 - uint16_t
            1 - uint32_t
            2 - uint64_t
        */
        // [[nodiscard]] inline int32_t get_type_index_for_size(const uint64_t _N) noexcept {
        //     constexpr uint64_t s16 = uint64_t(std::sqrt(std::numeric_limits<uint16_t>::max())) + 1ul;
        //     constexpr uint64_t s32 = uint64_t(std::sqrt(std::numeric_limits<uint32_t>::max())) + 1ul;
        //     constexpr uint64_t s64 = uint64_t(std::sqrt(std::numeric_limits<uint64_t>::max())) + 1ul;
        //     assert(_N < s64);
        //     if(_N < s16) return 0;
        //     if(_N < s32) return 1;
        //     else return 2;
        // }//get_type_index_for_size

        //----------------------------------

        template<class T, class B_T, uint32_t SIZE>
        struct Bucket {
            size_t idx_;
            uint32_t level_;
            //------------------
            Vec2<B_T> bmin_, bmax_;
            //------------------
            uint8_t isMonoton_ : 1;
            uint8_t isFlat_ : 1;
            //------------------
            Detail::var_type_t<T> median_;
            //------------------
            std::array<T, SIZE * SIZE> vals_;
            //----------------
            Bucket() = default;
            Bucket(
                const uint32_t _idx,
                const uint32_t _level,
                const Vec2<B_T>& _bmin,
                const Vec2<B_T>& _bmax              
            ) :  idx_(_idx), level_(_level), bmin_(_bmin), bmax_(_bmax) {}
            //----------------
            Bucket(Bucket&&) = default;
            Bucket(const Bucket&) = delete;
            Bucket& operator=(Bucket&&) = default;
            Bucket& operator=(const Bucket&) = delete;
            //----------------
        };//Bucket

        //----------------

        template<class T, class B_T>
        struct Node {
            Vec2<B_T> bmin_, bmax_; //8
            T max_, min_; //4
            //uint16_t level_;
            uint8_t isFlat_ : 1; //1
            //----------------
            Node() = default;
            //----------------
            Node(Node&&) = default;
            Node(const Node&) = delete;
            Node& operator=(Node&&) = default;
            Node& operator=(const Node&) = delete;
        };//Node

    }//Detail

    //----------------------------------

    /*
        T: the underlying type of the map
        B_T: type of all Vec2 used for bounds. In most cases uint16_t should be enough
        SIZE: width of a bucket (total bucket size: SIZE*SIZE)
        ALLOCATOR: allocator of the map. allows for aligned maps
    */
    template<class T, class B_T = uint16_t, uint32_t SIZE = 15, class ALLOCATOR = std::allocator<T>>
    class MedianQuadTree {

        static inline std::vector<T, ALLOCATOR> dummy_map_;

        const std::vector<T, ALLOCATOR>& map_;
        uint32_t N_;
        uint32_t max_level_;

        //int32_t idd = 0;

        std::vector<Detail::Bucket<T, B_T, SIZE>> b_;
        std::vector<Detail::Node<T, B_T>> n_;

        //-----------------------

        void impl_recompute(
            const size_t _idx, 
            const uint32_t _level, 
            Detail::Node<T, B_T>& _node, 
            const std::vector<bool>& _m
        ) noexcept;

        void impl_recompute(
            Detail::Bucket<T, B_T, SIZE>& _bucket, 
            const std::vector<bool>& _m
        ) noexcept;

        //-----------------------

        void impl_print(
            const size_t _idx, 
            const uint32_t _level, 
            const Detail::Node<T, B_T>& _node
        ) const;

        void impl_print(
            const Detail::Bucket<T, B_T, SIZE>& _bucket
        ) const;

        //-----------------------

        [[nodiscard]] Result impl_overlap(
            const Vec2<B_T>& _min,
            const Vec2<B_T>& _max,
            const T _h,
            const size_t _idx, 
            const uint32_t _level, 
            const Detail::Node<T, B_T>& _node
        ) const noexcept;
        
        [[nodiscard]] Result impl_overlap(
            const Vec2<B_T>& _min,
            const Vec2<B_T>& _max,
            const T _h,
            const Detail::Bucket<T, B_T, SIZE>& _bucket
        ) const noexcept;

    public:
        constexpr static B_T BUCKET_SIZE = SIZE;
        using TYPE = T;
        using B_Type = B_T;
        using ALLOCATOR_T = ALLOCATOR;
        //----------------
        MedianQuadTree(
            const std::vector<T, ALLOCATOR>& _map,
            const uint32_t _n
        );
        //----------------
        MedianQuadTree();
        MedianQuadTree(MedianQuadTree&&);
        MedianQuadTree(const MedianQuadTree&) = delete;
        MedianQuadTree& operator=(MedianQuadTree&&);
        MedianQuadTree& operator=(const MedianQuadTree&) = delete;
        ~MedianQuadTree() = default;
        //----------------
        void recompute(const std::vector<bool>& _m) noexcept;
        //----------------
        //[min, max)
        [[nodiscard]] Result 
        check_overlap(const Vec2<B_T>& _min, const Vec2<B_T>& _max, const T _h) const noexcept;

        //[min, max)
        [[nodiscard]] Result
        check_border_overlap(const Vec2<B_T>& _min, const Vec2<B_T>& _max, const T _h) const noexcept;
        //----------------
        void print_debug() const;
    };//MedianQuadTree

}//MQT2

//----------------------------------

template <class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::MedianQuadTree() : map_(dummy_map_) {}

template <class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::MedianQuadTree(
    MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR> &&_other
) : map_(_other.map_), N_(_other.N_), max_level_(_other.max_level_), b_(std::move(_other.b_)), n_(std::move(_other.n_)) {}

template <class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR> & MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::operator=(
    MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR> &&_other
) {
  const_cast<std::vector<T, ALLOCATOR> &>(map_) = _other.map_;
  N_ = _other.N_;
  max_level_ = _other.max_level_;
  b_ = std::move(_other.b_);
  n_ = std::move(_other.n_);
  return *this;
}//operator=

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::MedianQuadTree(
    const std::vector<T, ALLOCATOR>& _map,
    const uint32_t _n
) : map_(_map), N_(_n) 
{

    using namespace Detail;

    assert(_n%BUCKET_SIZE == 0);
    //-----------------
    const uint32_t bc = _n / BUCKET_SIZE;
    //max_level_ = int32_t( std::round(std::log(bc) / std::log(2)) ) + 1;
    max_level_ = uint32_t( std::round(std::log2(bc)) ) + 1;

    const uint32_t dc = int32_t((std::pow(4, max_level_ - 1) - 1) / 3);

    n_.resize(dc);
    b_.resize(bc * bc);

    size_t k = 0;
    for(B_T j = 0; j < bc; j+=2){
        for(B_T i = 0; i < bc; i+=2){

            b_[k] = Bucket<T, B_T, SIZE>( 
                i + j * bc,
                max_level_,
                Vec2<B_T>{B_T(i * BUCKET_SIZE), B_T(j * BUCKET_SIZE)},
                Vec2<B_T>{B_T(i * BUCKET_SIZE + BUCKET_SIZE), B_T(j * BUCKET_SIZE + BUCKET_SIZE)}
            );
            k++;
            b_[k] = Bucket<T, B_T, SIZE>( 
                (i + 1) + j * bc,
                max_level_,
                Vec2<B_T>{B_T((i + 1) * BUCKET_SIZE), B_T(j * BUCKET_SIZE)},
                Vec2<B_T>{B_T((i + 1) * BUCKET_SIZE + BUCKET_SIZE), B_T(j * BUCKET_SIZE + BUCKET_SIZE)}
            );
            k++;
            b_[k] = Bucket<T, B_T, SIZE>( 
                i + (j + 1) * bc,
                max_level_,
                Vec2<B_T>{B_T(i * BUCKET_SIZE), B_T((j + 1) * BUCKET_SIZE)},
                Vec2<B_T>{B_T(i * BUCKET_SIZE + BUCKET_SIZE), B_T((j + 1) * BUCKET_SIZE + BUCKET_SIZE)}
            );
            k++;
            b_[k] = Bucket<T, B_T, SIZE>( 
                (i + 1) + (j + 1) * bc,
                max_level_,
                Vec2<B_T>{B_T((i + 1) * BUCKET_SIZE), B_T((j + 1) * BUCKET_SIZE)},
                Vec2<B_T>{B_T((i + 1) * BUCKET_SIZE + BUCKET_SIZE), B_T((j + 1) * BUCKET_SIZE + BUCKET_SIZE)}
            );
            k++;
        }
    }
    
    //---------------
    std::vector<bool> mm;
    mm.resize(bc * bc, true);
    recompute(mm);

}//MQT2::MedianQuadTree::MedianQuadTree

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::recompute(const std::vector<bool>& _m) noexcept {
    impl_recompute(0, 1, n_[0], _m);
}//MQT2::MedianQuadTree::recompute

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::Result MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::check_overlap(
    const Vec2<B_T>& _min, 
    const Vec2<B_T>& _max, 
    const T _h
) const noexcept {
    return impl_overlap(_min, _max, _h, 0, 1, n_[0]);
    //return impl_overlap_it(_min, _max, _h);
}//MQT2::MedianQuadTree::check_overlap

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::Result MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::check_border_overlap(
    const Vec2<B_T>& _min, 
    const Vec2<B_T>& _max, 
    const T _h
) const noexcept {
    return Detail::naive_border_tester<T>(map_, _min, _max, N_, _h);
}//MQT2::MedianQuadTree::check_border_overlap

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::impl_recompute(
    const size_t _idx, 
    const uint32_t _level, 
    Detail::Node<T, B_T>& _n, 
    const std::vector<bool>& _m
) noexcept {
    using namespace Detail;

    if(_level + 1 == max_level_){
        constexpr double frac1 = 1./3.;
        constexpr double frac2 = 4./3.;
        const size_t c1 = size_t(std::round(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2));
        const size_t c2 = c1 + 1;
        const size_t c3 = c2 + 1;
        const size_t c4 = c3 + 1;

        assert(c1 < b_.size() || c2 < b_.size() || c3 < b_.size() || c4 < b_.size());
        
        impl_recompute(b_[c1], _m);
        impl_recompute(b_[c2], _m);
        impl_recompute(b_[c3], _m);
        impl_recompute(b_[c4], _m);

        _n.min_ = std::min<uint32_t>(b_[c1].vals_.front(), std::min<uint32_t>(b_[c2].vals_.front(), std::min<uint32_t>(b_[c3].vals_.front(), b_[c4].vals_.front())));
        _n.max_ = std::max<uint32_t>(b_[c1].vals_.back(), std::max<uint32_t>(b_[c2].vals_.back(), std::max<uint32_t>(b_[c3].vals_.back(), b_[c4].vals_.back())));
        _n.isFlat_ = isEqual(_n.min_, _n.max_);

        _n.bmin_[0] = std::min<uint32_t>(b_[c1].bmin_[0], std::min<uint32_t>(b_[c2].bmin_[0], std::min<uint32_t>(b_[c3].bmin_[0], b_[c4].bmin_[0])));
        _n.bmin_[1] = std::min<uint32_t>(b_[c1].bmin_[1], std::min<uint32_t>(b_[c2].bmin_[1], std::min<uint32_t>(b_[c3].bmin_[1], b_[c4].bmin_[1])));

        _n.bmax_[0] = std::max<uint32_t>(b_[c1].bmax_[0], std::max<uint32_t>(b_[c2].bmax_[0], std::max<uint32_t>(b_[c3].bmax_[0], b_[c4].bmax_[0])));
        _n.bmax_[1] = std::max<uint32_t>(b_[c1].bmax_[1], std::max<uint32_t>(b_[c2].bmax_[1], std::max<uint32_t>(b_[c3].bmax_[1], b_[c4].bmax_[1])));

    } else {

        const size_t c1 = 4 * _idx + 1;
        const size_t c2 = c1 + 1;
        const size_t c3 = c2 + 1;
        const size_t c4 = c3 + 1;

        assert(c1 < n_.size() || c2 < n_.size() || c3 < n_.size() || c4 < n_.size());

        impl_recompute(c1, _level + 1, n_[c1], _m);
        impl_recompute(c2, _level + 1, n_[c2], _m);
        impl_recompute(c3, _level + 1, n_[c3], _m);
        impl_recompute(c4, _level + 1, n_[c4], _m);

        _n.min_ = std::min<uint32_t>(n_[c1].min_, std::min<uint32_t>(n_[c2].min_, std::min<uint32_t>(n_[c3].min_, n_[c4].min_)));
        _n.max_ = std::max<uint32_t>(n_[c1].max_, std::max<uint32_t>(n_[c2].max_, std::max<uint32_t>(n_[c3].max_, n_[c4].max_)));
        _n.isFlat_ = isEqual(_n.min_, _n.max_);

        _n.bmin_[0] = std::min<uint32_t>(n_[c1].bmin_[0], std::min<uint32_t>(n_[c2].bmin_[0], std::min<uint32_t>(n_[c3].bmin_[0], n_[c4].bmin_[0])));
        _n.bmin_[1] = std::min<uint32_t>(n_[c1].bmin_[1], std::min<uint32_t>(n_[c2].bmin_[1], std::min<uint32_t>(n_[c3].bmin_[1], n_[c4].bmin_[1])));

        _n.bmax_[0] = std::max<uint32_t>(n_[c1].bmax_[0], std::max<uint32_t>(n_[c2].bmax_[0], std::max<uint32_t>(n_[c3].bmax_[0], n_[c4].bmax_[0])));
        _n.bmax_[1] = std::max<uint32_t>(n_[c1].bmax_[1], std::max<uint32_t>(n_[c2].bmax_[1], std::max<uint32_t>(n_[c3].bmax_[1], n_[c4].bmax_[1])));
    }

}//MQT2::MedianQuadTree::impl_recompute

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::impl_recompute(
    Detail::Bucket<T, B_T, SIZE>& _b, 
    const std::vector<bool>& _m
) noexcept {
    using namespace Detail;

    if(!_m[_b.idx_]) return;

    size_t k = 0;
    for (uint32_t n0 = _b.bmin_[0]; n0 < _b.bmax_[0]; ++n0) {
        for (uint32_t n1 = _b.bmin_[1]; n1 < _b.bmax_[1]; ++n1) {
            const size_t i = n1 + n0 * N_;
            _b.vals_[k] = map_[i];
            k++;
        }
    }

    std::sort(_b.vals_.begin(), _b.vals_.end(), [](const auto _e1, const auto _e2) {
        return _e1 < _e2;
    });

    _b.isFlat_ = isEqual<T>(_b.vals_.front(), _b.vals_.back());
    if(_b.isFlat_){
        _b.median_ = Detail::var_type_t<T>(_b.vals_.back());
        return;
    }
    
    if constexpr ((BUCKET_SIZE * BUCKET_SIZE) % 2 == 1) {
        constexpr size_t idx = int32_t(BUCKET_SIZE * BUCKET_SIZE) / 2;

        _b.median_ = Detail::var_type_t<T>(_b.vals_[idx]);
        _b.isMonoton_ = BUCKET_SIZE >= 3 && !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx - 1]), _b.median_) && 
            !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx + 1]), _b.median_);

    } else {
        constexpr size_t idx1 = size_t((BUCKET_SIZE * BUCKET_SIZE) / 2) - 1;
        constexpr size_t idx2 = idx1 + 1;

        _b.median_ = Detail::var_type_t<T>(_b.vals_[idx1] + _b.vals_[idx2]) * Detail::var_type_t<T>(0.5);

        _b.isMonoton_ = BUCKET_SIZE >= 3 && !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx1]), _b.median_) && 
            !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx2]), _b.median_);

    }

}//MQT2::MedianQuadTree::impl_recompute

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::Result MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::impl_overlap(
    const Vec2<B_T>& _min,
    const Vec2<B_T>& _max,
    const T _h,
    const size_t _idx, 
    const uint32_t _level, 
    const Detail::Node<T, B_T>& _n
) const noexcept {
    using namespace Detail;

    if(_max[0] < _n.bmin_[0] || _n.bmax_[0] < _min[0] || _max[1] < _n.bmin_[1] || _n.bmax_[1] < _min[1]) return { 0, 0, 0 };

    const B_T min0 = std::max<B_T>(_min[0], _n.bmin_[0]);
    const B_T min1 = std::max<B_T>(_min[1], _n.bmin_[1]);

    const B_T max0 = std::min<B_T>(_max[0], _n.bmax_[0]);
    const B_T max1 = std::min<B_T>(_max[1], _n.bmax_[1]);

    const bool isPartial = !(min0 == _n.bmin_[0] && min1 == _n.bmin_[1] && max0 == _n.bmax_[0] && max1 == _n.bmax_[1]);
    const bool isH = _h > _n.max_;
    const bool isM = isEqual<T>(_n.max_, _h);

    if (_n.isFlat_) {

        if(isPartial){

            if (isH) return { (max0 - min0) * (max1 - min1), 0, 0 };
            if (isM) return { 0, (max0 - min0) * (max1 - min1), 0 };
            else return { 0, 0, (max0 - min0) * (max1 - min1) };

        } else {
            const uint32_t r = N_ / uint32_t(std::round(std::pow(2, _level - 1)));
            if(isH) return { r*r, 0, 0};
            if (isM) return { 0, r*r, 0};
            else return { 0, 0, r*r};
        }

    }

    //----------------------------

    if(_level + 1 == max_level_){
        constexpr double frac1 = 1./3.;
        constexpr double frac2 = 4./3.;
        const size_t c1 = size_t(std::round(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2));
        const size_t c2 = c1 + 1;
        const size_t c3 = c2 + 1;
        const size_t c4 = c3 + 1;

        assert(c1 < b_.size() || c2 < b_.size() || c3 < b_.size() || c4 < b_.size());

        const auto[l1, m1, h1] = impl_overlap(_min, _max, _h, b_[c1]);
        const auto[l2, m2, h2] = impl_overlap(_min, _max, _h, b_[c2]);
        const auto[l3, m3, h3] = impl_overlap(_min, _max, _h, b_[c3]);
        const auto[l4, m4, h4] = impl_overlap(_min, _max, _h, b_[c4]);

        return { l1 + l2 + l3 + l4, m1 + m2 + m3 + m4, h1 + h2 + h3 + h4 };
    } else {
        const size_t c1 = 4 * _idx + 1;
        const size_t c2 = c1 + 1;
        const size_t c3 = c2 + 1;
        const size_t c4 = c3 + 1;

        assert(c1 < n_.size() || c2 < n_.size() || c3 < n_.size() || c4 < n_.size());

        const auto[l1, m1, h1] = impl_overlap(_min, _max, _h, c1, _level + 1, n_[c1]);
        const auto[l2, m2, h2] = impl_overlap(_min, _max, _h, c2, _level + 1, n_[c2]);
        const auto[l3, m3, h3] = impl_overlap(_min, _max, _h, c3, _level + 1, n_[c3]);
        const auto[l4, m4, h4] = impl_overlap(_min, _max, _h, c4, _level + 1, n_[c4]);

        return { l1 + l2 + l3 + l4, m1 + m2 + m3 + m4, h1 + h2 + h3 + h4 };
    }

}//MQT2::MedianQuadTree::impl_overlap

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
MQT2::Result MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::impl_overlap(
    const Vec2<B_T>& _min,
    const Vec2<B_T>& _max,
    const T _h,
    const Detail::Bucket<T, B_T, SIZE>& _b
) const noexcept {
    using namespace Detail;

    const uint32_t min0 = std::max<uint32_t>(_min[0], _b.bmin_[0]);
    const uint32_t min1 = std::max<uint32_t>(_min[1], _b.bmin_[1]);

    const uint32_t max0 = std::min<uint32_t>(_max[0], _b.bmax_[0]);
    const uint32_t max1 = std::min<uint32_t>(_max[1], _b.bmax_[1]);

    const bool isPartial = !(min0 == _b.bmin_[0] && min1 == _b.bmin_[1] && max0 == _b.bmax_[0] && max1 == _b.bmax_[1]);

    if ( _b.isFlat_ && !isPartial) {
        if(Detail::isEqual<T>(_h, _b.vals_.front())) return { 0, ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]), 0 };   
        else if (_h > _b.vals_.front()) return { ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]), 0, 0 };
        else return { 0, 0, ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]) };
    }

    //--------------------

    //partial overlap
    if (isPartial) {
        uint32_t l = 0, m = 0, h = 0;
        // n1: x, n0: y
        for (uint32_t n0 = min0; n0 < max0; ++n0) {
            for (uint32_t n1 = min1; n1 < max1; ++n1) {
                const size_t i = n1 + n0 * N_;
                const auto hh = map_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;    
            }
        }
        return { l, m, h };
    }

    if(!_b.isMonoton_){
        uint32_t l = 0, m = 0, h = 0;
        for(size_t i = 0; i < BUCKET_SIZE * BUCKET_SIZE; ++i){
            const auto hh = _b.vals_[i];
            if(Detail::isEqual<T>(hh, _h)) m++;
            else if(hh > _h) h++;
            else l++;  
        }
        return { l, m, h };
    }

    if constexpr (false){
        uint32_t l2 = 0, m2 = 0, h2 = 0;
        for (uint32_t n0 = min0; n0 < max0; ++n0) {
            for (uint32_t n1 = min1; n1 < max1; ++n1) {
                const size_t i = n1 + n0 * N_;
                const auto hh = map_[i];
                if(Detail::isEqual<T>(hh, _h)) m2++;
                else if(hh > _h) h2++;
                else l2++;    
            }
        }
        return {l2, m2, h2};
   }

    if constexpr((BUCKET_SIZE * BUCKET_SIZE) % 2 == 1){
        constexpr size_t idx = int32_t(BUCKET_SIZE * BUCKET_SIZE) / 2;
        
        if (Detail::var_type_t<T>(_h) > _b.median_) {
            uint32_t l = idx, m = 0, h = 0;
            for(size_t i = idx; i < BUCKET_SIZE * BUCKET_SIZE; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            return { l, m, h };
        } else {
            uint32_t l = 0, m = 0, h = idx;
            for(size_t i = 0; i < idx + 1; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            return { l, m, h };
        }

    } else {
        constexpr size_t idx = size_t((BUCKET_SIZE * BUCKET_SIZE) / 2);
        if (Detail::var_type_t<T>(_h) > _b.median_) {
            uint32_t l = idx, m = 0, h = 0;
            for(size_t i = idx; i < BUCKET_SIZE * BUCKET_SIZE; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            return { l, m, h };
        } else {
            uint32_t l = 0, m = 0, h = idx;
            for(size_t i = 0; i < idx; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            return { l, m, h };
        }
    }

}//MQT2::MedianQuadTree::impl_overlap

//----------------------------------------------
template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::print_debug() const {
    std::cout << "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-" << std::endl;
    std::cout << "Nodes: " << n_.size() << std::endl;
    std::cout << "Buckets: " << b_.size() << std::endl;
    impl_print(0, 1, n_[0]);
    std::cout << "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-" << std::endl;
}//MQT2::MedianQuadTree::print_debug

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::impl_print(
    const size_t _idx, 
    const uint32_t _level, 
    const Detail::Node<T, B_T>& t
) const {
    std::stringstream ss;
    for(uint32_t i = 0; i < _level; ++i)
        ss << "  ";

    std::cout << ss.str() << "Level: " << _level << std::endl;
    std::cout << ss.str() << "bounds: [" << t.bmin_[0] << ", " <<  t.bmin_[1] << "][" << t.bmax_[0] << ", " <<  t.bmax_[1] << "]" << std::endl;
    std::cout << ss.str() << "Interval: [" << t.min_ << ", " << ", " << t.max_ << "]" << std::endl;
    std::cout << ss.str() << "Flat: " << (t.isFlat_ ? "yes" : "no") << std::endl;
    std::cout << ss.str() << "---------------------------------" << std::endl;

    if(_level + 1 == max_level_){

        constexpr double frac1 = 1./3.;
        constexpr double frac2 = 4./3.;
        const size_t c1 = size_t(std::round(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2));
        const size_t c2 = c1 + 1;
        const size_t c3 = c2 + 1;
        const size_t c4 = c3 + 1;

        impl_print(b_[c1]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;
        impl_print(b_[c2]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;
        impl_print(b_[c3]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;
        impl_print(b_[c4]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;

    } else {

        const size_t c1 = 4 * _idx + 1;
        const size_t c2 = c1 + 1;
        const size_t c3 = c2 + 1;
        const size_t c4 = c3 + 1;

        impl_print(c1, _level + 1, n_[c1]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;
        impl_print(c2, _level + 1, n_[c2]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;
        impl_print(c3, _level + 1, n_[c3]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;
        impl_print(c4, _level + 1, n_[c4]);
        std::cout << ss.str() << "+++++++++++++" << std::endl;

    }
    std::cout << ss.str() << "---------------------------------" << std::endl;
}//MQT2::MedianQuadTree::impl_print

template<class T, class B_T, uint32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, B_T, SIZE, ALLOCATOR>::impl_print(const Detail::Bucket<T, B_T, SIZE>& t) const {
    std::stringstream ss;
    for(uint32_t i = 0; i < t.level_; ++i)
        ss << "  ";
 
    std::cout << ss.str() << "-----------" << std::endl;
    std::cout << ss.str() << "Bucket [id: " << t.idx_ << "]" << std::endl;
    std::cout << ss.str() << "bounds: [" << t.bmin_[0] << ", " <<  t.bmin_[1] << "][" << t.bmax_[0] << ", " <<  t.bmax_[1] << "]" << std::endl;
    std::cout << ss.str() << "Interval: [" << t.vals_.front() << ", " << t.median_ << ", " << t.vals_.back() << "]" << std::endl;
    std::cout << ss.str() << "Flat: " << (t.isFlat_ ? "yes" : "no") << std::endl;
    std::cout << ss.str() << "Monoton: " << (t.isMonoton_ ? "yes" : "no") << std::endl;
}//MQT2::MedianQuadTree::impl_print
