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

namespace MQT2 {

    using Vec2 = std::array<int32_t, 2>;

    namespace Detail {

        template<typename T>
        struct var_type {
            using type_ = std::conditional_t<std::is_integral_v<T>, float, T>;
        };

        template<typename T>
        using var_type_t = var_type<T>::type_;

        //----------------------------------

        //template<typename T>
        //struct idx {
        //    static constexpr int32_t value_ = std::conditional_t<std::is_integral_v<T>, std::integral_constant<int32_t, 0>, std::integral_constant<int32_t, 1>>::value;
        //};

        //template<typename T>
        //constexpr int32_t idx_v = idx<T>::value_;

        //----------------------------------

        template<class T>
        [[nodiscard]] inline bool isEqual(const T _v1, const T _v2, const T _tol = 1.e-8) noexcept {
            if constexpr (std::is_integral_v<T>)
                return _v1 == _v2;
            else return std::abs( _v1 - _v2 ) <= _tol;
        }//isEqual

        template<class T>
        [[nodiscard]] inline bool isZero(const T _v, const T _tol = 1.e-8) noexcept {
            return isEqual<T>(_v, T(0), _tol);
        }//isZero

        // template<class T>
        // requires std::negation_v<std::is_same<T, float>>
        // [[nodiscard]] inline bool isEqual(const T _v1, const float _v2) noexcept {
        //     return isEqual<float>(float(_v1), _v2, float(1.e-8));
        // }//isEqual

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
                    const auto hh = _map[i];
                    if(Detail::isEqual(hh, _h)) m++;
                    else if(hh > _h) h++;
                    else l++;  
                }
            }
            return { l, m, h };
	    };//naive_tester

        //----------------------------------

        template<class T, int32_t SIZE, class ALLOCATOR = std::allocator<T>>
        struct Bucket {
            int32_t idx_;
            //------------------
            Vec2 bmin_, bmax_;
            //------------------
            bool isMonoton_;
            bool isFlat_;
            //------------------
            Detail::var_type_t<T> median_;
            //------------------
            std::array<T, SIZE * SIZE> vals_;
            //----------------
            Bucket() = default;
            Bucket(
                const int32_t _idx,
                const int32_t _level,
                const Vec2& _bmin,
                const Vec2& _bmax              
            ) :  idx_(_idx), bmin_(_bmin), bmax_(_bmax) {}
            //----------------
            Bucket(Bucket&&) = default;
            Bucket(const Bucket&) = delete;
            Bucket& operator=(Bucket&&) = default;
            Bucket& operator=(const Bucket&) = delete;
        };//Bucket

        //----------------

        template<class T, int32_t SIZE, class ALLOCATOR = std::allocator<T>>
        struct Node {
            Vec2 bmin_, bmax_;
            //------------------
            int32_t level_;
            T max_, min_;
            bool isFlat_;
            //----------------
            Node() = default;
            Node(int32_t _l) : level_(_l) {}
            //----------------
            Node(Node&&) = default;
            Node(const Node&) = delete;
            Node& operator=(Node&&) = default;
            Node& operator=(const Node&) = delete;
            //----------------
        };//Node

    }

    template<class T, int32_t SIZE = 15, class ALLOCATOR = std::allocator<T>>
    class MedianQuadTree {

        const std::vector<T, ALLOCATOR>& map_;
        const int32_t N_;
        int32_t max_level_;

        int32_t idd = 0;

        std::vector<Detail::Bucket<T, SIZE, ALLOCATOR>> b_;
        std::vector<Detail::Node<T, SIZE, ALLOCATOR>> n_;

        void impl_recompute(const int32_t _idx, const int32_t _level, Detail::Node<T, SIZE, ALLOCATOR>& _node, const std::vector<bool>& _m);
        void impl_recompute(Detail::Bucket<T, SIZE, ALLOCATOR>& _bucket, const std::vector<bool>& _m);

        std::tuple<int32_t, int32_t, int32_t> impl_overlap(
            const Vec2& _min,
            const Vec2& _max,
            const T _h,
            const int32_t _idx, 
            const int32_t _level, 
            const Detail::Node<T, SIZE, ALLOCATOR>& _node
        ) const;
        
        std::tuple<int32_t, int32_t, int32_t> impl_overlap(
            const Vec2& _min,
            const Vec2& _max,
            const T _h,
            const Detail::Bucket<T, SIZE, ALLOCATOR>& _bucket
        ) const;

    public:
        constexpr static int32_t BUCKET_SIZE = SIZE;
        using TYPE = T;
        using ALLOCATOR_T = ALLOCATOR;
        //----------------
        MedianQuadTree(
            const std::vector<T, ALLOCATOR>& _map,
            const int32_t _n
        );
        //----------------
        MedianQuadTree(MedianQuadTree&&) = default;
        MedianQuadTree(const MedianQuadTree&) = delete;
        MedianQuadTree& operator=(MedianQuadTree&&) = default;
        MedianQuadTree& operator=(const MedianQuadTree&) = delete;
        ~MedianQuadTree() = default;
        //----------------
        void recompute(const std::vector<bool>& _m);
        //----------------
        //[min, max)
        [[nodiscard]] std::tuple<int32_t, int32_t, int32_t> check_overlap(const Vec2& _min, const Vec2& _max, const T _h) const noexcept;
    };//MedianQuadTree

}//MQT2

template<class T, int32_t SIZE, class ALLOCATOR>
MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::MedianQuadTree(
    const std::vector<T, ALLOCATOR>& _map,
    const int32_t _n
) : map_(_map), N_(_n) 
{

    std::cout << typeid(Detail::var_type_t<T>).name() << std::endl;

    using namespace Detail;

    assert(_n%BUCKET_SIZE == 0);
    //-----------------
    const int32_t bc = _n / BUCKET_SIZE;
    max_level_ = int32_t( std::round(std::log(bc) / std::log(2)) ) + 1;

    const int32_t dc = int32_t((std::pow(4, max_level_ - 1) - 1) / 3);

    n_.resize(dc);
    b_.resize(bc * bc);

    int32_t k = 0;
    for(int32_t j = 0; j < bc; j+=2){
        for(int32_t i = 0; i < bc; i+=2){

            b_[k] = Bucket<T, SIZE, ALLOCATOR>( 
                k,
                max_level_,
                Vec2{i * BUCKET_SIZE, j * BUCKET_SIZE},
                Vec2{i * BUCKET_SIZE + BUCKET_SIZE, j * BUCKET_SIZE + BUCKET_SIZE}
            );
            k++;
            b_[k] = Bucket<T, SIZE, ALLOCATOR>( 
                k,
                max_level_,
                Vec2{(i + 1) * BUCKET_SIZE, j * BUCKET_SIZE},
                Vec2{(i + 1) * BUCKET_SIZE + BUCKET_SIZE, j * BUCKET_SIZE + BUCKET_SIZE}
            );
            k++;
            b_[k] = Bucket<T, SIZE, ALLOCATOR>( 
                k,
                max_level_,
                Vec2{i * BUCKET_SIZE, (j + 1) * BUCKET_SIZE},
                Vec2{i * BUCKET_SIZE + BUCKET_SIZE, (j + 1) * BUCKET_SIZE + BUCKET_SIZE}
            );
            k++;
            b_[k] = Bucket<T, SIZE, ALLOCATOR>( 
                k,
                max_level_,
                Vec2{(i + 1) * BUCKET_SIZE, (j + 1) * BUCKET_SIZE},
                Vec2{(i + 1) * BUCKET_SIZE + BUCKET_SIZE, (j + 1) * BUCKET_SIZE + BUCKET_SIZE}
            );
            k++;
        }
    }
    
    //---------------
    std::vector<bool> mm;
    mm.resize(bc * bc);
    std::fill(mm.begin(), mm.end(), true);
    recompute(mm);

}//MQT2::MedianQuadTree::MedianQuadTree

template<class T, int32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::recompute(const std::vector<bool>& _m) {
    impl_recompute(0, 1, n_[0], _m);
}//MQT2::MedianQuadTree::recompute

template<class T, int32_t SIZE, class ALLOCATOR>
std::tuple<int32_t, int32_t, int32_t> MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::check_overlap(const Vec2& _min, const Vec2& _max, const T _h) const noexcept {
    return impl_overlap(_min, _max, _h, 0, 1, n_[0]);
}

template<class T, int32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::impl_recompute(
    const int32_t _idx, 
    const int32_t _level, 
    Detail::Node<T, SIZE, ALLOCATOR>& _n, 
    const std::vector<bool>& _m
) {
    using namespace Detail;

    if(_level + 1 == max_level_){
        constexpr double frac1 = T(1./3.);
        constexpr double frac2 = T(4./3.);
        const int32_t c1 = int32_t(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2);
        const int32_t c2 = c1 + 1;
        const int32_t c3 = c2 + 1;
        const int32_t c4 = c3 + 1;
        
        impl_recompute(b_[c1], _m);
        impl_recompute(b_[c2], _m);
        impl_recompute(b_[c3], _m);
        impl_recompute(b_[c4], _m);

        _n.min_ = std::min(b_[c1].vals_.front(), std::min(b_[c2].vals_.front(), std::min(b_[c3].vals_.front(), b_[c4].vals_.front())));
        _n.max_ = std::max(b_[c1].vals_.back(), std::max(b_[c2].vals_.back(), std::max(b_[c3].vals_.back(), b_[c4].vals_.back())));
        _n.isFlat_ = isEqual(_n.min_, _n.max_);

        _n.bmin_[0] = std::min(b_[c1].bmin_[0], std::min(b_[c2].bmin_[0], std::min(b_[c3].bmin_[0], b_[c4].bmin_[0])));
        _n.bmin_[1] = std::min(b_[c1].bmin_[1], std::min(b_[c2].bmin_[1], std::min(b_[c3].bmin_[1], b_[c4].bmin_[1])));

        _n.bmax_[0] = std::max(b_[c1].bmax_[0], std::max(b_[c2].bmax_[0], std::max(b_[c3].bmax_[0], b_[c4].bmax_[0])));
        _n.bmax_[1] = std::max(b_[c1].bmax_[1], std::max(b_[c2].bmax_[1], std::max(b_[c3].bmax_[1], b_[c4].bmax_[1])));

    } else {

        const int32_t c1 = 4 * _idx + 1;
        const int32_t c2 = c1 + 1;
        const int32_t c3 = c2 + 1;
        const int32_t c4 = c3 + 1;

        impl_recompute(c1, _level + 1, n_[c1], _m);
        impl_recompute(c2, _level + 1, n_[c2], _m);
        impl_recompute(c3, _level + 1, n_[c3], _m);
        impl_recompute(c4, _level + 1, n_[c4], _m);

        _n.min_ = std::min(n_[c1].min_, std::min(n_[c2].min_, std::min(n_[c3].min_, n_[c4].min_)));
        _n.max_ = std::max(n_[c1].max_, std::max(n_[c2].max_, std::max(n_[c3].max_, n_[c4].max_)));
        _n.isFlat_ = isEqual(_n.min_, _n.max_);

        _n.bmin_[0] = std::min(n_[c1].bmin_[0], std::min(n_[c2].bmin_[0], std::min(n_[c3].bmin_[0], n_[c4].bmin_[0])));
        _n.bmin_[1] = std::min(n_[c1].bmin_[1], std::min(n_[c2].bmin_[1], std::min(n_[c3].bmin_[1], n_[c4].bmin_[1])));

        _n.bmax_[0] = std::max(n_[c1].bmax_[0], std::max(n_[c2].bmax_[0], std::max(n_[c3].bmax_[0], n_[c4].bmax_[0])));
        _n.bmax_[1] = std::max(n_[c1].bmax_[1], std::max(n_[c2].bmax_[1], std::max(n_[c3].bmax_[1], n_[c4].bmax_[1])));
    }

}//MQT2::MedianQuadTree::impl_recompute

template<class T, int32_t SIZE, class ALLOCATOR>
void MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::impl_recompute(Detail::Bucket<T, SIZE, ALLOCATOR>& _b, const std::vector<bool>& _m) {
    using namespace Detail;

    //if(!_m[_b.idx_]) return;

    //_b.max_ = -std::numeric_limits<T>::infinity();
    //_b.min_ = std::numeric_limits<T>::infinity();

    int32_t k = 0;
    for (int32_t n0 = _b.bmin_[0]; n0 < _b.bmax_[0]; ++n0) {
        for (int32_t n1 = _b.bmin_[1]; n1 < _b.bmax_[1]; ++n1) {
            const int32_t i = n1 + n0 * N_;
            _b.vals_[k] = map_[i];
            k++;
            //_b.max_ = std::max(_b.max_, map_[i]);
           // _b.min_ = std::min(_b.min_, map_[i]);
        }
    }

    std::sort(_b.vals_.begin(), _b.vals_.end(), [](const auto _e1, const auto _e2) {
        return _e1 < _e2;
    });

    _b.isFlat_ = isEqual<T>(_b.vals_.front(), _b.vals_.back());
    if(_b.isFlat_){
        //std::cout << _b.idx_ << ", " << _b.vals_.front() << ", " << _b.vals_.back() << std::endl;
        _b.median_ = Detail::var_type_t<T>(_b.vals_.back());
        return;
    }
    
    if constexpr ((BUCKET_SIZE * BUCKET_SIZE) % 2 == 1) {
        constexpr int32_t idx = int32_t((BUCKET_SIZE * BUCKET_SIZE) + 1) / 2;

        _b.median_ = Detail::var_type_t<T>(_b.vals_[idx]);
        _b.isMonoton_ = BUCKET_SIZE >= 3 && !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx - 1]), _b.median_) && 
            !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx + 1]), _b.median_);

    } else {
        constexpr int32_t idx1 = int32_t((BUCKET_SIZE * BUCKET_SIZE) / 2);
        constexpr int32_t idx2 = idx1 + 1;

        _b.median_ = Detail::var_type_t<T>(_b.vals_[idx1] + _b.vals_[idx2]) * Detail::var_type_t<T>(0.5);

        _b.isMonoton_ = BUCKET_SIZE >= 3 && !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx1]), _b.median_) && 
            !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx2]), _b.median_);

    }

}//MQT2::MedianQuadTree::impl_recompute

template<class T, int32_t SIZE, class ALLOCATOR>
std::tuple<int32_t, int32_t, int32_t> MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::impl_overlap(
    const Vec2& _min,
    const Vec2& _max,
    const T _h,
    const int32_t _idx, 
    const int32_t _level, 
    const Detail::Node<T, SIZE, ALLOCATOR>& _n
) const {
    using namespace Detail;

    if(_max[0] < _n.bmin_[0] || _n.bmax_[0] < _min[0] || _max[1] < _n.bmin_[1] || _n.bmax_[1] < _min[1]) return { 0, 0, 0 };

    const int32_t min0 = std::max(_min[0], _n.bmin_[0]);
    const int32_t min1 = std::max(_min[1], _n.bmin_[1]);

    const int32_t max0 = std::min(_max[0], _n.bmax_[0]);
    const int32_t max1 = std::min(_max[1], _n.bmax_[1]);

    const bool isPartial = !(min0 == _n.bmin_[0] && min1 == _n.bmin_[1] && max0 == _n.bmax_[0] && max1 == _n.bmax_[1]);
    const bool isH = _h > _n.max_;
    const bool isM = isEqual<T>(_n.max_, _h);

    if (_n.isFlat_) {

        if(isPartial){

            if (isH) return { (max0 - min0) * (max1 - min1), 0, 0 };
            if (isM) return { 0, (max0 - min0) * (max1 - min1), 0 };
            else return { 0, 0, (max0 - min0) * (max1 - min1) };

        } else {
            const int32_t r = N_ / int32_t(std::pow(2, _level - 1));
            if(isH) return { r*r, 0, 0};
            if (isM) return { 0, r*r, 0};
            else return { 0, 0, r*r};
        }

    }

    //----------------------------

    if(_level + 1 == max_level_){
        constexpr double frac1 = T(1./3.);
        constexpr double frac2 = T(4./3.);
        const int32_t c1 = int32_t(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2);
        const int32_t c2 = c1 + 1;
        const int32_t c3 = c2 + 1;
        const int32_t c4 = c3 + 1;

        const auto[l1, m1, h1] = impl_overlap(_min, _max, _h, b_[c1]);
        const auto[l2, m2, h2] = impl_overlap(_min, _max, _h, b_[c2]);
        const auto[l3, m3, h3] = impl_overlap(_min, _max, _h, b_[c3]);
        const auto[l4, m4, h4] = impl_overlap(_min, _max, _h, b_[c4]);

        return { l1 + l2 + l3 + l4, m1 + m2 + m3 + m4, h1 + h2 + h3 + h4 };
    } else {
        const int32_t c1 = 4 * _idx + 1;
        const int32_t c2 = c1 + 1;
        const int32_t c3 = c2 + 1;
        const int32_t c4 = c3 + 1;

        const auto[l1, m1, h1] = impl_overlap(_min, _max, _h, c1, _level + 1, n_[c1]);
        const auto[l2, m2, h2] = impl_overlap(_min, _max, _h, c2, _level + 1, n_[c2]);
        const auto[l3, m3, h3] = impl_overlap(_min, _max, _h, c3, _level + 1, n_[c3]);
        const auto[l4, m4, h4] = impl_overlap(_min, _max, _h, c4, _level + 1, n_[c4]);

        return { l1 + l2 + l3 + l4, m1 + m2 + m3 + m4, h1 + h2 + h3 + h4 };
    }

}//MQT2::MedianQuadTree::impl_overlap

template<class T, int32_t SIZE, class ALLOCATOR>
std::tuple<int32_t, int32_t, int32_t> MQT2::MedianQuadTree<T, SIZE, ALLOCATOR>::impl_overlap(
    const Vec2& _min,
    const Vec2& _max,
    const T _h,
    const Detail::Bucket<T, SIZE, ALLOCATOR>& _b
) const{
    using namespace Detail;

    const int32_t min0 = std::max(_min[0], _b.bmin_[0]);
    const int32_t min1 = std::max(_min[1], _b.bmin_[1]);

    const int32_t max0 = std::min(_max[0], _b.bmax_[0]);
    const int32_t max1 = std::min(_max[1], _b.bmax_[1]);

    const bool isPartial = !(min0 == _b.bmin_[0] && min1 == _b.bmin_[1] && max0 == _b.bmax_[0] && max1 == _b.bmax_[1]);

    if ( _b.isFlat_ && !isPartial) {
        //std::cout << "flat: ";
        if(Detail::isEqual<T>(_h, _b.vals_.front())){
            //std::cout << 0 << ", " << ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]) << ", " << 0 << std::endl;
            return { 0, ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]), 0 };
        } else if (_h > _b.vals_.front()){
            //std::cout << ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]) << ", " << 0 << ", " << 0 << std::endl;
            return { ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]), 0, 0 };
        } else {
            //std::cout << 0 << ", " << 0 << ", " << ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]) << std::endl;
            return { 0, 0, ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]) };
        }
    }

    //--------------------

    //partial overlap
    if (isPartial) {
        int32_t l = 0, m = 0, h = 0;
        // n1: x, n0: y
        for (int32_t n0 = min0; n0 < max0; ++n0) {
            for (int32_t n1 = min1; n1 < max1; ++n1) {
                const int32_t i = n1 + n0 * N_;
                const auto hh = map_[i];
                //std::cout << hh << std::endl;
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;    
            }
        }
        //std::cout << "partial: " << l << ", " << m << ", " << h << std::endl;
        // std::cout << _b.idx_ << std::endl;
        // for(int32_t n0 = 0; n0 < 40; ++n0){
        //     for(int32_t n1 = 0; n1 < 40; ++n1){
        //         const int32_t i = n1 + n0 * 40;
        //         if(n0 >= _b.bmin_[0] && n0 < _b.bmax_[0] && n1 >= _b.bmin_[1] && n1 < _b.bmax_[1]){
        //             const int32_t i = n1 + n0 * 40;
        //             const auto hh = map_[i];
        //             if(Detail::isEqual<T>(hh, _h)) std::cout << 'm' << ' ';
        //             else if(hh > _h) std::cout << 'h' << ' ';
        //             else std::cout << 'l' << ' ';   
        //         }else
        //             std::cout << '0' << ' ';
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        return { l, m, h };
    }

    if(!_b.isMonoton_){
        int32_t l = 0, m = 0, h = 0;
        for(int32_t i = 0; i < BUCKET_SIZE * BUCKET_SIZE; ++i){
            const auto hh = _b.vals_[i];
            if(Detail::isEqual<T>(hh, _h)) m++;
            else if(hh > _h) h++;
            else l++;  
        }
        //std::cout << "monoton: " << l << ", " << m << ", " << h << std::endl;
        return { l, m, h };
    }

    if constexpr (false){
        int32_t l2 = 0, m2 = 0, h2 = 0;
        for (int32_t n0 = min0; n0 < max0; ++n0) {
            for (int32_t n1 = min1; n1 < max1; ++n1) {
                const int32_t i = n1 + n0 * N_;
                const auto hh = map_[i];
                if(Detail::isEqual<T>(hh, _h)) m2++;
                else if(hh > _h) h2++;
                else l2++;    
            }
        }
        return {l2, m2, h2};
        //std::cout << "----- " << _b.idx_ << "----- " << std::endl;
        //std::cout << "exp: " << l2 << ", " << m2 << ", " << h2 << " (" << l2 + m2 + h2 << ")" << std::endl;
   }

    if constexpr((BUCKET_SIZE * BUCKET_SIZE) % 2 == 1){
        constexpr int32_t idx = int32_t((BUCKET_SIZE * BUCKET_SIZE) + 1) / 2;
        
        if (Detail::var_type_t<T>(_h) > _b.median_) {
            int32_t l = idx, m = 0, h = 0;
            for(int32_t i = idx; i < BUCKET_SIZE * BUCKET_SIZE; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            //std::cout << "> median: " << l << ", " << m << ", " << h << std::endl;
            return { l, m, h };
        } else {
            int32_t l = 0, m = 0, h = idx;
            for(int32_t i = 0; i < idx + 1; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            //std::cout << "< median: " << l << ", " << m << ", " << h << std::endl;
            return { l, m, h };
        }

    } else {
        constexpr int32_t idx = int32_t((BUCKET_SIZE * BUCKET_SIZE) / 2);
        if (Detail::var_type_t<T>(_h) > _b.median_) {
            int32_t l = idx, m = 0, h = 0;
            for(int32_t i = idx; i < BUCKET_SIZE * BUCKET_SIZE; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            return { l, m, h };
        } else {
            int32_t l = 0, m = 0, h = idx;
            for(int32_t i = 0; i < idx; ++i){
                const auto hh = _b.vals_[i];
                if(Detail::isEqual<T>(hh, _h)) m++;
                else if(hh > _h) h++;
                else l++;  
            }
            return { l, m, h };
        }
    }

}//MQT2::MedianQuadTree::impl_overlap
