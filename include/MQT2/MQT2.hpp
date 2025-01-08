/*
MIT License

Copyright (c) Damian Heer
https://github.com/Heerdam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <type_traits>
#include <variant>
#include <vector>

namespace MQT2 {

#ifndef TRIPLET_INT_TYPE
#define TRIPLET_INT_TYPE uint32_t
#endif

#ifndef VECTOR_INT_TYPE
#define VECTOR_INT_TYPE uint32_t
#endif

  using Vec2    = std::array<VECTOR_INT_TYPE, 2>;

  using Triplet = std::tuple<TRIPLET_INT_TYPE, TRIPLET_INT_TYPE, TRIPLET_INT_TYPE>;

  //----------------------------------

  template <class T, uint32_t BucketSize, class Allocator>
  class MedianQuadTree;

  namespace Detail {

    template <typename T>
    using var_type_t = std::conditional_t<std::is_integral_v<T>, float, T>;

    //----------------------------------

    template <class T>
    [[nodiscard]] constexpr bool isEqual(const T _v1, const T _v2) noexcept {
      if constexpr (std::is_integral_v<T>)
        return _v1 == _v2;
      else
        return std::abs(_v1 - _v2) <= std::numeric_limits<T>::epsilon() * std::max<T>(T(1.), std::max<T>(_v1, _v2));
    }  // isEqual

    template <class T>
    [[nodiscard]] constexpr bool isZero(const T _v) noexcept {
      return isEqual<T>(_v, T(0));
    }  // isZero

    template <class T>
    [[nodiscard]] constexpr T get_power_of_2(const T _target, const T _base) noexcept
      requires(std::is_integral_v<T>)
    {
      T t = _base;
      while (t < _target) t = t << 1;
      return std::max(2 * _base, t);
    }  // get_power_2

    //----------------------------------

    // computes the tri-convolution using a naive, brute force approach. the domain is clamped to 0 and or the max
    // value. The bounds are [min[0], max[0]) x [min[1], max[1])
    template <class T, class ALLOCATOR = std::allocator<T>>
    [[nodiscard]] Triplet naive_tester(const std::vector<T, ALLOCATOR>& _map, const Vec2& _min, const Vec2& _max,
                                       const uint32_t _N, const T _h) noexcept {
      uint32_t h = 0, m = 0, l = 0;
      for (uint32_t n0 = std::max<uint32_t>(0, _min[0]); n0 < std::min<uint32_t>(_N, _max[0]); ++n0) {
        for (uint32_t n1 = std::max<uint32_t>(0, _min[1]); n1 < std::min<uint32_t>(_N, _max[1]); ++n1) {
          const uint32_t i = n1 + n0 * _N;
          const auto hh    = _map[i];
          if (Detail::isEqual(hh, _h))
            m++;
          else if (hh > _h)
            h++;
          else
            l++;
        }
      }
      return {l, m, h};
    };  // naive_tester

    // computes the tri-convolution of the border using a naive, brute force approach.
    // the border is the domain expanded by 1 in all 4 directions. the border is clamped to 0 and or the max value.
    // border of [min[0], max[0]) x [min[1], max[1])
    template <class T, class ALLOCATOR = std::allocator<T>>
    [[nodiscard]] Triplet naive_border_tester(const std::vector<T, ALLOCATOR>& _map, const Vec2& _min, const Vec2& _max,
                                              const uint32_t _N, const T _h) noexcept {
      uint32_t h = 0, m = 0, l = 0;
      // lowest index
      const uint32_t min0 = std::max<uint32_t>(0, _min[0]);
      const uint32_t min1 = std::max<uint32_t>(0, _min[1]);
      // highest index
      const uint32_t max0 = std::min<uint32_t>(_N, _max[0]) - 1;
      const uint32_t max1 = std::min<uint32_t>(_N, _max[1]) - 1;
      //----------------------
      for (uint32_t n0 = min0; n0 < max0 + 1; ++n0) {
        const uint32_t i1 = min1 + n0 * _N;
        const uint32_t i2 = max1 + n0 * _N;
        const auto hh1    = _map[i1];
        const auto hh2    = _map[i2];
        if (Detail::isEqual(hh1, _h))
          m++;
        else if (hh1 > _h)
          h++;
        else
          l++;
        if (Detail::isEqual(hh2, _h))
          m++;
        else if (hh2 > _h)
          h++;
        else
          l++;
      }
      // offset by 1 to not count corners twice
      for (uint32_t n1 = min1 + 1; n1 < max1; ++n1) {
        const uint32_t i1 = n1 + min0 * _N;
        const uint32_t i2 = n1 + max0 * _N;
        const auto hh1    = _map[i1];
        const auto hh2    = _map[i2];
        if (Detail::isEqual(hh1, _h))
          m++;
        else if (hh1 > _h)
          h++;
        else
          l++;
        if (Detail::isEqual(hh2, _h))
          m++;
        else if (hh2 > _h)
          h++;
        else
          l++;
      }
      return {l, m, h};
    };  // naive_border_tester

    //----------------------------------

    template <class T, uint32_t BucketSize>
    struct Bucket {
      size_t idx_;
      uint32_t level_;
      //------------------
      Vec2 bmin_, bmax_;
      //------------------
      uint8_t isMonoton_ : 1;
      uint8_t isFlat_ : 1;
      //------------------
      Detail::var_type_t<T> median_;
      //------------------
      std::array<T, BucketSize * BucketSize> vals_;
      //----------------
      Bucket() = default;
      Bucket(const uint32_t _idx, const uint32_t _level, const Vec2& _bmin, const Vec2& _bmax)
          : idx_(_idx), level_(_level), bmin_(_bmin), bmax_(_bmax) {}
      //----------------
      Bucket(Bucket&&)                 = default;
      Bucket(const Bucket&)            = delete;
      Bucket& operator=(Bucket&&)      = default;
      Bucket& operator=(const Bucket&) = delete;
    };  // Bucket

    //----------------

    template <class T>
    struct Node {
      Vec2 bmin_, bmax_;
      T max_, min_;
      uint8_t isFlat_ : 1;
      //----------------
      Node()                       = default;
      //----------------
      Node(Node&&)                 = default;
      Node(const Node&)            = delete;
      Node& operator=(Node&&)      = default;
      Node& operator=(const Node&) = delete;
    };  // Node

    //----------------------------------

    template <class T, uint32_t BucketSize, class Allocator>
    void impl_recompute(std::vector<T, Allocator>* _map, const size_t _N,
                        std::vector<Detail::Bucket<T, BucketSize>>& _b, std::vector<Detail::Node<T>>& _n,
                        const size_t _max_level, const size_t _idx, const uint32_t _level, Detail::Node<T>& _node,
                        const std::vector<bool>& _m) noexcept;

    template <class T, uint32_t BucketSize, class Allocator>
    void impl_recompute(std::vector<T, Allocator>* _map, const size_t _N, Detail::Bucket<T, BucketSize>& _bucket,
                        const std::vector<bool>& _m) noexcept;

    //-----------------------

    // template <class T, uint32_t BucketSize, class Allocator>
    // void impl_print(std::stringstream& _ss, const MedianQuadTree<T, BucketSize, Allocator>& _tree);

    // template <class T, uint32_t BucketSize, class Allocator>
    // void impl_print(std::stringstream& _ss, const Detail::Bucket<T, BucketSize>& _bucket);

    // template <class T, uint32_t BucketSize, class Allocator>
    // void impl_print(std::stringstream& _ss, const size_t _idx, const uint32_t _level, const Detail::Node<T>& _node);

    //-----------------------

    template <class T, uint32_t BucketSize, class Allocator>
    [[nodiscard]] Triplet impl_overlap(const std::vector<T, Allocator>* _map, const size_t _N,
                                       const std::vector<Detail::Bucket<T, BucketSize>>& _b,
                                       const std::vector<Detail::Node<T>>& _n, const size_t _max_level,
                                       const Vec2& _min, const Vec2& _max, const T _h, const size_t _idx,
                                       const uint32_t _level, const Detail::Node<T>& _node) noexcept;

    template <class T, uint32_t BucketSize, class Allocator>
    [[nodiscard]] Triplet impl_overlap(const std::vector<T, Allocator>* _map, const size_t _N, const Vec2& _min,
                                       const Vec2& _max, const T _h,
                                       const Detail::Bucket<T, BucketSize>& _bucket) noexcept;

  }  // namespace Detail

  //----------------------------------

  /*
      T: the underlying type of the map
      BucketSize: width of a bucket (total bucket size: BucketSize*BucketSize)
      Allocator: allocator of the map. allows for aligned maps
  */
  template <class T, uint32_t BucketSize = 15, class Allocator = std::allocator<T>>
  class MedianQuadTree {
    std::vector<T, Allocator>* map_ = nullptr;

    uint32_t N_;
    uint32_t max_level_;

    std::vector<bool> recomp_map_;

    std::vector<Detail::Bucket<T, BucketSize>> b_;
    std::vector<Detail::Node<T>> n_;

   public:
    constexpr static uint32_t BUCKET_BucketSize = BucketSize;
    using TYPE                                  = T;
    using ALLOCATOR_T                           = Allocator;
    //----------------
    // _n: needs to the width/height of the map. the value needs to be a multiply of BucketSize and a square. this new
    // size can be easily computed using the Detail::get_power_of_2(your_size, bucket_size) function
    MedianQuadTree(std::vector<T, Allocator>* _map, const uint32_t _n);
    //----------------
    MedianQuadTree();
    MedianQuadTree(MedianQuadTree&&)                 = default;
    MedianQuadTree(const MedianQuadTree&)            = delete;
    MedianQuadTree& operator=(MedianQuadTree&&)      = default;
    MedianQuadTree& operator=(const MedianQuadTree&) = delete;
    ~MedianQuadTree()                                = default;
    //----------------
    // recomputes all buckets and nodes, as in the complete tree. This might be slow for larger trees
    void recompute() noexcept;
    // recomputes only the buckets with _m[linear_index_of_bucket] == true. the size of this array is
    // (FixedSize / BucketSize)^2
    void recompute(const std::vector<bool>& _m) noexcept;
    //----------------
    // overlap of [min[0], max[0]) x [min[1], max[1])
    [[nodiscard]] Triplet check_overlap(const Vec2& _min, const Vec2& _max, const T _h) const noexcept;

    // border of [min[0], max[0]) x [min[1], max[1])
    [[nodiscard]] Triplet check_border_overlap(const Vec2& _min, const Vec2& _max, const T _h) const noexcept;
  };  // MedianQuadTree

}  // namespace MQT2

//----------------------------------

template <class T, uint32_t BucketSize, class Allocator>
MQT2::MedianQuadTree<T, BucketSize, Allocator>::MedianQuadTree() {}

template <class T, uint32_t BucketSize, class Allocator>
MQT2::MedianQuadTree<T, BucketSize, Allocator>::MedianQuadTree(std::vector<T, Allocator>* _map, const uint32_t _n)
    : map_(_map), N_(_n) {
  using namespace Detail;

  assert(_n % BucketSize == 0);
  //-----------------
  const uint32_t bc = _n / BucketSize;
  // max_level_ = int32_t( std::round(std::log(bc) / std::log(2)) ) + 1;
  max_level_        = uint32_t(std::round(std::log2(bc))) + 1;

  const uint32_t dc = int32_t((std::pow(4, max_level_ - 1) - 1) / 3);

  n_.resize(dc);
  b_.resize(bc * bc);

  using B_T = VECTOR_INT_TYPE;

  size_t k  = 0;
  for (B_T j = 0; j < bc; j += 2) {
    for (B_T i = 0; i < bc; i += 2) {
      b_[k] = Bucket<T, BucketSize>(i + j * bc, max_level_, Vec2{B_T(i * BucketSize), B_T(j * BucketSize)},
                                    Vec2{B_T(i * BucketSize + BucketSize), B_T(j * BucketSize + BucketSize)});
      k++;
      b_[k] = Bucket<T, BucketSize>((i + 1) + j * bc, max_level_, Vec2{B_T((i + 1) * BucketSize), B_T(j * BucketSize)},
                                    Vec2{B_T((i + 1) * BucketSize + BucketSize), B_T(j * BucketSize + BucketSize)});
      k++;
      b_[k] = Bucket<T, BucketSize>(i + (j + 1) * bc, max_level_, Vec2{B_T(i * BucketSize), B_T((j + 1) * BucketSize)},
                                    Vec2{B_T(i * BucketSize + BucketSize), B_T((j + 1) * BucketSize + BucketSize)});
      k++;
      b_[k] = Bucket<T, BucketSize>(
          (i + 1) + (j + 1) * bc, max_level_, Vec2{B_T((i + 1) * BucketSize), B_T((j + 1) * BucketSize)},
          Vec2{B_T((i + 1) * BucketSize + BucketSize), B_T((j + 1) * BucketSize + BucketSize)});
      k++;
    }
  }

  //---------------
  std::vector<bool> mm;
  mm.resize(bc * bc, true);
  recompute(mm);

}  // MQT2::MedianQuadTree::MedianQuadTree

template <class T, uint32_t BucketSize, class Allocator>
void MQT2::MedianQuadTree<T, BucketSize, Allocator>::recompute(const std::vector<bool>& _m) noexcept {
  impl_recompute(map_, N_, b_, n_, max_level_, 0, 1, n_[0], _m);
}  // MQT2::MedianQuadTree::recompute

template <class T, uint32_t BucketSize, class Allocator>
MQT2::Triplet MQT2::MedianQuadTree<T, BucketSize, Allocator>::check_overlap(const Vec2& _min, const Vec2& _max,
                                                                            const T _h) const noexcept {
  return impl_overlap(map_, N_, b_, n_, max_level_, _min, _max, _h, 0, 1, n_[0]);
  // return impl_overlap_it(_min, _max, _h);
}  // MQT2::MedianQuadTree::check_overlap

template <class T, uint32_t BucketSize, class Allocator>
MQT2::Triplet MQT2::MedianQuadTree<T, BucketSize, Allocator>::check_border_overlap(const Vec2& _min, const Vec2& _max,
                                                                                   const T _h) const noexcept {
  return Detail::naive_border_tester<T>(map_, _min, _max, N_, _h);
}  // MQT2::MedianQuadTree::check_border_overlap

//----------------------------------

template <class T, uint32_t BucketSize, class Allocator>
void MQT2::Detail::impl_recompute(std::vector<T, Allocator>* _map, const size_t _N,
                                  std::vector<Detail::Bucket<T, BucketSize>>& _b, std::vector<Detail::Node<T>>& _n,
                                  const size_t _max_level, const size_t _idx, const uint32_t _level,
                                  Detail::Node<T>& _node, const std::vector<bool>& _m) noexcept {
  using namespace Detail;

  if (_level + 1 == _max_level) {
    constexpr double frac1 = 1. / 3.;
    constexpr double frac2 = 4. / 3.;
    const size_t c1        = size_t(std::round(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2));
    const size_t c2        = c1 + 1;
    const size_t c3        = c2 + 1;
    const size_t c4        = c3 + 1;

    assert(c1 < _b.size() || c2 < _b.size() || c3 < _b.size() || c4 < _b.size());

    impl_recompute(_map, _N, _b[c1], _m);
    impl_recompute(_map, _N, _b[c2], _m);
    impl_recompute(_map, _N, _b[c3], _m);
    impl_recompute(_map, _N, _b[c4], _m);

    _node.min_ = std::min<uint32_t>(
        _b[c1].vals_.front(),
        std::min<uint32_t>(_b[c2].vals_.front(), std::min<uint32_t>(_b[c3].vals_.front(), _b[c4].vals_.front())));
    _node.max_ = std::max<uint32_t>(
        _b[c1].vals_.back(),
        std::max<uint32_t>(_b[c2].vals_.back(), std::max<uint32_t>(_b[c3].vals_.back(), _b[c4].vals_.back())));
    _node.isFlat_  = isEqual(_node.min_, _node.max_);

    _node.bmin_[0] = std::min<uint32_t>(
        _b[c1].bmin_[0], std::min<uint32_t>(_b[c2].bmin_[0], std::min<uint32_t>(_b[c3].bmin_[0], _b[c4].bmin_[0])));
    _node.bmin_[1] = std::min<uint32_t>(
        _b[c1].bmin_[1], std::min<uint32_t>(_b[c2].bmin_[1], std::min<uint32_t>(_b[c3].bmin_[1], _b[c4].bmin_[1])));

    _node.bmax_[0] = std::max<uint32_t>(
        _b[c1].bmax_[0], std::max<uint32_t>(_b[c2].bmax_[0], std::max<uint32_t>(_b[c3].bmax_[0], _b[c4].bmax_[0])));
    _node.bmax_[1] = std::max<uint32_t>(
        _b[c1].bmax_[1], std::max<uint32_t>(_b[c2].bmax_[1], std::max<uint32_t>(_b[c3].bmax_[1], _b[c4].bmax_[1])));

  } else {
    const size_t c1 = 4 * _idx + 1;
    const size_t c2 = c1 + 1;
    const size_t c3 = c2 + 1;
    const size_t c4 = c3 + 1;

    assert(c1 < _n.size() || c2 < _n.size() || c3 < _n.size() || c4 < _n.size());

    impl_recompute(_map, _N, _b, _n, _max_level, c1, _level + 1, _n[c1], _m);
    impl_recompute(_map, _N, _b, _n, _max_level, c2, _level + 1, _n[c2], _m);
    impl_recompute(_map, _N, _b, _n, _max_level, c3, _level + 1, _n[c3], _m);
    impl_recompute(_map, _N, _b, _n, _max_level, c4, _level + 1, _n[c4], _m);

    _node.min_ =
        std::min<uint32_t>(_n[c1].min_, std::min<uint32_t>(_n[c2].min_, std::min<uint32_t>(_n[c3].min_, _n[c4].min_)));
    _node.max_ =
        std::max<uint32_t>(_n[c1].max_, std::max<uint32_t>(_n[c2].max_, std::max<uint32_t>(_n[c3].max_, _n[c4].max_)));
    _node.isFlat_  = isEqual(_node.min_, _node.max_);

    _node.bmin_[0] = std::min<uint32_t>(
        _n[c1].bmin_[0], std::min<uint32_t>(_n[c2].bmin_[0], std::min<uint32_t>(_n[c3].bmin_[0], _n[c4].bmin_[0])));
    _node.bmin_[1] = std::min<uint32_t>(
        _n[c1].bmin_[1], std::min<uint32_t>(_n[c2].bmin_[1], std::min<uint32_t>(_n[c3].bmin_[1], _n[c4].bmin_[1])));

    _node.bmax_[0] = std::max<uint32_t>(
        _n[c1].bmax_[0], std::max<uint32_t>(_n[c2].bmax_[0], std::max<uint32_t>(_n[c3].bmax_[0], _n[c4].bmax_[0])));
    _node.bmax_[1] = std::max<uint32_t>(
        _n[c1].bmax_[1], std::max<uint32_t>(_n[c2].bmax_[1], std::max<uint32_t>(_n[c3].bmax_[1], _n[c4].bmax_[1])));
  }

}  // MQT2::MedianQuadTree::impl_recompute

template <class T, uint32_t BucketSize, class Allocator>
void MQT2::Detail::impl_recompute(std::vector<T, Allocator>* _map, const size_t _N, Detail::Bucket<T, BucketSize>& _b,
                                  const std::vector<bool>& _m) noexcept {
  using namespace Detail;

  if (!_m[_b.idx_]) return;

  size_t k = 0;
  for (uint32_t n0 = _b.bmin_[0]; n0 < _b.bmax_[0]; ++n0) {
    for (uint32_t n1 = _b.bmin_[1]; n1 < _b.bmax_[1]; ++n1) {
      const size_t i = n1 + n0 * _N;
      _b.vals_[k]    = (*_map)[i];
      k++;
    }
  }

  std::sort(_b.vals_.begin(), _b.vals_.end(), [](const auto _e1, const auto _e2) { return _e1 < _e2; });

  _b.isFlat_ = isEqual<T>(_b.vals_.front(), _b.vals_.back());
  if (_b.isFlat_) {
    _b.median_ = Detail::var_type_t<T>(_b.vals_.back());
    return;
  }

  if constexpr ((BucketSize * BucketSize) % 2 == 1) {
    constexpr size_t idx = int32_t(BucketSize * BucketSize) / 2;

    _b.median_           = Detail::var_type_t<T>(_b.vals_[idx]);
    _b.isMonoton_        = BucketSize >= 3 && !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx - 1]), _b.median_) &&
                    !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx + 1]), _b.median_);

  } else {
    constexpr size_t idx1 = size_t((BucketSize * BucketSize) / 2) - 1;
    constexpr size_t idx2 = idx1 + 1;

    _b.median_            = Detail::var_type_t<T>(_b.vals_[idx1] + _b.vals_[idx2]) * Detail::var_type_t<T>(0.5);

    _b.isMonoton_         = BucketSize >= 3 && !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx1]), _b.median_) &&
                    !isEqual<T>(Detail::var_type_t<T>(_b.vals_[idx2]), _b.median_);
  }

}  // MQT2::MedianQuadTree::impl_recompute

template <class T, uint32_t BucketSize, class Allocator>
MQT2::Triplet MQT2::Detail::impl_overlap(const std::vector<T, Allocator>* _map, const size_t _N,
                                         const std::vector<Detail::Bucket<T, BucketSize>>& _b,
                                         const std::vector<Detail::Node<T>>& _n, const size_t _max_level,
                                         const Vec2& _min, const Vec2& _max, const T _h, const size_t _idx,
                                         const uint32_t _level, const Detail::Node<T>& _node) noexcept {
  using namespace Detail;

  if (_max[0] < _node.bmin_[0] || _node.bmax_[0] < _min[0] || _max[1] < _node.bmin_[1] || _node.bmax_[1] < _min[1])
    return {0, 0, 0};

  using B_T      = VECTOR_INT_TYPE;

  const B_T min0 = std::max<B_T>(_min[0], _node.bmin_[0]);
  const B_T min1 = std::max<B_T>(_min[1], _node.bmin_[1]);

  const B_T max0 = std::min<B_T>(_max[0], _node.bmax_[0]);
  const B_T max1 = std::min<B_T>(_max[1], _node.bmax_[1]);

  const bool isPartial =
      !(min0 == _node.bmin_[0] && min1 == _node.bmin_[1] && max0 == _node.bmax_[0] && max1 == _node.bmax_[1]);
  const bool isH = _h > _node.max_;
  const bool isM = isEqual<T>(_node.max_, _h);

  if (_node.isFlat_) {
    if (isPartial) {
      if (isH) return {(max0 - min0) * (max1 - min1), 0, 0};
      if (isM)
        return {0, (max0 - min0) * (max1 - min1), 0};
      else
        return {0, 0, (max0 - min0) * (max1 - min1)};

    } else {
      const uint32_t r = _N / uint32_t(std::round(std::pow(2, _level - 1)));
      if (isH) return {r * r, 0, 0};
      if (isM)
        return {0, r * r, 0};
      else
        return {0, 0, r * r};
    }
  }

  //----------------------------

  if (_level + 1 == _max_level) {
    constexpr double frac1 = 1. / 3.;
    constexpr double frac2 = 4. / 3.;
    const size_t c1        = size_t(std::round(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2));
    const size_t c2        = c1 + 1;
    const size_t c3        = c2 + 1;
    const size_t c4        = c3 + 1;

    assert(c1 < _b.size() || c2 < _b.size() || c3 < _b.size() || c4 < _b.size());

    const auto [l1, m1, h1] = impl_overlap(_map, _N, _min, _max, _h, _b[c1]);
    const auto [l2, m2, h2] = impl_overlap(_map, _N, _min, _max, _h, _b[c2]);
    const auto [l3, m3, h3] = impl_overlap(_map, _N, _min, _max, _h, _b[c3]);
    const auto [l4, m4, h4] = impl_overlap(_map, _N, _min, _max, _h, _b[c4]);

    return {l1 + l2 + l3 + l4, m1 + m2 + m3 + m4, h1 + h2 + h3 + h4};
  } else {
    const size_t c1 = 4 * _idx + 1;
    const size_t c2 = c1 + 1;
    const size_t c3 = c2 + 1;
    const size_t c4 = c3 + 1;

    assert(c1 < _n.size() || c2 < _n.size() || c3 < _n.size() || c4 < _n.size());

    const auto [l1, m1, h1] = impl_overlap(_map, _N, _b, _n, _max_level, _min, _max, _h, c1, _level + 1, _n[c1]);
    const auto [l2, m2, h2] = impl_overlap(_map, _N, _b, _n, _max_level, _min, _max, _h, c2, _level + 1, _n[c2]);
    const auto [l3, m3, h3] = impl_overlap(_map, _N, _b, _n, _max_level, _min, _max, _h, c3, _level + 1, _n[c3]);
    const auto [l4, m4, h4] = impl_overlap(_map, _N, _b, _n, _max_level, _min, _max, _h, c4, _level + 1, _n[c4]);

    return {l1 + l2 + l3 + l4, m1 + m2 + m3 + m4, h1 + h2 + h3 + h4};
  }

}  // MQT2::MedianQuadTree::impl_overlap

template <class T, uint32_t BucketSize, class Allocator>
MQT2::Triplet MQT2::Detail::impl_overlap(const std::vector<T, Allocator>* _map, const size_t _N, const Vec2& _min,
                                         const Vec2& _max, const T _h,
                                         const Detail::Bucket<T, BucketSize>& _b) noexcept {
  using namespace Detail;

  const uint32_t min0  = std::max<uint32_t>(_min[0], _b.bmin_[0]);
  const uint32_t min1  = std::max<uint32_t>(_min[1], _b.bmin_[1]);

  const uint32_t max0  = std::min<uint32_t>(_max[0], _b.bmax_[0]);
  const uint32_t max1  = std::min<uint32_t>(_max[1], _b.bmax_[1]);

  const bool isPartial = !(min0 == _b.bmin_[0] && min1 == _b.bmin_[1] && max0 == _b.bmax_[0] && max1 == _b.bmax_[1]);

  if (_b.isFlat_ && !isPartial) {
    if (Detail::isEqual<T>(_h, _b.vals_.front()))
      return {0, (_b.bmax_[0] - _b.bmin_[0]) * (_b.bmax_[1] - _b.bmin_[1]), 0};
    else if (_h > _b.vals_.front())
      return {(_b.bmax_[0] - _b.bmin_[0]) * (_b.bmax_[1] - _b.bmin_[1]), 0, 0};
    else
      return {0, 0, (_b.bmax_[0] - _b.bmin_[0]) * (_b.bmax_[1] - _b.bmin_[1])};
  }

  //--------------------

  // partial overlap
  if (isPartial) {
    uint32_t l = 0, m = 0, h = 0;
    // n1: x, n0: y
    for (uint32_t n0 = min0; n0 < max0; ++n0) {
      for (uint32_t n1 = min1; n1 < max1; ++n1) {
        const size_t i = n1 + n0 * _N;
        const auto hh  = (*_map)[i];
        if (Detail::isEqual<T>(hh, _h))
          m++;
        else if (hh > _h)
          h++;
        else
          l++;
      }
    }
    return {l, m, h};
  }

  if (!_b.isMonoton_) {
    uint32_t l = 0, m = 0, h = 0;
    for (size_t i = 0; i < BucketSize * BucketSize; ++i) {
      const auto hh = _b.vals_[i];
      if (Detail::isEqual<T>(hh, _h))
        m++;
      else if (hh > _h)
        h++;
      else
        l++;
    }
    return {l, m, h};
  }

  if constexpr (false) {
    uint32_t l2 = 0, m2 = 0, h2 = 0;
    for (uint32_t n0 = min0; n0 < max0; ++n0) {
      for (uint32_t n1 = min1; n1 < max1; ++n1) {
        const size_t i = n1 + n0 * _N;
        const auto hh  = (*_map)[i];
        if (Detail::isEqual<T>(hh, _h))
          m2++;
        else if (hh > _h)
          h2++;
        else
          l2++;
      }
    }
    return {l2, m2, h2};
  }

  if constexpr ((BucketSize * BucketSize) % 2 == 1) {
    constexpr size_t idx = int32_t(BucketSize * BucketSize) / 2;

    if (Detail::var_type_t<T>(_h) > _b.median_) {
      uint32_t l = idx, m = 0, h = 0;
      for (size_t i = idx; i < BucketSize * BucketSize; ++i) {
        const auto hh = _b.vals_[i];
        if (Detail::isEqual<T>(hh, _h))
          m++;
        else if (hh > _h)
          h++;
        else
          l++;
      }
      return {l, m, h};
    } else {
      uint32_t l = 0, m = 0, h = idx;
      for (size_t i = 0; i < idx + 1; ++i) {
        const auto hh = _b.vals_[i];
        if (Detail::isEqual<T>(hh, _h))
          m++;
        else if (hh > _h)
          h++;
        else
          l++;
      }
      return {l, m, h};
    }

  } else {
    constexpr size_t idx = size_t((BucketSize * BucketSize) / 2);
    if (Detail::var_type_t<T>(_h) > _b.median_) {
      uint32_t l = idx, m = 0, h = 0;
      for (size_t i = idx; i < BucketSize * BucketSize; ++i) {
        const auto hh = _b.vals_[i];
        if (Detail::isEqual<T>(hh, _h))
          m++;
        else if (hh > _h)
          h++;
        else
          l++;
      }
      return {l, m, h};
    } else {
      uint32_t l = 0, m = 0, h = idx;
      for (size_t i = 0; i < idx; ++i) {
        const auto hh = _b.vals_[i];
        if (Detail::isEqual<T>(hh, _h))
          m++;
        else if (hh > _h)
          h++;
        else
          l++;
      }
      return {l, m, h};
    }
  }

}  // MQT2::Detail::impl_overlap

//----------------------------------------------

// template <class T, uint32_t BucketSize, class Allocator>
// void MQT2::Detail::impl_print(std::stringstream& _ss, const MedianQuadTree<T, BucketSize, Allocator>& _tree) {
//   std::cout << "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-" << std::endl;
//   std::cout << "Nodes: " << n_.size() << std::endl;
//   std::cout << "Buckets: " << b_.size() << std::endl;
//   impl_print(0, 1, n_[0]);
//   std::cout << "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-" << std::endl;
// }  // MQT2::MedianQuadTree::print_debug

// template <class T, uint32_t BucketSize, class Allocator>
// void MQT2::Detail::impl_print(std::stringstream& _ss, const size_t _idx, const uint32_t _level,
//                               const Detail::Node<T>& _node) {
//   std::stringstream ss;
//   for (uint32_t i = 0; i < _level; ++i) ss << "  ";

//   std::cout << ss.str() << "Level: " << _level << std::endl;
//   std::cout << ss.str() << "bounds: [" << t.bmin_[0] << ", " << t.bmin_[1] << "][" << t.bmax_[0] << ", " <<
//   t.bmax_[1]
//             << "]" << std::endl;
//   std::cout << ss.str() << "Interval: [" << t.min_ << ", " << ", " << t.max_ << "]" << std::endl;
//   std::cout << ss.str() << "Flat: " << (t.isFlat_ ? "yes" : "no") << std::endl;
//   std::cout << ss.str() << "---------------------------------" << std::endl;

//   if (_level + 1 == max_level_) {
//     constexpr double frac1 = 1. / 3.;
//     constexpr double frac2 = 4. / 3.;
//     const size_t c1        = size_t(std::round(4. * double(_idx) - 4. * std::pow(4., _level - 1) * frac1 + frac2));
//     const size_t c2        = c1 + 1;
//     const size_t c3        = c2 + 1;
//     const size_t c4        = c3 + 1;

//     impl_print(b_[c1]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//     impl_print(b_[c2]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//     impl_print(b_[c3]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//     impl_print(b_[c4]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;

//   } else {
//     const size_t c1 = 4 * _idx + 1;
//     const size_t c2 = c1 + 1;
//     const size_t c3 = c2 + 1;
//     const size_t c4 = c3 + 1;

//     impl_print(c1, _level + 1, n_[c1]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//     impl_print(c2, _level + 1, n_[c2]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//     impl_print(c3, _level + 1, n_[c3]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//     impl_print(c4, _level + 1, n_[c4]);
//     std::cout << ss.str() << "+++++++++++++" << std::endl;
//   }
//   std::cout << ss.str() << "---------------------------------" << std::endl;
// }  // MQT2::MedianQuadTree::impl_print

// template <class T, uint32_t BucketSize, class Allocator>
// void MQT2::MedianQuadTree<T, BucketSize, Allocator>::impl_print(std::stringstream& _ss,
//                                                                 const Detail::Bucket<T, BucketSize>& _bucket) {
//   std::stringstream ss;
//   for (uint32_t i = 0; i < t.level_; ++i) ss << "  ";

//   std::cout << ss.str() << "-----------" << std::endl;
//   std::cout << ss.str() << "Bucket [id: " << t.idx_ << "]" << std::endl;
//   std::cout << ss.str() << "bounds: [" << t.bmin_[0] << ", " << t.bmin_[1] << "][" << t.bmax_[0] << ", " <<
//   t.bmax_[1]
//             << "]" << std::endl;
//   std::cout << ss.str() << "Interval: [" << t.vals_.front() << ", " << t.median_ << ", " << t.vals_.back() << "]"
//             << std::endl;
//   std::cout << ss.str() << "Flat: " << (t.isFlat_ ? "yes" : "no") << std::endl;
//   std::cout << ss.str() << "Monoton: " << (t.isMonoton_ ? "yes" : "no") << std::endl;
// }  // MQT2::MedianQuadTree::impl_print
