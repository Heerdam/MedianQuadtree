
#include "../include/MQT2.hpp"

namespace MQT2::CUDA {

    namespace Detail {

        template<class T>
        __forceinline__ __device__ bool isEqual(const T _v1, const T _v2, const T _tol = 1.e-8) {
            if constexpr (std::is_integral_v<T>)
                return _v1 == _v2;
            else return abs( _v1 - _v2 ) <= _tol;
        }//isEqual

        //--------------------------

        template<class T>
        class BufferView {
            int32_t size_;
            T* buffer_;
        public:

            __global__ BufferView() = delete;
            __global__ BufferView(T* _ptr, int32_t _s) : size_(_s), buffer_(_ptr) {}

            __device__ T& operator[](int32_t _idx){
                assert(_idx >= 0 && _idx < size());
                return buffer_[_idx];
            }

            __device__ const T& operator[](int32_t _idx) const{
                assert(_idx >= 0 && _idx < size());
                return buffer_[_idx];
            }

            __global__ int32_t size() const {
                return size_;
            }

            __device__ T* operator*() {
                return buffer_;
            }

            __global__ void init_with_zero() {
                cudaMemset(buffer_, 0, size() * sizeof(T));
            }

            __device__ void cpy_from_device_to_device(BufferView<T>& _o){
                cpy_from_device_to_device((void*)(*_o));
            }

        };

        template<class T>
        class Buffer {

            int32_t size_;
            T* buffer_;

        public:

            __global__ Buffer() : buffer_(nullptr) {}

            __global__ Buffer(int32_t _size) : size_(_size) {
                assert(_size >= 0);
                cudaMalloc(&buffer_, size() * sizeof(T));
            }

            __global__ Buffer(Buffer&&) = default;
            __global__ Buffer(const Buffer&) = delete;

            __global__ Buffer& operator=(Buffer&&) = default;
            __global__ Buffer& operator=(const Buffer&) = delete;

            __global__ ~Buffer() {
                if(buffer_)
                    cudaFree(buffer_);
            }

            __global__ int32_t size() const {
                return size_;
            }

            __device__ T& operator[](int32_t _idx){
                assert(_idx >= 0 && _idx < size());
                return buffer_[_idx];
            }

            __device__ const T& operator[](int32_t _idx) const{
                assert(_idx >= 0 && _idx < size());
                return buffer_[_idx];
            }
         
            __device__ T* operator*() {
                return buffer_;
            }

            __global__ void cpy_to_device(void* _b){
                cudaMemcpy(buffer_, _b, size() * sizeof(T), cudaMemcpyHostToDevice);
            }

            __global__ void cpy_from_device(void* _b){
                 cudaMemcpy(_b, buffer_, size() * sizeof(T), cudaMemcpyDeviceToHost);
            }

            __global__ void init_with_zero() {
                cudaMemset(buffer_, 0, size() * sizeof(T));
            }

            __global__ BufferView<T> to_view() const {
                return BufferView<T>(buffer_, size_);
            }

        };//Buffer

        //---------------------------

        template<class T>
        struct Queue {
            Buffer<T> buffer_;
            int32_t head_, tail_;

            __device__ Queue(int32_t _size) :  head_(0), tail_(0) {
                buffer_ = Buffer<T>(_size);
            }

            __device__ void push(T element) {
                const int32_t next = (tail_ + 1) % buffer_.size();
                if (next != head_) {
                    buffer_[tail_] = element;
                    tail_ = next;
                }
            }

            __device__ T front() const {
                return buffer_[head_];
            }

            __device__ void pop() {
                if (head_ != tail_)
                    head_ = (head_ + 1) % buffer_.size();
            }

            __device__ bool empty() const {
                return head_ == tail_;
            }

            __device__ bool full() const {
                return ((tail_ + 1) % buffer_.size()) == head_;
            }

        };//Queue

        //-----------------------------------------

        template<class T, int32_t SIZE>
        __device__ inline std::tuple<int32_t, int32_t, int32_t> impl_node(
            Queue<std::pair<int32_t, int32_t>>& qn,
            Queue<int32_t>& qb,
            const MQT2::Detail::Node<T, SIZE>& _n,
            const Vec2& _min,
            const Vec2& _max,
            const T _h,
            const int32_t _idx, 
            const int32_t _level
        ) {

            if(_max[0] < _n.bmin_[0] || _n.bmax_[0] < _min[0] || _max[1] < _n.bmin_[1] || _n.bmax_[1] < _min[1]) return { 0, 0, 0 };

            const int32_t min0 = max(_min[0], _n.bmin_[0]);
            const int32_t min1 = max(_min[1], _n.bmin_[1]);

            const int32_t max0 = min(_max[0], _n.bmax_[0]);
            const int32_t max1 = min(_max[1], _n.bmax_[1]);

            const bool isPartial = !(min0 == _n.bmin_[0] && min1 == _n.bmin_[1] && max0 == _n.bmax_[0] && max1 == _n.bmax_[1]);
            const bool isH = _h > _n.max_;
            const bool isM = isEqual<T>(_n.max_, _h);

            if (_n.isFlat_) {

                if(isPartial){

                    if (isH) return { (max0 - min0) * (max1 - min1), 0, 0 };
                    if (isM) return { 0, (max0 - min0) * (max1 - min1), 0 };
                    else return { 0, 0, (max0 - min0) * (max1 - min1) };

                } else {
                    const int32_t r = N_ / int32_t(round(pow(2, _level - 1)));
                    if(isH) return { r*r, 0, 0};
                    if (isM) return { 0, r*r, 0};
                    else return { 0, 0, r*r};
                }

            }

            //----------------------------

            if(_level + 1 == max_level_){
                constexpr T frac1 = 1./3.;
                constexpr T frac2 = 4./3.;
                const int32_t c1 = int32_t(round(4. * T(_idx) - 4. * pow(4., _level - 1) * frac1 + frac2));
                const int32_t c2 = c1 + 1;
                const int32_t c3 = c2 + 1;
                const int32_t c4 = c3 + 1;

                assert(c1 < b_.size() || c2 < b_.size() || c3 < b_.size() || c4 < b_.size());

                qb.push(c1);
                qb.push(c2);
                qb.push(c3);
                qb.push(c4);

                return { 0, 0, 0 };

            } else {
                const int32_t c1 = 4 * _idx + 1;
                const int32_t c2 = c1 + 1;
                const int32_t c3 = c2 + 1;
                const int32_t c4 = c3 + 1;

                assert(c1 < n_.size() || c2 < n_.size() || c3 < n_.size() || c4 < n_.size());

                qn.push({ c1, _level + 1 });
                qn.push({ c2, _level + 1 });
                qn.push({ c3, _level + 1 });
                qn.push({ c4, _level + 1 });

                return { 0, 0, 0 };

            }
        }//impl_node

        template<class T, int32_t SIZE>
        __device__ inline std::tuple<int32_t, int32_t, int32_t> impl_bucket(
            const MQT2::Detail::Bucket<T, SIZE>& _b,
            const Vec2& _min,
            const Vec2& _max,
            const T _h
        ) {

            const int32_t min0 = max(_min[0], _b.bmin_[0]);
            const int32_t min1 = max(_min[1], _b.bmin_[1]);

            const int32_t max0 = min(_max[0], _b.bmax_[0]);
            const int32_t max1 = min(_max[1], _b.bmax_[1]);

            const bool isPartial = !(min0 == _b.bmin_[0] && min1 == _b.bmin_[1] && max0 == _b.bmax_[0] && max1 == _b.bmax_[1]);

            if ( _b.isFlat_ && !isPartial) {
                if(isEqual<T>(_h, _b.vals_.front())) return { 0, ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]), 0 };   
                else if (_h > _b.vals_.front()) return { ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]), 0, 0 };
                else return { 0, 0, ( _b.bmax_[0] -  _b.bmin_[0]) * ( _b.bmax_[1] -  _b.bmin_[1]) };
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
                        if(isEqual<T>(hh, _h)) m++;
                        else if(hh > _h) h++;
                        else l++;    
                    }
                }
                return { l, m, h };
            }

            if(!_b.isMonoton_){
                int32_t l = 0, m = 0, h = 0;
                for(int32_t i = 0; i < SIZE * SIZE; ++i){
                    const auto hh = _b.vals_[i];
                    if(isEqual<T>(hh, _h)) m++;
                    else if(hh > _h) h++;
                    else l++;  
                }
                return { l, m, h };
            }

            if constexpr (false){
                int32_t l2 = 0, m2 = 0, h2 = 0;
                for (int32_t n0 = min0; n0 < max0; ++n0) {
                    for (int32_t n1 = min1; n1 < max1; ++n1) {
                        const int32_t i = n1 + n0 * N_;
                        const auto hh = map_[i];
                        if(isEqual<T>(hh, _h)) m2++;
                        else if(hh > _h) h2++;
                        else l2++;    
                    }
                }
                return {l2, m2, h2};
        }

            if constexpr((SIZE * SIZE) % 2 == 1){
                constexpr int32_t idx = int32_t(SIZE * SIZE) / 2;
                
                if (MQT2::Detail::var_type_t<T>(_h) > _b.median_) {
                    int32_t l = idx, m = 0, h = 0;
                    for(int32_t i = idx; i < SIZE * SIZE; ++i){
                        const auto hh = _b.vals_[i];
                        if(isEqual<T>(hh, _h)) m++;
                        else if(hh > _h) h++;
                        else l++;  
                    }
                    return { l, m, h };
                } else {
                    int32_t l = 0, m = 0, h = idx;
                    for(int32_t i = 0; i < idx + 1; ++i){
                        const auto hh = _b.vals_[i];
                        if(isEqual<T>(hh, _h)) m++;
                        else if(hh > _h) h++;
                        else l++;  
                    }
                    return { l, m, h };
                }

            } else {
                constexpr int32_t idx = int32_t((SIZE * SIZE) / 2);
                if (MQT2::Detail::var_type_t<T>(_h) > _b.median_) {
                    int32_t l = idx, m = 0, h = 0;
                    for(int32_t i = idx; i < SIZE * SIZE; ++i){
                        const auto hh = _b.vals_[i];
                        if(isEqual<T>(hh, _h)) m++;
                        else if(hh > _h) h++;
                        else l++;  
                    }
                    return { l, m, h };
                } else {
                    int32_t l = 0, m = 0, h = idx;
                    for(int32_t i = 0; i < idx; ++i){
                        const auto hh = _b.vals_[i];
                        if(isEqual<T>(hh, _h)) m++;
                        else if(hh > _h) h++;
                        else l++;  
                    }
                    return { l, m, h };
                }
            }
        }//impl_bucket

        template<class T, int32_t SIZE, class CF>
        __device__ void impl_overlap(
            BufferView<MQT2::Detail::Bucket<T, SIZE>> _b,
            BufferView<MQT2::Detail::Node<T, SIZE>> _n,
            BufferView<int32_t> _boxes,
            BufferView<T> _res, 
            CF _cf,
            const Vec2 _min,
            const Vec2 _max,
            const T _h
        ){

            Queue<std::pair<int32_t, int32_t>> qn (_n.size());
            Queue<int32_t> qb (_b.size());

            const int32_t bid = blockIdx.x*blockDim.x + threadIdx.x;

           
            for(int32_t n0 = _min.y_; n0 < 1; ++n0){
                for(int32_t n1 = 0; n1 < 1; ++n1){

                    int32_t l = 0, m = 0, h = 0;

                    while(!qn.empty()){

                        const auto[idx, lvl] n = qn.front();
                        qn.pop();

                        const auto[l1, m1, h1] = impl_node(qn, qb, _n[idx], _min, _max, _h, idx, lvl);
                        l += l1;
                        m += m1;
                        h += h1;
                    }

                    while(!qb.empty()){

                        const int32_t idx = qb.front();
                        qb.pop();

                        const auto[l1, m1, h1] = impl_bucket(_b[idx], _min, _max, _h);
                        l += l1;
                        m += m1;
                        h += h1;
                    }

                    //border
                    int32_t bl, bm, bh;

                    const T cost = _cf(l, m, h, bl, bm, bh);

                    _res[bid] = cost;

                }
            }

        }

    }//Detail

    template<class T, int32_t SIZE, class CF>
    __global__ std::pair<int32_t, int32_t> init(
        const std::vector<MQT2::Detail::Bucket<T, SIZE>>& _buckets,
        const std::vector<MQT2::Detail::Node<T, SIZE>>& _nodes,
        const int32_t _m0, const int32_t _m1,
        const std::vector<int32_t>& _boxes,
        CF _cf
    ){

        using namespace Detail;

        Buffer<MQT2::Detail::Bucket<T, SIZE>> buckets (_buckets.size());
        buckets.cpy_to_device((void*)_buckets.data());

        Buffer<MQT2::Detail::Node<T, SIZE>> nodes (_nodes.size());
        nodes.cpy_to_device((void*)_nodes.data());

        Buffer<int32_t> boxes (_boxes.size());
        boxes.cpy_to_device((void*)_boxes.data());

        Buffer<T> res (_boxes.size());
        res.init_with_zero();

        //1024 threads per box
        for(size_t i = 0; i < _boxes.size() / 3; ++i){
            impl_overlap<<<32, 32, 1>>>(buckets.to_view(), nodes.to_view(), _boxes.to_view(), res.to_view(), _cf);
        }

        cudaDeviceSynchronize();

        std::vector<T> rd (res.size());
        res.cpy_from_device(rd.data());

        int32_t idx = 0;
        int32_t perm = 0;
        T min = std::numerical_limits<T>::infinity();
        for(size_t i = 0; i < res.size(); ++i){

        }
        
        return { idx, perm };

    }//init

}//MQT2::CUDA 
