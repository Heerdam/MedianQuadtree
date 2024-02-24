
#include "../include/MQT2.hpp"

namespace MQT2::CUDA {

    namespace Detail {

        template<class T>
        class Buffer {

            int32_t size_;
            T* buffer_;

        public:

            __device__ Buffer() : buffer_(nullptr) {}

            __device__ Buffer(int32_t _size) : size_(_size) {
                assert(_size >= 0);
                cudaMalloc(&buffer_, size() * sizeof(T));
            }

            Buffer(Buffer&&) = default;
            Buffer(const Buffer&) = delete;

            Buffer& operator=(Buffer&&) = default;
            Buffer& operator=(const Buffer&) = delete;

            __device__ ~Buffer() {
                if(buffer_)
                    cudaFree(buffer_);
            }

            T& operator[](int32_t _idx){
                assert(_idx >= 0 && _idx < size());
                return buffer_[_idx];
            }

            const T& operator[](int32_t _idx) const{
                assert(_idx >= 0 && _idx < size());
                return buffer_[_idx];
            }

            int32_t size() const {
                return size_;
            }

            T* operator*() {
                return buffer_;
            }

            void cpy_to_device(void* _b){
                cudaMemcpy(buffer_, _b, size() * sizeof(T), cudaMemcpyHostToDevice);
            }

            void cpy_from_device(void* _b){
                 cudaMemcpy(_b, buffer_, size() * sizeof(T), cudaMemcpyDeviceToHost);
            }

            void cpy_from_device_to_device(void* _b){
                cudaMemcpy(buffer_, _b, size() * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            void cpy_from_device_to_device(Buffer& _o){
                cpy_from_device_to_device((void*)(*_o));
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
        __device__ void impl_node() {

        }

        template<class T, int32_t SIZE>
        __device__ void impl_bucket() {

        }

        template<class T, int32_t SIZE>
        __device__ void impl_overlap(
            const Buffer<MQT2::Detail::Bucket<T, SIZE>>& _b,
            const Buffer<MQT2::Detail::Node<T, SIZE>>& _n,
            const int32_t _b1,
            const int32_t _b2,
            const int32_t _b3
        ){

            Queue<int32_t> qn (_n.size());
            Queue<int32_t> qb (_b.size());

            const int32_t bid = blockIdx.x*blockDim.x + threadIdx.x;

           
            for(int32_t n0 = 0; n0 < 1; ++n0){
                for(int32_t n1 = 0; n1 < 1; ++n1){

                    int32_t l = 0, m = 0, h = 0;

                    while(!qn.empty()){

                    }

                    while(!qb.empty()){

                    }

                }
            }

        }

    }//Detail



    template<class T, int32_t SIZE, class CF>
    __global__ std::pair<int32_t, int32_t> init(
        const std::vector<MQT2::Detail::Bucket<T, SIZE>>& _buckets,
        const std::vector<MQT2::Detail::Node<T, SIZE>>& _nodes,
        const std::vector<int32_t>& _boxes,
        const int32_t _m0, const int32_t _m1,
        CF&& _cf
    ){

        using namespace Detail;

        Buffer<MQT2::Detail::Bucket<T, SIZE>> buckets (_buckets.size());
        buckets.cpy_to_device((void*)_buckets.data());

        Buffer<MQT2::Detail::Node<T, SIZE>> nodes (_nodes.size());
        nodes.cpy_to_device((void*)_nodes.data());

        Buffer<int32_t> boxes (_boxes.size());
        boxes.cpy_to_device((void*)_boxes.data());


        Buffer<int32_t> res (3 * _m0 * _m1);

        //-----------------------



    }//init

}//MQT2::CUDA 
