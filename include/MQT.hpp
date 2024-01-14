#pragma once

#include <vector>
#include <array>
#include <variant>
#include <memory>
#include <queue>

namespace MQT {

    using Vec2 = std::array<int32_t, 2>;
    using Vec = std::array<int32_t, 3>;

    template<class T, class ALLOCATOR = std::allocator<T>>
    class MedianQuadTree {

        struct Bucket {
            MedianQuadTree* p_ = nullptr;
            Vec2 bmin_, bmax_;
            //------------------
            const int32_t level_;
            T median_, max_, min_;
            bool isFlat_ = false;
            std::vector<Vec2> l_;
            std::vector<Vec2> h_;
            //----------------
            Bucket(
                MedianQuadTree* _p, 
                const Vec2& _bmin,
                const Vec2& _bmax,
                int32_t _l
            ) : p_(_p), bmin_(_bmin), bmax_(_bmax), level_(_l) {}
            //----------------
            Bucket(Bucket&&) = default;
            Bucket(const Bucket&) = delete;
            Bucket& operator=(Bucket&&) = default;
            Bucket& operator=(const Bucket&) = delete;
            //----------------
            void recompute();
            //----------------
            bool overlap_fast(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            std::pair<int32_t, int32_t> overlap(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            int32_t overlap_border(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
        };//Bucket

        //----------------

        struct Node {
            MedianQuadTree* p_ = nullptr;
            Vec2 bmin_, bmax_;
            //------------------
            const int32_t level_;
            T median_, max_, min_;
            bool isFlat_ = false;
            std::array<std::variant<std::unique_ptr<Node>, std::unique_ptr<Bucket>>, 4> c_;
            std::vector<int32_t> l_;
            std::vector<int32_t> h_;
            //----------------
            Node(
                MedianQuadTree* _p,
                const Vec2& _bmin,
                const Vec2& _bmax,
                int32_t _l
            ) : p_(_p), bmin_(_bmin), bmax_(_bmax), level_(_l) {}
            //----------------
            Node(Node&&) = default;
            Node(const Node&) = delete;
            Node& operator=(Node&&) = default;
            Node& operator=(const Node&) = delete;
            //----------------
            void recompute();
            //----------------
            bool overlap_fast(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            std::pair<int32_t, int32_t> overlap(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
            int32_t overlap_border(const Vec2& _min, const Vec2& _max, T _h) const noexcept;
        };//Node

        //----------------

        std::unique_ptr<Node> root_;

        const std::vector<T, ALLOCATOR>& map_;
        const int32_t n0_;
        const int32_t n1_;

        const int32_t extent_;
        int32_t max_level_;

    public:
        using TYPE = T;
        using ALLOCATOR_T = ALLOCATOR;
        //----------------
        MedianQuadTree(
            const std::vector<T, ALLOCATOR>& _map,
            const int32_t _n0,
            const int32_t _n1,
            const int32_t _min_bucket_size = 12
        );
        //----------------
        MedianQuadTree(MedianQuadTree&&) = default;
        MedianQuadTree(const MedianQuadTree&) = delete;
        MedianQuadTree& operator=(MedianQuadTree&&) = default;
        MedianQuadTree& operator=(const MedianQuadTree&) = delete;
        ~MedianQuadTree() = default;
        //----------------
        void rebuild();
        void recompute();
        //----------------
        [[nodiscard]] bool check_fast(const Vec& _pos, const Vec2& _ext) const noexcept;
        [[nodiscard]] std::pair<int32_t, int32_t> check(const Vec& _pos, const Vec2& _ext) const noexcept;
        [[nodiscard]] int32_t check_border(const Vec& _pos, const Vec2& _ext) const noexcept;
    };//MedianQuadTree

}//MQT

//--------------------------------------------------
//------------------- MedianQuadTree ---------------
//--------------------------------------------------

template<class T, class ALLOCATOR>
void MQT::MedianQuadTree<T, ALLOCATOR>::Bucket::recompute() {
	// n1: x, n0: y
	max_ = -std::numeric_limits<T>::infinity();
	min_ = std::numeric_limits<T>::infinity();

	std::vector<std::pair<T, Vec2>> m;
	m.reserve((bmax_[0] - bmin_[0] + 1) * (bmax_[1] - bmin_[1] + 1));

	for (int32_t n0 = bmin_[1]; n0 < bmax_[1]; ++n0) {
		for (int32_t n1 = bmin_[0]; n1 < bmax_[0]; ++n1) {
			const int32_t i = n1 + n0 * p_->n1_;
			if (i > p_->map.size()) continue;
			m.push_back({ p_->map[i], { n0, n1} });
			max_ = std::max(max_, p_->map[i]);
			min_ = std::min(max_, p_->map[i]);
		}
	}

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
		for (int32_t j = idx + 1; j < m.size(); ++j)
			h_.push_back(std::get<1>(m[j]));

	} else {
		const int32_t idx1 = m.size() / 2;
		const int32_t idx2 = idx1 + 1;
		median_ = (std::get<0>(m[idx1]) + std::get<0>(m[idx2])) * 0.5;

		l_.clear();
		for (int32_t j = 0; j <= idx1; ++j)
			l_.push_back(std::get<1>(m[j]));

		h_.clear();
		for (int32_t j = idx2; j < m.size(); ++j)
			h_.push_back(std::get<1>(m[j]));
	}


}//MQT::MedianQuadTree::Bucket::recompute

template<class T, class ALLOCATOR>
bool MQT::MedianQuadTree<T, ALLOCATOR>::Bucket::overlap_fast(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Bucket::overlap_fast

template<class T, class ALLOCATOR>
std::pair<int32_t, int32_t> MQT::MedianQuadTree<T, ALLOCATOR>::Bucket::overlap(
	const Vec2& _min, 
	const Vec2& _max, 
	T _h
) const noexcept {

	const auto contains = [&](int32_t _x, int32_t _y) {
		return _min[0] <= _x && _x < _max[0] && _min[1] <= _y && _y < _max[1];
	};

    if (isFlat_) {

		if (_h >= median_) return { 0, l_.size() + h_.size() };
		else return { l_.size() + h_.size(), 0 };

	}

    //--------------------

	//partial overlap
	if (_min[0] > bmin_[0] || _max[0] < bmax_[0] || _min[1] > bmin_[1] || _max[1] < bmax_[1]) {

		int32_t low = 0;
		int32_t high = 0;

		for (int32_t n0 = std::max(bmin_[1], _min[1]); n0 < std::min(bmax_[1], _max[1]); ++n0) {
			for (int32_t n1 = std::max(bmin_[0], _min[0]); n1 < std::min(bmax_[0], _max[0]); ++n1) {
				const int32_t i = n0 + n1 * p_->n1_;
				if (i > p_->map.size()) continue;
				if (p_->map[i] >= _h) high++;
				else low++;
			}
		}

		return { low, high };

	}

	//--------------------
	
	if (_h > median_) {

		int32_t low = l_.size();
		int32_t high = 0;

		for (const Vec2 pos : h_) {
			const int32_t i = pos.Y + pos.X * p_->n1_;
			if (!contains(pos.X, pos.Y)) continue;

			if (p_->map[i] >= _h)
				high++;
			if (p_->map[i] < _h)
				low++;
		}

		return { low, high };

	} else {

		int32_t low = 0;
		int32_t high = h_.size();

		for (const Vec2 pos : l_) {
			const int32_t i = pos.Y + pos.X * p_->n1_;
			if (!contains(pos.X, pos.Y)) continue;

			if (p_->map[i] >= _h)
				high++;
			if (p_->map[i] < _h)
				low++;
		}

		return { low, high };

	}

}//MQT::MedianQuadTree::Bucket::overlap

template<class T, class ALLOCATOR>
int32_t MQT::MedianQuadTree<T, ALLOCATOR>::Bucket::overlap_border(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Bucket::overlap_border

//--------------

template<class T, class ALLOCATOR>
void MQT::MedianQuadTree<T, ALLOCATOR>::Node::recompute() {

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

	std::vector<std::pair<int32_t, T>> m;

	switch (c_[0].index()) {
		case 0:
		{			
			for (int32_t i = 0; i < 4; ++i)
				m.push_back({ i, std::get<0>(c_[i])->median });
		}
		break;
		case 1:
		{
			for (int32_t i = 0; i < 4; ++i)
				m.push_back({ i, std::get<1>(c_[i])->median });
		}
		break;
	}

	std::sort(m.begin(), m.end(), [](const auto& _e1, const auto& _e2) {
		return std::get<1>(_e1) < std::get<1>(_e2);
	});

	median_ = (std::get<1>(m[1]) + std::get<1>(m[2])) * 0.5;
	min_ = std::get<1>(m[0]);
	max_ = std::get<1>(m[3]);

	l_[0] = std::get<0>(m[0]);
	l_[1] = std::get<0>(m[1]);

	h_[0] = std::get<0>(m[2]);
	h_[1] = std::get<0>(m[3]);

}//MQT::MedianQuadTree::Node::recompute

template<class T, class ALLOCATOR>
bool MQT::MedianQuadTree<T, ALLOCATOR>::Node::overlap_fast(
    const Vec2& _min, 
    const Vec2& _max, 
    T _h
) const noexcept {

}//MQT::MedianQuadTree::Node::overlap_fast

template<class T, class ALLOCATOR>
std::pair<int32_t, int32_t> MQT::MedianQuadTree<T, ALLOCATOR>::Node::overlap(
	const Vec2& _min,
	const Vec2& _max,
	T _h
) const noexcept {

	//partial overlap
	if (_min[0] > bmin_[0] || _max[0] < bmax_[0] || _min[1] > bmin_[1] || _max[1] < bmax_[1]) {
		int32_t low = 0;
		int32_t high = 0;
		switch (c_[0].index()) {
			case 0:
			{
				for (int32_t i = 0; i < 4; ++i) {
					const auto r = std::get<0>(c_[i])->overlap(_min, _max, _h);
					low += std::get<0>(r);
					high += std::get<1>(r);
				}
			}
			break;
			case 1:
			{
				for (int32_t i = 0; i < 4; ++i) {
					const auto r = std::get<1>(c_[i])->overlap(_min, _max, _h);
					low += std::get<0>(r);
					high += std::get<1>(r);
					//std::cout << std::get<0>(r) << ", " << std::get<1>(r) << std::endl;
				}
			}
			break;
		}
		return { low, high };
	}

	//--------------------

	if (isFlat_) {

		if (_h >= median_) return { 0, 4 * (p_->maxLevel - level_) * p_->extend * p_->extend };
		else return { 4 * (p_->maxLevel - level_) * p_->extend * p_->extend, 0 };

	}

	//--------------------
	
	if (_h > median_) {

		int32_t low = l_.size();
		int32_t high = 0;

		switch (c_[0].index()) {
			case 0:
			{
				const auto r1 = std::get<0>(c_[h_[0]])->overlap(_min, _max, _h);
				const auto r2 = std::get<0>(c_[h_[1]])->overlap(_min, _max, _h);
				low += std::get<0>(r1) + std::get<0>(r2);
				high += std::get<1>(r1) + std::get<1>(r2);
			}
			break;
			case 1:
			{
				const auto r1 = std::get<1>(c_[h_[0]])->overlap(_min, _max, _h);
				const auto r2 = std::get<1>(c_[h_[1]])->overlap(_min, _max, _h);
				low += std::get<0>(r1) + std::get<0>(r2);
				high += std::get<1>(r1) + std::get<1>(r2);
			}
			break;
		}
		
		return { low, high };

	} else {

		int32_t low = 0;
		int32_t high = h_.size();

		switch (c_[0].index()) {
			case 0:
			{
				const auto r1 = std::get<0>(c_[l_[0]])->overlap(_min, _max, _h);
				const auto r2 = std::get<0>(c_[l_[1]])->overlap(_min, _max, _h);
				low += std::get<0>(r1) + std::get<0>(r2);
				high += std::get<1>(r1) + std::get<1>(r2);
			}
			break;
			case 1:
			{
				const auto r1 = std::get<1>(c_[l_[0]])->overlap(_min, _max, _h);
				const auto r2 = std::get<1>(c_[l_[1]])->overlap(_min, _max, _h);
				low += std::get<0>(r1) + std::get<0>(r2);
				high += std::get<1>(r1) + std::get<1>(r2);
			}
			break;
		}

		return { low, high };

	}

}//MQT::MedianQuadTree::Node::overlap

template<class T, class ALLOCATOR>
int32_t MQT::MedianQuadTree<T, ALLOCATOR>::Node::overlap_border(
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
	const int32_t _min_bucket_size
) : map_(_map), n0_(_n0), n1_(_n1), extent_(_min_bucket_size) {

	const int32_t s = std::max(_n0, _n1);
	max_level_ = 1;
	int32_t t = _min_bucket_size;
	while (t < s) {
		max_level_++;
		t *= 2;
	}
	//--------------
	root_ = std::make_unique<Node>(this, { 0, 0 }, { t, t }, 1);

	std::queue<Node*> q;
	q.push(root_.get());

	int32_t nc = 0;
	int32_t bc = 0;
	while (!q.empty()) {

		Node* n = q.front();
		q.pop();

		nc++;

		/*
		children:
			0 | 1
			2 | 3
		*/
		const Vec2 min_0 = { n->bmin.X, n->bmin.Y + (n->bmax.Y - n->bmin.Y) / 2 };
		const Vec2 max_0 = { n->bmin.X + (n->bmax.X - n->bmin.X) / 2, n->bmax.Y };

		const Vec2 min_1 = { n->bmin.X + (n->bmax.X - n->bmin.X) / 2, n->bmin.Y + (n->bmax.Y - n->bmin.Y) / 2 };
		const Vec2 max_1 = n->bmax;

		const Vec2 min_2 = n->bmin;
		const Vec2 max_2 = { n->bmin.X + (n->bmax.X - n->bmin.X) / 2, n->bmin.Y + (n->bmax.Y - n->bmin.Y) / 2 };

		const Vec2 min_3 = { n->bmin.X + (n->bmax.X - n->bmin.X) / 2, n->bmin.Y };
		const Vec2 max_3 = { n->bmax.X, n->bmin.Y + (n->bmax.Y - n->bmin.Y) / 2 };

		if (n->level == max_level_ - 1) {

			bc += 4;

			// --- 0 ---
			n->c[0] = std::make_unique<Bucket>(this, min_0, max_0, n->level + 1);
			
			// --- 1 ---
			n->c[1] = std::make_unique<Bucket>(this, min_1, max_1, n->level + 1);

			// --- 2 ---
			n->c[2] = std::make_unique<Bucket>(this, min_2, max_2, n->level + 1);

			// --- 3 ---
			n->c[3] = std::make_unique<Bucket>(this, min_3, max_3, n->level + 1);

		} else {

			n->c[0] = std::make_unique<Node>(this, min_0, max_0, n->level + 1);
			q.push(std::get<0>(n->c[0]).get());

			n->c[1] = std::make_unique<Node>(this, min_1, max_1, n->level + 1);
			q.push(std::get<0>(n->c[1]).get());

			n->c[2] = std::make_unique<Node>(this, min_2, max_2, n->level + 1);
			q.push(std::get<0>(n->c[2]).get());

			n->c[3] = std::make_unique<Node>(this, min_3, max_3, n->level + 1);
			q.push(std::get<0>(n->c[3]).get());

		}
	}

}//MQT::MedianQuadTree::MedianQuadTree

template<class T, class ALLOCATOR>
void MQT::MedianQuadTree<T, ALLOCATOR>::recompute() {
	root_->recompute();
}//MQT::MedianQuadTree::recompute

template<class T, class ALLOCATOR>
std::pair<int32_t, int32_t> MQT::MedianQuadTree<T, ALLOCATOR>::check(
	const Vec& _pos, 
	const Vec2& _ext
) const noexcept {
	return root_->overlap(
        { _pos[0] - _ext[0], _pos[1] - _ext[1] },
		{ _pos[0] + _ext[0], _pos[1] + _ext[1] },
		_pos[2]);
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
