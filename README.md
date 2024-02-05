# MedianQuadtree
The MQT allows nlog(n) range lookup on 2d heightmaps. A range lookup is basically a convolution with a symmetrical kernel with all values 1.
While a convolution cannot give any information about the topology of the heightmap, the MQT can. It will return the count how many values are at the same
level, lower or higher than the requested height. MQT supports integral and floating point height maps.

## Quick start
Add this project with cmake or simply copy the MQT2.hpp header. This project has zero dependencies.

```
#include <mqt2.hpp>

//in your code
using namespace MQT2;
...
std::vector<int16_t> map;
map.resize(domian_size*domian_size);
...
using Tree = MedianQuadTree<int16_t, 15>;
//the domain size needs to be quadratic and of the size of bucket_size * 2^n
Tree tree(map, domian_size);

//the overlap function takes a 2d aabb with [mind, max) and a desired height
const auto [lower, same, heigher] = tree.check_overlap(
				Vec2{ min_x, min_y },
				Vec2{ max_x, max_y },
				some_height);

```

## Important considerations
The domain size needs to be of the size of bucket_size * 2^n. For smaller or non-quadratic domains simply just use a subset of the domain. 
The bucket size should be chosen in a way that the tree has a depth of 8 to 13.
Performance will rely heavily on the depth and bucket size.

## Benchmark

![image](https://github.com/Heerdam/MedianQuadtree/assets/18620323/555925e4-1391-459a-8045-e6c2a8669aa8)
