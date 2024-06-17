# MedianQuadtree
A MedianQuadTree or MQT for short is a data structure that allows to compute the tri-level convolution for any height map in $O(\sqrt(n))$ and thread-safe manner. The MQT serves as back-end for the TurboPacker library and was part of my master thesis.

## Tri-Level Convolution
For a height map $H(x, y)$, kernel $\kappa \in \mathbf{R}^2$ and height $\eta \in \mathbf{R}$ the tri-level convolution is $\wp:(l, m, h)$<br />
w.r.t.
$$l \ = \ \int^{\kappa} \mathcal{H}\ \ \  \forall \ h \ < \ \eta$$
$$m \ = \ \int^{\kappa} \mathcal{H}\ \ \  \forall \ h \ \equiv \ \eta$$ 
$$h \ = \ \int^{\kappa} \mathcal{H}\ \ \  \forall \ h \ > \ \eta$$


## Quick start
Add this project with cmake or simply copy the MQT2.hpp header. This project has zero dependencies.

```
#include <MQT2.hpp>
using namespace MQT2;

int main() {
    std::vector<uint32_t> map;
    //fill map with values
    //...
    MedianQuadTree<uint32_t, uint16_t, 15> tree(map, 40);

    //overlap
    const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{40, 40}, 1);

    //recompute
    //this bool map containes a bool for every bucket. if true it will recompute this bucket.
    std::vector<bool> mm;
    //do something
    tree.recompute(mm);
}
```

## Important considerations
The domain size needs to be of the size of bucket_size * 2^n. For smaller or non-quadratic domains simply just use a subset of the domain. 
The bucket size should be chosen in a way that the tree has a depth of 8 to 13.
Performance will rely heavily on the depth and bucket size.
