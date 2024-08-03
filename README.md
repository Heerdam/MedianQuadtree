# MedianQuadtree
A MedianQuadTree or MQT for short is a data structure that allows to compute the tri-level convolution for any height map in $O(\sqrt(n))$ and thread-safe manner. The MQT serves as back-end for the TurboPacker library and was part of my master thesis.<br />
For a exhaustive discussion of the MQT I refere to [here](https://github.com/Heerdam/Master-Thesis/blob/80decb188484df99bec2bb260a8a0fc62432bbd0/Heerdam_master_thesis_v1.1.pdf).

## Tri-Level Convolution
Let $H(x, y)$ represent the values from the heightmap H, D the domain, $(\hat{x}, \hat{y})$ the coordinates withing the support region $\text{supp}(K)\_{(x,y)}$, $K \in \mathbb{R}^2$ the kernel and $h \in \mathbb{R}$ the given height. Then the tri-level convolution $\{l, m, h\}$ is defined as<br />
$$l = \{ H(\hat{x}, \hat{y}) \mid (\hat{x}, \hat{y}) \in \text{supp}(K)\_{(x,y)}, \forall (x, y) \in D, \text{ and } H(\hat{x}, \hat{y}) < h \}$$
$$m = \{ H(\hat{x}, \hat{y}) \mid (\hat{x}, \hat{y}) \in \text{supp}(K)\_{(x,y)}, \forall (x, y) \in D, \text{ and } H(\hat{x}, \hat{y}) \equiv h \}$$
$$h = \{ H(\hat{x}, \hat{y}) \mid \hat{x}, \hat{y}) \in \text{supp}(K)\_{(x,y)}, \forall (x, y) \in D, \text{ and } H(\hat{x}, \hat{y}) > h \} $$


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
