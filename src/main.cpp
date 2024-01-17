
#include <chrono>
#include <random>

#include <MQT.hpp>

void test_bucket_node() {
    using namespace MQT;

    std::vector<double> map;
    map.resize(10 * 10);
    std::fill(map.begin(), map.end(), 0.);

    Detail::Bucket<double> b(map, Vec2{0, 0}, Vec2{10, 10}, 10, 1, 0);
    b.recompute();

    std::cout << "----- Bucket flat -----" << std::endl;
    //full overlap
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{0, 0}, Vec2{10, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{10, 10}, 10, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{2, 2}, Vec2{8, 8}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{2, 2}, Vec2{8, 8}, 10, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{-5, -5}, Vec2{5, 5}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{5, 5}, 10, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{5, -5}, Vec2{15, 5}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, -5}, Vec2{15, 5}, 10, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{-5, 5}, Vec2{5, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 5}, Vec2{5, 15}, 10, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 10, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    for(int32_t n1 = 0; n1 < 10; ++n1){
        for(int32_t n0 = 0; n0 < 10; ++n0){
            const int32_t i = n1 + n0 * 10;
            if(i%2) map[i] = 2.;
        }
    }
    b.recompute();

    std::cout << "----- Bucket checker -----" << std::endl;
    //full overlap
     if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{0, 0}, Vec2{10, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{10, 10}, 10, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{2, 2}, Vec2{8, 8}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{2, 2}, Vec2{8, 8}, 10, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{-5, -5}, Vec2{5, 5}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{5, 5}, 10, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{5, -5}, Vec2{15, 5}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, -5}, Vec2{15, 5}, 10, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{-5, 5}, Vec2{5, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 5}, Vec2{5, 15}, 10, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = b.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 10, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //---------------------------

    map.resize(20*20);
    std::fill(map.begin(), map.end(), 0.);
    Detail::Node<double> node (Vec2{0, 0}, Vec2{20, 20}, 20, 1, 0);
    node.c_[0] = std::make_unique<Detail::Bucket<double>>(map, Vec2{0, 0}, Vec2{10, 10}, 20, 2, 1);
    node.c_[1] = std::make_unique<Detail::Bucket<double>>(map, Vec2{10, 0}, Vec2{20, 10}, 20, 2, 2);
    node.c_[2] = std::make_unique<Detail::Bucket<double>>(map, Vec2{0, 10}, Vec2{10, 20}, 20, 2, 3);
    node.c_[3] = std::make_unique<Detail::Bucket<double>>(map, Vec2{10, 10}, Vec2{20, 20}, 20, 2, 4);
    node.recompute();

    std::cout << "----- Node flat -----" << std::endl;
    //full overlap
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{10, -5}, Vec2{25, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{-5, 10}, Vec2{10, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 7: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //------------------------------------

    for(int32_t n1 = 0; n1 < 20; ++n1){
        for(int32_t n0 = 0; n0 < 20; ++n0){
            const int32_t i = n1 + n0 * 10;
            if(i%2) map[i] = 2.;
        }
    }

    node.recompute();

    std::cout << "----- Node checker -----" << std::endl;
    //full overlap
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{10, -5}, Vec2{25, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{-5, 10}, Vec2{10, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 7: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = node.overlap(Vec2{0, 5}, Vec2{12, 17}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 5}, Vec2{12, 17}, 20, 1.);
        std::cout << "Test 8: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //---------------------------
    map.resize(40*40);
    for(int32_t n1 = 0; n1 < 40; ++n1){
        for(int32_t n0 = 0; n0 < 40; ++n0){
            const int32_t i = n1 + n0 * 40;
            if(i%2) map[i] = 2.;
        }
    }

    MedianQuadTree<double> tree(map, 40, 40, 100, 10);
    //std::cout << tree << std::endl;
    std::cout << "----- Tree checker -----" << std::endl;

    //full overlap
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 40, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 40, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{-5, -5}, Vec2{10, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 40, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{10, -5}, Vec2{25, 10}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 40, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{-5, 10}, Vec2{10, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 40, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{10, 10}, Vec2{25, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 40, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{-5, -5}, Vec2{10, 25}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 40, 1.);
        std::cout << "Test 7: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, h1] = tree.check(Vec2{0, 5}, Vec2{12, 17}, 1.);
        const auto[l2, h2] = Detail::naive_tester<double>(map, Vec2{0, 5}, Vec2{12, 17}, 40, 1.);
        std::cout << "Test 8: ";
        std::cout << " Res: " << l1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << h2;
        std::cout << (l1 == l2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
}

int main() {
    test_bucket_node();
    return 0;
}