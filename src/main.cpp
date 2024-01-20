
#include <chrono>
#include <random>

#include <MQT.hpp>

void test_bucket_node() {
    using namespace MQT;

    std::vector<double> map;
    map.resize(20 * 20);
    std::fill(map.begin(), map.end(), 0.);

    Detail::Bucket<double> b(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1, 0);
    b.recompute();

    std::cout << "----- Bucket flat -----" << std::endl;
    //full overlap
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{2, 2}, Vec2{8, 8}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{2, 2}, Vec2{8, 8}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{-5, -5}, Vec2{5, 5}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{5, 5}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{5, -5}, Vec2{15, 5}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, -5}, Vec2{15, 5}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{-5, 5}, Vec2{5, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 5}, Vec2{5, 15}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    for(int32_t n0 = 0; n0 <= 11; ++n0){
        for(int32_t n1 = 0; n1 <= 8; ++n1){
            const int32_t i = n1 + n0 * 20;
            map[i] = 2.;
        }
    }
    b.recompute();

    std::cout << "----- Bucket partial -----" << std::endl;
    //full overlap
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{2, 2}, Vec2{8, 8}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{2, 2}, Vec2{8, 8}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{-5, -5}, Vec2{5, 5}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{5, 5}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{5, -5}, Vec2{15, 5}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, -5}, Vec2{15, 5}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{-5, 5}, Vec2{5, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 5}, Vec2{5, 15}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    for(int32_t n1 = 0; n1 < 20; ++n1){
        for(int32_t n0 = 0; n0 < 20; ++n0){
            const int32_t i = n1 + n0 * 20;
            if(i%2) map[i] = 2.;
        }
    }
    b.recompute();

    std::cout << "----- Bucket checker -----" << std::endl;
    //full overlap
     if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{2, 2}, Vec2{8, 8}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{2, 2}, Vec2{8, 8}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{-5, -5}, Vec2{5, 5}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{5, 5}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{5, -5}, Vec2{15, 5}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, -5}, Vec2{15, 5}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{-5, 5}, Vec2{5, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 5}, Vec2{5, 15}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = b.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
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
        const auto[l1, m1, h1] = node.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 10}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{10, -5}, Vec2{25, 10}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{-5, 10}, Vec2{10, 25}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 25}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 7: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
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
        const auto[l1, m1, h1] = node.overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 20, 1.);
        std::cout << "Test 1: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //mid
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 20, 1.);
        std::cout << "Test 2: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //partial
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 10}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 20, 1.);
        std::cout << "Test 3: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{10, -5}, Vec2{25, 10}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 20, 1.);
        std::cout << "Test 4: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{-5, 10}, Vec2{10, 25}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 5: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 20, 1.);
        std::cout << "Test 6: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{-5, -5}, Vec2{10, 25}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 20, 1.);
        std::cout << "Test 7: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }
    if constexpr(true){
        const auto[l1, m1, h1] = node.overlap(Vec2{0, 5}, Vec2{12, 17}, 1.);
        const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 5}, Vec2{12, 17}, 20, 1.);
        std::cout << "Test 8: ";
        std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
        std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
        std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
    }

    //---------------------------
    std::fill(map.begin(), map.end(), 0.);
    map.resize(40*40);
    for(int32_t n0 = 23; n0 <= 31; ++n0){
        for(int32_t n1 = 20; n1 <= 25; ++n1){
            const int32_t i = n1 + n0 * 40;
            map[i] = 2.;
        }
    }

     {
        MedianQuadTree<double> tree(map, 40, 40, 100, 10);
        //std::cout << tree << std::endl;
        std::cout << "----- Tree single box -----" << std::endl;

        //full overlap
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 40, 1.);
            std::cout << "Test 1: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //mid
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 40, 1.);
            std::cout << "Test 2: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //partial
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{-5, -5}, Vec2{10, 10}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 40, 1.);
            std::cout << "Test 3: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{10, -5}, Vec2{25, 10}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 40, 1.);
            std::cout << "Test 4: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{-5, 10}, Vec2{10, 25}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 40, 1.);
            std::cout << "Test 5: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 40, 1.);
            std::cout << "Test 6: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{-5, -5}, Vec2{10, 25}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 40, 1.);
            std::cout << "Test 7: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{0, 5}, Vec2{12, 17}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 5}, Vec2{12, 17}, 40, 1.);
            std::cout << "Test 8: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
    }

    for(int32_t n1 = 0; n1 < 40; ++n1){
        for(int32_t n0 = 0; n0 < 40; ++n0){
            const int32_t i = n1 + n0 * 40;
            if(i%2) map[i] = 2.;
        }
    }

    {
        MedianQuadTree<double> tree(map, 40, 40, 100, 10);
        //std::cout << tree << std::endl;
        std::cout << "----- Tree checker -----" << std::endl;

        //full overlap
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{0, 0}, Vec2{20, 20}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 0}, Vec2{20, 20}, 40, 1.);
            std::cout << "Test 1: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //mid
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{5, 5}, Vec2{15, 15}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{5, 5}, Vec2{15, 15}, 40, 1.);
            std::cout << "Test 2: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //partial
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{-5, -5}, Vec2{10, 10}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 10}, 40, 1.);
            std::cout << "Test 3: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{10, -5}, Vec2{25, 10}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, -5}, Vec2{25, 10}, 40, 1.);
            std::cout << "Test 4: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{-5, 10}, Vec2{10, 25}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, 10}, Vec2{10, 25}, 40, 1.);
            std::cout << "Test 5: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{10, 10}, Vec2{25, 25}, 40, 1.);
            std::cout << "Test 6: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{-5, -5}, Vec2{10, 25}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{-5, -5}, Vec2{10, 25}, 40, 1.);
            std::cout << "Test 7: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2{0, 5}, Vec2{12, 17}, 1.);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{0, 5}, Vec2{12, 17}, 40, 1.);
            std::cout << "Test 8: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
    }


    //-------------------------------

    map.resize(6400*6400);
    std::fill(map.begin(), map.end(), 0.);

    MedianQuadTree<double>tree1(map, 6400, 6400, 250, 100);

    using Dist = ::std::uniform_int_distribution<>;
    using Rand = std::mt19937_64;
    using DistD = ::std::uniform_real_distribution<double>;

    Rand rnd (1234567890);

    int32_t suc = 0;
    int32_t fail = 0;

    for(int32_t k = 0; k < 100; ++k){

        {
            const int32_t width = Dist(10, 500)(rnd);
            const int32_t height = Dist(10, 500)(rnd);
            const int32_t xmin = Dist(0, 6400 - width)(rnd);
            const int32_t ymin = Dist(0, 6400 - height)(rnd);
            const double h = std::round(DistD(10., 200.)(rnd));

            for(int32_t n0 = ymin; n0 <= ymin + height; ++n0){
                for(int32_t n1 = xmin; n1 <= xmin + width; ++n1){
                    const int32_t i = n1 + n0 * 6400;
                    map[i] = h;
                }
            }
        }

        tree1.recompute();

        //----------------

        
        for(int32_t j = 0; j < 10; ++j){

            const int32_t width = Dist(250, 1250)(rnd);
            const int32_t height = Dist(250, 1250)(rnd);
            const int32_t n0 = Dist(width + 1, 6400 - width - 1)(rnd);
            const int32_t n1 = Dist(height + 1, 6400 - height - 1)(rnd);
            const double h = std::round(DistD(10., 200.)(rnd));

            const auto[l1, m1, h1] = tree1.check_overlap(Vec2{n0 - width, n1 - height}, Vec2{n0 + width, n1 + height}, h);
            const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{n0 - width, n1 - height}, Vec2{n0 + width, n1 + height}, 6400, h);

            if(l1 == l2 && m1 == m2 && h1 == h2) suc++;
            else{
                std::cout << "----" << std::endl;
                std::cout << n0 - width << ", " << n1 - height << std::endl;
                std::cout << n0 + width << ", " << n1 + height << std::endl;
                std::cout << width << ", " << height << std::endl;
                std::cout << h << std::endl;
                std::cout << l1 << ", " << m1 << ", " << h1 << "(" << l1 + m1 + h1 << ")" << std::endl;
                std::cout << l2 << ", " << m2 << ", " << h2 << "(" << l2 + m2 + h2 << ")" << std::endl;
                std::cout << "----" << std::endl;
                fail++;
            } 
        }

        std::cout << "\r" << k;

    }

    std::cout << "\rSuccess: " << suc << std::endl;
    std::cout << "Fail: " << fail << std::endl;
}

void bench_tree() {
    using namespace MQT;

    std::mt19937_64 rand(1234567890);

    std::vector<double> map;
    // map.resize(1000 * 1000);
    // std::fill(map.begin(), map.end(), 0.);

    

    // for(int32_t i = 0; i < 1000; ++i)
    //     map[dist(rand)] = 2.;

    // int32_t t = 0;
    // for(int32_t i = 1; i < 21; ++i){
    //     MedianQuadTree<double> tree(map, 1000, 1000, 100, 5 * i);

    //     const auto start = std::chrono::high_resolution_clock::now();
    //     const auto[l, m, h] = tree.check_overlap(Vec2{250, 200}, Vec2{750, 700}, 1.);
    //     const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;
    //     std::cout << i << " - " << ee.count() << std::endl;
    //     t += l;
    // }
   

    using Dist = ::std::uniform_int_distribution<>;
    using Rand = std::mt19937_64;
    using DistD = ::std::uniform_real_distribution<double>;

    map.resize(7680 * 7680);
    std::uniform_int_distribution<> dist (0, int32_t(map.size() - 1));
    std::fill(map.begin(), map.end(), 0.);
    int32_t t = 0;

    if constexpr(false){
        for(int32_t i = 0; i < 2000 * 7680; ++i)
            map[dist(rand)] = 2.;

    }

    for(int32_t k = 0; k < 100; ++k){

        {
            const int32_t width = Dist(10, 500)(rand);
            const int32_t height = Dist(10, 500)(rand);
            const int32_t xmin = Dist(0, 7680 - width)(rand);
            const int32_t ymin = Dist(0, 7680 - height)(rand);
            const double h = k;

            for(int32_t n0 = ymin; n0 <= ymin + height; ++n0){
                for(int32_t n1 = xmin; n1 <= xmin + width; ++n1){
                    const int32_t i = n1 + n0 * 7680;
                    map[i] = h;
                }
            }
        }

    }

    const double hh = 50.; // std::round(DistD(10., 200.)(rand));

    std::cout << "-----------------" << std::endl;

    {
        std::cout << "tree 1" << std::endl;
        {
            MedianQuadTree<double> tree(map, 7680, 7680, 100, 15);

            for(int32_t i = 1; i < 50; ++i){

                double tmp = 0.;
                for(int32_t j = 0; j < 12; ++j){
                    const auto start = std::chrono::high_resolution_clock::now();
                    const auto[l1, m1, h1] = tree.check_overlap(Vec2{i * 50, i * 50}, Vec2{7680 - i * 50, 7680 - i * 50}, hh);
                    const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;          
                    tmp += ee.count();
                    t += l1;
                }

                std::cout << tmp / 12. << std::endl;

            }
        }

        // std::cout << std::endl;
        // std::cout << "tree 2" << std::endl;
        // {
        //     MedianQuadTree<double> tree(map, 7680, 7680, 100, 30);

        //     for(int32_t i = 1; i < 50; ++i){

        //         double tmp = 0.;
        //         for(int32_t j = 0; j < 12; ++j){
        //             const auto start = std::chrono::high_resolution_clock::now();
        //             const auto[l1, m1, h1] = tree.check_overlap(Vec2{i * 50, i * 50}, Vec2{7680 - i * 50, 7680 - i * 50}, hh);
        //             const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;          
        //             tmp += ee.count();
        //             t += l1;
        //         }

        //         std::cout << tmp / 12. << std::endl;

        //     }
        // }

        // if constexpr(true){
        //     std::cout << std::endl;
        //     std::cout << "naive 1" << std::endl;
        //     for(int32_t i = 1; i < 50; ++i){

        //         double tmp = 0.;
        //         for(int32_t j = 0; j < 12; ++j){
        //             const auto start = std::chrono::high_resolution_clock::now();
        //             const auto[l2, m2, h2] = Detail::naive_tester<double>(map, Vec2{i * 50, i * 50}, Vec2{7680 - i * 50, 7680 - i * 50}, 7680, hh);

        //             const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;
        //             tmp += ee.count();

        //             t += l2;
        //         }
        //         std::cout << tmp / 12. << std::endl;

        //     }
        // }
    }

    std::cout << t << std::endl;

}

int main() {
    //test_bucket_node();
    bench_tree();
    return 0;
}