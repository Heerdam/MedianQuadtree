
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>


#include <MQT2.hpp>

void MQT2_tester_float() {

    using namespace MQT2;

    std::vector<double> map;
    map.resize(40*40);
    std::fill(map.begin(), map.end(), 0.);   
    for(uint32_t n0 = 23; n0 <= 31; ++n0){
        for(uint32_t n1 = 20; n1 <= 25; ++n1){
            const uint32_t i = n1 + n0 * 40;
            map[i] = 2.;
        }
    }

    // for(uint32_t n0 = 0; n0 < 40; ++n0){
    //     for(uint32_t n1 = 0; n1 < 40; ++n1){
    //         const uint32_t i = n1 + n0 * 40;
    //         std::cout << (map[i] > 1. ? 'x' : '0' ) << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    

    {
        using BT = uint16_t;
        MedianQuadTree<double, BT, 10> tree(map, 40);
        //std::cout << tree << std::endl;
        std::cout << "----- Tree single box (double) -----" << std::endl;

        //full overlap
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{40, 40}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{40, 40}, 40, 1.);
            std::cout << "Test 1: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //mid
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{5, 5}, Vec2<BT>{35, 35}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{5, 5}, Vec2<BT>{35, 35}, 40, 1.);
            std::cout << "Test 2: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;

            // for(uint32_t n0 = 0; n0 < 40; ++n0){
            //     for(uint32_t n1 = 0; n1 < 40; ++n1){
            //         const uint32_t i = n1 + n0 * 40;
            //         if(n0 >= 5 && n0 < 35 && n1 >= 5 && n1 < 35)
            //             std::cout << 'x' << ' ';
            //         else
            //             std::cout << '0' << ' ';
            //     }
            //     std::cout << std::endl;
            // }
        }

        //return;

        //partial
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 40, 1.);
            std::cout << "Test 3: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 40, 1.);
            std::cout << "Test 4: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 40, 1.);
            std::cout << "Test 5: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;        
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 40, 1.);
            std::cout << "Test 6: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 40, 1.);
            std::cout << "Test 7: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 40, 1.);
            std::cout << "Test 8: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
    }

    for(uint32_t n1 = 0; n1 < 40; ++n1){
        for(uint32_t n0 = 0; n0 < 40; ++n0){
            const uint32_t i = n1 + n0 * 40;
            if(i%2) map[i] = 2.;
        }
    }

    {
        using BT = uint16_t;
        MedianQuadTree<double, BT, 10> tree(map, 40);
        //std::cout << tree << std::endl;
        std::cout << "----- Tree checker (double) -----" << std::endl;

        //full overlap
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{20, 20}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{20, 20}, 40, 1.);
            std::cout << "Test 1: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //mid
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{5, 5}, Vec2<BT>{15, 15}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{5, 5}, Vec2<BT>{15, 15}, 40, 1.);
            std::cout << "Test 2: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //partial
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 40, 1.);
            std::cout << "Test 3: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 40, 1.);
            std::cout << "Test 4: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 40, 1.);
            std::cout << "Test 5: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 40, 1.);
            std::cout << "Test 6: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 40, 1.);
            std::cout << "Test 7: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 40, 1.);
            std::cout << "Test 8: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
    }

    //-------------------------------

    {
        using BT = uint32_t;
        constexpr uint32_t size = 10*2*2;
        map.resize(size*size);
        std::fill(map.begin(), map.end(), 0.);

        MedianQuadTree<double, BT, 10>tree1(map, size);

        std::cout << "Tree size 10 (double)" << std::endl;

        using Dist = ::std::uniform_int_distribution<>;
        using Rand = std::mt19937_64;
        using DistD = ::std::uniform_real_distribution<double>;

        std::random_device rd;
        const auto seed = 2937685668; // rd();
        std::cout << seed << std::endl;
        std::mt19937_64 rnd(seed);

        uint32_t suc = 0;
        uint32_t fail = 0;

        const uint32_t bc = size / 10;
        std::vector<bool> mm;
        mm.resize(bc * bc);
        std::fill(mm.begin(), mm.end(), true);

        for(uint32_t k = 0; k < 100; ++k){

            {
                const uint32_t width = Dist(2, size/4)(rnd) - 1;
                const uint32_t height = Dist(2, size/4)(rnd) - 1;
                const uint32_t xmin = Dist(0, size - width - 1)(rnd);
                const uint32_t ymin = Dist(0, size - height - 1)(rnd);
                const double h = std::round(DistD(10., 200.)(rnd));

                for(uint32_t n0 = ymin; n0 <= ymin + height; ++n0){
                    for(uint32_t n1 = xmin; n1 <= xmin + width; ++n1){
                        const uint32_t i = n1 + n0 * size;
                        assert(i >= 0 && i < map.size());
                        map[i] = k;
                    }
                }

                std::fill(mm.begin(), mm.end(), false);
                for (uint32_t n1 = xmin / 10; n1 <= std::min((xmin + width) / 10 + 1, bc - 1); ++n1) {
                    for (uint32_t n0 = ymin / 10; n0 <= std::min((ymin + height) / 10 + 1, bc - 1); ++n0) {         
                        const uint32_t iid = n0 + n1 * bc;
                        mm[iid] = true;
                    }
                }

            }

            tree1.recompute(mm);

            {

                const BT width = Dist(size/4, size/2 - 2)(rnd) - 1;
                const BT height = Dist(size/4, size/2 - 2)(rnd) - 1;
                const BT n0 = Dist(width + 1, size - width - 1)(rnd);
                const BT n1 = Dist(height + 1, size - height - 1)(rnd);
                const double h = std::round(DistD(10., 200.)(rnd));
                
                const auto[l1, m1, h1] = tree1.check_overlap(Vec2<BT>{BT(n0 - width), BT(n1 - height)}, Vec2<BT>{BT(n0 + width), BT(n1 + height)}, h);
                const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double, BT>(map, Vec2<BT>{BT(n0 - width), BT(n1 - height)}, Vec2<BT>{BT(n0 + width), BT(n1 + height)}, size, h);

                if(l1 == l2 && m1 == m2 && h1 == h2) suc++;
                else{
                    // std::cout << std::endl;
                    // for(uint32_t nn0 = 0; nn0 < 40; ++nn0){
                    //     for(uint32_t nn1 = 0; nn1 < 40; ++nn1){
                    //         const uint32_t i = nn1 + nn0 * 40;
                    //         if(map[i] > h)
                    //             std::cout << 'x' << ' ';
                    //         else
                    //             std::cout << '0' << ' ';
                    //     }
                    //     std::cout << std::endl;
                    // }
                    // std::cout << std::endl;
                    // for(uint32_t nn0 = 0; nn0 < 40; ++nn0){
                    //     for(uint32_t nn1 = 0; nn1 < 40; ++nn1){
                    //         const uint32_t i = nn1 + nn0 * 40;
                    //         if(nn0 >= (n0 - width) && nn0 < (n0 + width) && nn1 >= (n1 - height) && nn1 < (n1 + height))
                    //             std::cout << 'x' << ' ';
                    //         else
                    //             std::cout << '0' << ' ';
                    //     }
                    //     std::cout << std::endl;
                    // }

                    std::cout << "----" << std::endl;
                    std::cout << k << std::endl;
                    std::cout << n0 - width << ", " << n1 - height << std::endl;
                    std::cout << n0 + width << ", " << n1 + height << std::endl;
                    std::cout << width << ", " << height << std::endl;
                    std::cout << h << std::endl;
                    std::cout << l1 << ", " << m1 << ", " << h1 << "(" << l1 + m1 + h1 << ")" << std::endl;
                    std::cout << l2 << ", " << m2 << ", " << h2 << "(" << l2 + m2 + h2 << ")" << std::endl;
                    std::cout << "----" << std::endl;
                    fail++;
                    //tree1.print_debug();
                    return;
                } 
            }

            std::cout << "\r" << k;

        }

        std::cout << "\rSuccess: " << suc << std::endl;
        std::cout << "Fail: " << fail << std::endl;

    }

    {
        using BT = uint32_t;
        map.resize(6400*6400);
        std::fill(map.begin(), map.end(), 0.);

        MedianQuadTree<double, BT, 25>tree1(map, 6400);

        std::cout << "Tree size 25 (double)" << std::endl;

        using Dist = ::std::uniform_int_distribution<>;
        using Rand = std::mt19937_64;
        using DistD = ::std::uniform_real_distribution<double>;

        std::random_device rd;
        const auto seed = rd();
        std::cout << seed << std::endl;
        std::mt19937_64 rnd(seed);

        uint32_t suc = 0;
        uint32_t fail = 0;

        const uint32_t bc = 6400 / 25;
        std::vector<bool> mm;
        mm.resize(bc * bc);
        std::fill(mm.begin(), mm.end(), true);

        for(uint32_t k = 0; k < 100; ++k){

            {
                const BT width = Dist(10, 500)(rnd) - 1;
                const BT height = Dist(10, 500)(rnd) - 1;
                const BT xmin = Dist(0, 6400 - width - 1)(rnd);
                const BT ymin = Dist(0, 6400 - height - 1)(rnd);
                const BT h = std::round(Dist(10., 200.)(rnd));

                for(uint32_t n0 = ymin; n0 <= ymin + height; ++n0){
                    for(uint32_t n1 = xmin; n1 <= xmin + width; ++n1){
                        const uint32_t i = n1 + n0 * 6400;
                        assert(i >= 0 && i < map.size());
                        map[i] = k;
                    }
                }

                std::fill(mm.begin(), mm.end(), false);
                for (uint32_t n1 = xmin / 25; n1 <= std::min((xmin + width) / 25 + 1, bc - 1); ++n1) {
                    for (uint32_t n0 = ymin / 25; n0 <= std::min((ymin + height) / 25 + 1, bc - 1); ++n0) {         
                        const uint32_t iid = n0 + n1 * bc;
                        mm[iid] = true;
                    }
                }

                tree1.recompute(mm);
            }

            //----------------

            
            for(uint32_t j = 0; j < 100; ++j){

                const uint32_t width = Dist(250, 1250)(rnd) - 1;
                const uint32_t height = Dist(250, 1250)(rnd) - 1;
                const uint32_t n0 = Dist(width + 1, 6400 - width - 1)(rnd);
                const uint32_t n1 = Dist(height + 1, 6400 - height - 1)(rnd);
                const uint32_t h = std::round(Dist(10., 200.)(rnd));

                const auto[l1, m1, h1] = tree1.check_overlap(Vec2<BT>{n0 - width, n1 - height}, Vec2<BT>{n0 + width, n1 + height}, h);
                const auto[l2, m2, h2] = MQT2::Detail::naive_tester<double>(map, Vec2<BT>{n0 - width, n1 - height}, Vec2<BT>{n0 + width, n1 + height}, 6400, h);

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
    
}//MQT2_tester_float

void MQT2_tester_int() {

    using namespace MQT2;

    std::vector<uint32_t> map;
    map.resize(40*40);
    std::fill(map.begin(), map.end(), 0.);   
    for(uint32_t n0 = 23; n0 <= 31; ++n0){
        for(uint32_t n1 = 20; n1 <= 25; ++n1){
            const uint32_t i = n1 + n0 * 40;
            assert(i >= 0 && i < map.size());
            map[i] = 2;
        }
    }

    // for(uint32_t n0 = 0; n0 < 40; ++n0){
    //     for(uint32_t n1 = 0; n1 < 40; ++n1){
    //         const uint32_t i = n1 + n0 * 40;
    //         std::cout << (map[i] > 1. ? 'x' : '0' ) << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    {
        using BT = uint16_t;
        MedianQuadTree<uint32_t, BT, 10> tree(map, 40);
        //tree.print_debug();
        //return;

        //std::cout << tree << std::endl;
        std::cout << "----- Tree single box (int)-----" << std::endl;

        //full overlap
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{40, 40}, 1);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{40, 40}, 40, 1);
            std::cout << "Test 1: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //mid
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{5, 5}, Vec2<BT>{35, 35}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{5, 5}, Vec2<BT>{35, 35}, 40, 1);
            std::cout << "Test 2: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;

            // for(uint32_t n0 = 0; n0 < 40; ++n0){
            //     for(uint32_t n1 = 0; n1 < 40; ++n1){
            //         const uint32_t i = n1 + n0 * 40;
            //         if(n0 >= 5 && n0 < 35 && n1 >= 5 && n1 < 35)
            //             std::cout << 'x' << ' ';
            //         else
            //             std::cout << '0' << ' ';
            //     }
            //     std::cout << std::endl;
            // }
        }

        //return;

        //partial
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 40, 1);
            std::cout << "Test 3: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 40, 1);
            std::cout << "Test 4: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 40, 1);
            std::cout << "Test 5: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;        
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 40, 1);
            std::cout << "Test 6: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 40, 1);
            std::cout << "Test 7: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 40, 1);
            std::cout << "Test 8: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
    }

    for(uint32_t n1 = 0; n1 < 40; ++n1){
        for(uint32_t n0 = 0; n0 < 40; ++n0){
            const uint32_t i = n1 + n0 * 40;
            assert(i >= 0 && i < map.size());
            if(i%2) map[i] = 2.;
        }
    }

    {
        using BT = uint16_t;
        MedianQuadTree<uint32_t, BT, 10> tree(map, 40);
        //std::cout << tree << std::endl;
        std::cout << "----- Tree checker (int) -----" << std::endl;

        //full overlap
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{20, 20}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{20, 20}, 40, 1);
            std::cout << "Test 1: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //mid
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{5, 5}, Vec2<BT>{15, 15}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{5, 5}, Vec2<BT>{15, 15}, 40, 1);
            std::cout << "Test 2: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }

        //partial
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 10}, 40, 1);
            std::cout << "Test 3: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{10, 0}, Vec2<BT>{25, 10}, 40, 1);
            std::cout << "Test 4: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 10}, Vec2<BT>{10, 25}, 40, 1);
            std::cout << "Test 5: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{10, 10}, Vec2<BT>{25, 25}, 40, 1);
            std::cout << "Test 6: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 0}, Vec2<BT>{10, 25}, 40, 1);
            std::cout << "Test 7: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
        if constexpr(true){
            const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 1.);
            const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t, BT>(map, Vec2<BT>{0, 5}, Vec2<BT>{12, 17}, 40, 1);
            std::cout << "Test 8: ";
            std::cout << " Res: " << l1 << ", " << m1 << ", " << h1;
            std::cout << " Ex: " << l2 << ", " << m2 << ", " << h2;
            std::cout << (l1 == l2 && m1 == m2 && h1 == h2 ? " passed" : " failed") << std::endl;
        }
    }

    //-------------------------------

    {
        using BT = uint32_t;
        constexpr uint32_t size = 10*2*2*2*2*2*2*2;
        map.resize(size*size);
        std::fill(map.begin(), map.end(), 0);

        MedianQuadTree<uint32_t, BT, 10>tree1(map, size);

        std::cout << "Tree size 10 (int)" << std::endl;

        using Dist = ::std::uniform_int_distribution<>;
        using Rand = std::mt19937_64;
        //using DistD = ::std::uniform_real_distribution<double>;

        std::random_device rd;
        const auto seed = 2937685668; // rd();
        std::cout << seed << std::endl;
        std::mt19937_64 rnd(seed);

        uint32_t suc = 0;
        uint32_t fail = 0;

        const uint32_t bc = size / 10;
        std::vector<bool> mm;
        mm.resize(bc * bc);
        std::fill(mm.begin(), mm.end(), true);

        for(uint32_t k = 0; k < 100; ++k){

            {
                const uint32_t width = Dist(2, size/4)(rnd);
                const uint32_t height = Dist(2, size/4)(rnd);
                const uint32_t xmin = Dist(0, size - width - 1)(rnd);
                const uint32_t ymin = Dist(0, size - height - 1)(rnd);
                const double h = std::round(Dist(10, 200)(rnd));

                for(uint32_t n0 = ymin; n0 <= ymin + height; ++n0){
                    for(uint32_t n1 = xmin; n1 <= xmin + width; ++n1){
                        const uint32_t i = n1 + n0 * size;
                        assert(i >= 0 && i < map.size());
                        map[i] = k;
                    }
                }

                std::fill(mm.begin(), mm.end(), false);
                for (uint32_t n1 = xmin / 10; n1 <= std::min((xmin + width) / 10 + 1, bc - 1); ++n1) {
                    for (uint32_t n0 = ymin / 10; n0 <= std::min((ymin + height) / 10 + 1, bc - 1); ++n0) {         
                        const uint32_t iid = n0 + n1 * bc;
                        mm[iid] = true;
                    }
                }

            }

            
            //std::cout << "-----" << std::endl;
            tree1.recompute(mm);

            {

                const uint32_t width = Dist(size/4, size/2 - 2)(rnd);
                const uint32_t height = Dist(size/4, size/2 - 2)(rnd);
                const uint32_t n0 = Dist(width + 1, size - width - 1)(rnd);
                const uint32_t n1 = Dist(height + 1, size - height - 1)(rnd);
                const double h = std::round(Dist(10., 200.)(rnd));
                //if(k != 34) continue;
                const auto[l1, m1, h1] = tree1.check_overlap(Vec2<BT>{n0 - width, n1 - height}, Vec2<BT>{n0 + width, n1 + height}, h);
                const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t>(map, Vec2<BT>{n0 - width, n1 - height}, Vec2<BT>{n0 + width, n1 + height}, size, h);

                if(l1 == l2 && m1 == m2 && h1 == h2) suc++;
                else{
                    // std::cout << std::endl;
                    // for(uint32_t nn0 = 0; nn0 < 40; ++nn0){
                    //     for(uint32_t nn1 = 0; nn1 < 40; ++nn1){
                    //         const uint32_t i = nn1 + nn0 * 40;
                    //         if(map[i] > h)
                    //             std::cout << 'x' << ' ';
                    //         else
                    //             std::cout << '0' << ' ';
                    //     }
                    //     std::cout << std::endl;
                    // }
                    // std::cout << std::endl;
                    // for(uint32_t nn0 = 0; nn0 < 40; ++nn0){
                    //     for(uint32_t nn1 = 0; nn1 < 40; ++nn1){
                    //         const uint32_t i = nn1 + nn0 * 40;
                    //         if(nn0 >= (n0 - width) && nn0 < (n0 + width) && nn1 >= (n1 - height) && nn1 < (n1 + height))
                    //             std::cout << 'x' << ' ';
                    //         else
                    //             std::cout << '0' << ' ';
                    //     }
                    //     std::cout << std::endl;
                    // }

                    std::cout << "----" << std::endl;
                    std::cout << k << std::endl;
                    std::cout << n0 - width << ", " << n1 - height << std::endl;
                    std::cout << n0 + width << ", " << n1 + height << std::endl;
                    std::cout << width << ", " << height << std::endl;
                    std::cout << h << std::endl;
                    std::cout << l1 << ", " << m1 << ", " << h1 << "(" << l1 + m1 + h1 << ")" << std::endl;
                    std::cout << l2 << ", " << m2 << ", " << h2 << "(" << l2 + m2 + h2 << ")" << std::endl;
                    std::cout << "----" << std::endl;
                    fail++;
                    return;
                } 
            }

            std::cout << "\r" << k;

        }

        std::cout << "\rSuccess: " << suc << std::endl;
        std::cout << "Fail: " << fail << std::endl;

    }

    {
        using BT = uint32_t;
        map.resize(6400*6400);
        std::fill(map.begin(), map.end(), 0);

        MedianQuadTree<uint32_t, BT, 25>tree1(map, 6400);

        std::cout << "Tree size 25 (int)" << std::endl;

        using Dist = ::std::uniform_int_distribution<>;
        using Rand = std::mt19937_64;
        //using DistD = ::std::uniform_real_distribution<double>;

        std::random_device rd;
        const auto seed = rd();
        std::cout << seed << std::endl;
        std::mt19937_64 rnd(seed);

        uint32_t suc = 0;
        uint32_t fail = 0;

        const uint32_t bc = 6400 / 25;
        std::vector<bool> mm;
        mm.resize(bc * bc);
        std::fill(mm.begin(), mm.end(), true);

        for(uint32_t k = 0; k < 100; ++k){

            {
                const uint32_t width = Dist(10, 500)(rnd);
                const uint32_t height = Dist(10, 500)(rnd);
                const uint32_t xmin = Dist(0, 6400 - width - 1)(rnd);
                const uint32_t ymin = Dist(0, 6400 - height - 1)(rnd);
                const uint32_t h = std::round(Dist(10., 200.)(rnd));

                for(uint32_t n0 = ymin; n0 <= ymin + height; ++n0){
                    for(uint32_t n1 = xmin; n1 <= xmin + width; ++n1){
                        const uint32_t i = n1 + n0 * 6400;
                        map[i] = k;
                    }
                }

                std::fill(mm.begin(), mm.end(), false);
                for (uint32_t n1 = xmin / 25; n1 <= std::min((xmin + width) / 25 + 1, bc - 1); ++n1) {
                    for (uint32_t n0 = ymin / 25; n0 <= std::min((ymin + height) / 25 + 1, bc - 1); ++n0) {         
                        const uint32_t iid = n0 + n1 * bc;
                        mm[iid] = true;
                    }
                }

                tree1.recompute(mm);
            }

            //----------------

            
            for(uint32_t j = 0; j < 100; ++j){

                const uint32_t width = Dist(250, 1250)(rnd);
                const uint32_t height = Dist(250, 1250)(rnd);
                const uint32_t n0 = Dist(width + 1, 6400 - width - 1)(rnd);
                const uint32_t n1 = Dist(height + 1, 6400 - height - 1)(rnd);
                const uint32_t h = std::round(Dist(10., 200.)(rnd));

                const auto[l1, m1, h1] = tree1.check_overlap(Vec2<BT>{n0 - width, n1 - height}, Vec2<BT>{n0 + width, n1 + height}, h);
                const auto[l2, m2, h2] = MQT2::Detail::naive_tester<uint32_t>(map, Vec2<BT>{n0 - width, n1 - height}, Vec2<BT>{n0 + width, n1 + height}, 6400, h);

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
    
}//MQT2_tester_int

void bench_tree2() {
    using namespace MQT2;

    std::random_device rd;
    const auto seed = rd();
    std::cout << seed << std::endl;
    std::mt19937_64 rand(seed);

    using SCALAR = uint16_t;

    std::vector<SCALAR> map;
    // map.resize(1000 * 1000);
    // std::fill(map.begin(), map.end(), 0.);

    

    // for(uint32_t i = 0; i < 1000; ++i)
    //     map[dist(rand)] = 2.;

    // uint32_t t = 0;
    // for(uint32_t i = 1; i < 21; ++i){
    //     MedianQuadTree<double> tree(map, 1000, 1000, 100, 5 * i);

    //     const auto start = std::chrono::high_resolution_clock::now();
    //     const auto[l, m, h] = tree.check_overlap(Vec2<BT>{250, 200}, Vec2<BT>{750, 700}, 1.);
    //     const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;
    //     std::cout << i << " - " << ee.count() << std::endl;
    //     t += l;
    // }
   

    using Dist = ::std::uniform_int_distribution<>;
    using Rand = std::mt19937_64;
    //using DistD = ::std::uniform_real_distribution<SCALAR>;

    map.resize(7680 * 7680);
    std::uniform_int_distribution<> dist (0, uint32_t(map.size() - 1));
    std::fill(map.begin(), map.end(), 0.);
    uint32_t t = 0;

    if constexpr(false){
        for(uint32_t i = 0; i < 2000 * 7680; ++i)
            map[dist(rand)] = 2.;

    }

    //sudo chrt -f 99 ./mqttest 

    for(uint32_t k = 0; k < 100; ++k){

        {
            const uint32_t width = Dist(10, 500)(rand);
            const uint32_t height = Dist(10, 500)(rand);
            const uint32_t xmin = Dist(0, 7680 - width)(rand);
            const uint32_t ymin = Dist(0, 7680 - height)(rand);
            const SCALAR h = k;

            for(uint32_t n0 = ymin; n0 <= ymin + height; ++n0){
                for(uint32_t n1 = xmin; n1 <= xmin + width; ++n1){
                    const uint32_t i = n1 + n0 * 7680;
                    map[i] = h;
                }
            }
        }

    }
    std::ofstream file ("res.txt");
    const SCALAR hh = 50.; // std::round(DistD(10., 200.)(rand));

    std::cout << "-----------------" << std::endl;

    {
        file << "tree 1" << std::endl;
        {
            using BT = uint32_t;
            MedianQuadTree<SCALAR, BT, 15> tree(map, 7680);

            for(uint32_t i = 1; i < 50; ++i){

                double tmp = 0.;
                for(uint32_t j = 0; j < 12; ++j){
                    const auto start = std::chrono::high_resolution_clock::now();
                    const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{i * 50, i * 50}, Vec2<BT>{7680 - i * 50, 7680 - i * 50}, hh);
                    const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;          
                    tmp += ee.count();
                    t += l1;
                }

                file << tmp / 12. << std::endl;

            }
        }

        // std::cout << std::endl;
        // std::cout << "tree 2" << std::endl;
        // {
        //     MedianQuadTree<double> tree(map, 7680, 7680, 100, 30);

        //     for(uint32_t i = 1; i < 50; ++i){

        //         double tmp = 0.;
        //         for(uint32_t j = 0; j < 12; ++j){
        //             const auto start = std::chrono::high_resolution_clock::now();
        //             const auto[l1, m1, h1] = tree.check_overlap(Vec2<BT>{i * 50, i * 50}, Vec2<BT>{7680 - i * 50, 7680 - i * 50}, hh);
        //             const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;          
        //             tmp += ee.count();
        //             t += l1;
        //         }

        //         std::cout << tmp / 12. << std::endl;

        //     }
        // }

        if constexpr(true){
            using BT = uint32_t;
            file << std::endl;
            file << "naive 1" << std::endl;
            for(uint32_t i = 1; i < 50; ++i){

                double tmp = 0.;
                for(uint32_t j = 0; j < 12; ++j){
                    const auto start = std::chrono::high_resolution_clock::now();
                    const auto[l2, m2, h2] = MQT2::Detail::naive_tester<SCALAR, BT>(map, Vec2<BT>{BT(i * 50), BT(i * 50)}, Vec2<BT>{7680 - i * 50, 7680 - i * 50}, 7680, hh);

                    const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;
                    tmp += ee.count();

                    t += l2;
                }
                file << tmp / 12. << std::endl;

            }
        }
    }

    std::cout << t << std::endl;

}

void bench_border_tree2() {
    using namespace MQT2;

    std::random_device rd;
    const auto seed = rd();
    std::cout << seed << std::endl;
    std::mt19937_64 rand(seed);

    using SCALAR = uint8_t;

    std::vector<SCALAR> map;
 
    using Dist = ::std::uniform_int_distribution<>;
    using Rand = std::mt19937_64;
    //using DistD = ::std::uniform_real_distribution<SCALAR>;

    map.resize(7680 * 7680);
    std::uniform_int_distribution<> dist (0, uint32_t(map.size() - 1));
    std::fill(map.begin(), map.end(), 0.);
    uint32_t t = 0;

    if constexpr(false){
        for(uint32_t i = 0; i < 2000 * 7680; ++i)
            map[dist(rand)] = 2.;

    }

    //sudo chrt -f 99 ./mqttest 

    for(uint32_t k = 0; k < 100; ++k){

        {
            const uint32_t width = Dist(10, 500)(rand);
            const uint32_t height = Dist(10, 500)(rand);
            const uint32_t xmin = Dist(0, 7680 - width)(rand);
            const uint32_t ymin = Dist(0, 7680 - height)(rand);
            const SCALAR h = k;

            for(uint32_t n0 = ymin; n0 <= ymin + height; ++n0){
                for(uint32_t n1 = xmin; n1 <= xmin + width; ++n1){
                    const uint32_t i = n1 + n0 * 7680;
                    map[i] = h;
                }
            }
        }

    }
    std::ofstream file ("res.txt");
    const SCALAR hh = 50.; // std::round(DistD(10., 200.)(rand));

    std::cout << "-----------------" << std::endl;

    {
        file << "tree 1" << std::endl;
        {
            using BT = uint32_t;
            MedianQuadTree<SCALAR, BT, 15> tree(map, 7680);

            for(uint32_t i = 1; i < 50; ++i){

                double tmp = 0.;
                for(uint32_t j = 0; j < 12; ++j){
                    const auto start = std::chrono::high_resolution_clock::now();
                    const auto[l1, m1, h1] = tree.check_border_overlap(Vec2<BT>{i * 50, i * 50}, Vec2<BT>{7680 - i * 50, 7680 - i * 50}, hh);
                    const std::chrono::duration<double> ee = std::chrono::high_resolution_clock::now() - start;          
                    tmp += ee.count();
                    t += l1;
                }

                file << tmp / 12. << std::endl;

            }
        }

    }

    std::cout << t << std::endl;

}

void idx_test() {

    const auto idx_n = [](const uint32_t _level, const uint32_t _idx)->uint32_t {
        return 4 * _idx + 1;
    };

    const auto idx_b = [](const uint32_t _level, const uint32_t _idx)->uint32_t {
        return uint32_t(4. * double(_idx) - 4. * std::pow(4., _level - 1) / 3. + 4./3.);
    };

    std::queue<std::pair<uint32_t, uint32_t>> q;
    q.push({0, 0});

    while(!q.empty()){

        const auto[idx, lvl] = q.front();
        q.pop();

        std::cout << "l: " << lvl << " i: " << idx << std::endl;

        if(lvl < 2){
            const uint32_t c1 = 4 * idx + 1;
            const uint32_t c2 = c1 + 1;
            const uint32_t c3 = c2 + 1;
            const uint32_t c4 = c3 + 1;

            q.push({ c1, lvl + 1 });
            q.push({ c2, lvl + 1 });
            q.push({ c3, lvl + 1 });
            q.push({ c4, lvl + 1 });
        }

    }

    std::cout << std::endl;

    for(uint32_t i = 5; i < 21; ++i){
        std::cout << idx_b(3, i) << std::endl;
    }

}

void depth_test() {

    uint32_t BUCKET_SIZE = 5;
    uint32_t w = 1600;

    for(uint32_t i = 0; i < 10; ++i){
        const uint32_t bc = w / BUCKET_SIZE;
        const uint32_t max_level_ = uint32_t(std::round(std::log(bc) / std::log(2))) + 1;

        std::cout << "---------" << std::endl;
        std::cout << BUCKET_SIZE << std::endl;
        //std::cout << bc << std::endl;
        std::cout << max_level_ << std::endl;

        BUCKET_SIZE *= 2;
    }
}

int main() {
    //test_bucket_node();
    //bench_tree();
    bench_tree2();
    //bench_border_tree2();
    //morton_test();
    //MQT2_tester_float();
    //MQT2_tester_int();

    //idx_test();

    //depth_test();

    return 0;
}