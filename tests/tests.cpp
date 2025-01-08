
#include <MQT2/MQT2.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

template <class T>
void mqt2_flat_square_40(std::vector<T>& _map) {
  using namespace MQT2;
  {
    MedianQuadTree<T, 10> tree(&_map, 40);
    // full overlap
    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{0, 0}, Vec2{40, 40}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{0, 0}, Vec2{40, 40}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    // mid
    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{5, 5}, Vec2{35, 35}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{5, 5}, Vec2{35, 35}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    // partial
    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{0, 0}, Vec2{10, 10}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{0, 0}, Vec2{10, 10}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{10, 0}, Vec2{25, 10}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{10, 0}, Vec2{25, 10}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{0, 10}, Vec2{10, 25}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{0, 10}, Vec2{10, 25}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{10, 10}, Vec2{25, 25}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{10, 10}, Vec2{25, 25}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{0, 0}, Vec2{10, 25}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{0, 0}, Vec2{10, 25}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }

    {
      const auto [l1, m1, h1] = tree.check_overlap(Vec2{0, 5}, Vec2{12, 17}, 1.);
      const auto [l2, m2, h2] = MQT2::Detail::naive_tester<T>(_map, Vec2{0, 5}, Vec2{12, 17}, 40, 1.);
      REQUIRE(l1 == l2);
      REQUIRE(m1 == m2);
      REQUIRE(h1 == h2);
    }
  }

}  // mqt2_tester

TEST_CASE("float", "[flat with square]") {
  std::vector<float> map;
  map.resize(40 * 40);
  std::fill(map.begin(), map.end(), 0.f);
  for (uint32_t n0 = 23; n0 <= 31; ++n0) {
    for (uint32_t n1 = 20; n1 <= 25; ++n1) {
      const uint32_t i = n1 + n0 * 40;
      map[i]           = 2.f;
    }
  }

  mqt2_flat_square_40(map);
}

TEST_CASE("double", "[flat with square]") {
  std::vector<double> map;
  map.resize(40 * 40);
  std::fill(map.begin(), map.end(), 0.);
  for (uint32_t n0 = 23; n0 <= 31; ++n0) {
    for (uint32_t n1 = 20; n1 <= 25; ++n1) {
      const uint32_t i = n1 + n0 * 40;
      map[i]           = 2.;
    }
  }

  mqt2_flat_square_40(map);
}

TEST_CASE("int8", "[flat with square]") {
  std::vector<int8_t> map;
  map.resize(40 * 40);
  std::fill(map.begin(), map.end(), 0);
  for (uint32_t n0 = 23; n0 <= 31; ++n0) {
    for (uint32_t n1 = 20; n1 <= 25; ++n1) {
      const uint32_t i = n1 + n0 * 40;
      map[i]           = 2;
    }
  }

  mqt2_flat_square_40(map);
}

TEST_CASE("int16", "[flat with square]") {
  std::vector<int16_t> map;
  map.resize(40 * 40);
  std::fill(map.begin(), map.end(), 0);
  for (uint32_t n0 = 23; n0 <= 31; ++n0) {
    for (uint32_t n1 = 20; n1 <= 25; ++n1) {
      const uint32_t i = n1 + n0 * 40;
      map[i]           = 2;
    }
  }

  mqt2_flat_square_40(map);
}

TEST_CASE("int32", "[flat with square]") {
  std::vector<int32_t> map;
  map.resize(40 * 40);
  std::fill(map.begin(), map.end(), 0);
  for (uint32_t n0 = 23; n0 <= 31; ++n0) {
    for (uint32_t n1 = 20; n1 <= 25; ++n1) {
      const uint32_t i = n1 + n0 * 40;
      map[i]           = 2;
    }
  }

  mqt2_flat_square_40(map);
}

TEST_CASE("int64", "[flat with square]") {
  std::vector<int64_t> map;
  map.resize(40 * 40);
  std::fill(map.begin(), map.end(), 0);
  for (uint32_t n0 = 23; n0 <= 31; ++n0) {
    for (uint32_t n1 = 20; n1 <= 25; ++n1) {
      const uint32_t i = n1 + n0 * 40;
      map[i]           = 2;
    }
  }

  mqt2_flat_square_40(map);
}
