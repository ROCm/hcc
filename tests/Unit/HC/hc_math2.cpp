#define DISABLED_PENDING_REMOVAL true

#if !DISABLED_PENDING_REMOVAL
  // RUN: %hc %s -o %t.out && %t.out

  #include <hc.hpp>
  #include <hc_math.hpp>

  #include <algorithm>
  #include <random>

  // a test case which uses hc_math, which overrides math functions in the global namespace
  // in this test case we check min / max specically
  template<size_t GRID_SIZE, typename T>
  bool test() {
    using namespace hc;
    bool ret = true;

    array_view<T, 1> table1(GRID_SIZE); // input vector 1
    array_view<T, 1> table2(GRID_SIZE); // input vector 2
    array_view<T, 1> table3(GRID_SIZE); // output vector calculated by GPU
    extent<1> ex(GRID_SIZE);

    // setup RNG
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<T> dis(1, GRID_SIZE);

    // randomly produce input data
    for (int i = 0; i < GRID_SIZE; ++i) {
      table1[i] = dis(gen);
      table2[i] = dis(gen);
    }

  #define TEST(func) \
    { \
      parallel_for_each(ex, [=](index<1>& idx) __HC__ { \
        table3(idx) = func(table1(idx), table2(idx)); \
      }); \
      accelerator().get_default_view().wait(); \
      for (size_t i = 0; i < GRID_SIZE; ++i) { \
        if (func(table1[i], table2[i]) != table3[i]) { \
          ret = false; \
          break; \
        } \
      } \
    }

    TEST(min)
    TEST(max)

    return ret;
  }

  int main() {
    bool ret = true;

    ret &= test<16, float>();
    ret &= test<16, double>();
    ret &= test<4096, float>();
    ret &= test<4096, double>();

    return !(ret == true);
  }
#else
  int main() { return 0; }
#endif

