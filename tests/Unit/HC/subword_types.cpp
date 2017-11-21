// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <cstddef>
#include <cstdint>

using namespace hc;
using namespace std;

template<typename T>
bool test_aggregate_use()
{
    array_view<T> out{42};
    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
       T tmp[42] = {};
       for (auto i = 0u; i != out.get_extent().size(); ++i) out[i] = tmp[i];
    });

    for (auto i = 0u; i != out.get_extent().size(); ++i) {
        if (!(out[i] == T{})) return false;
    }

    return true;
}

template<typename T>
bool test_direct_use()
{
    array_view<T> out{1};
    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        T tmp = {};
        out[0] = tmp;
    });

    return out[0] == T{} ? true : false;
}

template<typename T>
bool test_nested_use()
{
    struct Tmp { T x; };

    array_view<Tmp> out{1};
    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        Tmp tmp = {};
        out[0] = tmp;
    });

    return out[0].x == T{} ? true : false;
}

struct Empty {
    friend inline bool operator==(const Empty&, const Empty&) { return true; }
};

int main()
{
    bool ret = true;

    ret = test_aggregate_use<Empty>() &&
          test_direct_use<Empty>() &&
          test_nested_use<Empty>();
    ret = test_aggregate_use<std::uint8_t>() &&
          test_direct_use<std::uint8_t>() &&
          test_nested_use<std::uint8_t>();
    ret = test_aggregate_use<std::uint16_t>() &&
          test_direct_use<std::uint16_t>() &&
          test_nested_use<std::uint16_t>();

    return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}