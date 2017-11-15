
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>
#include <iostream>

#define ERROR_THRESHOLD (1E-4) // Potentially dangerous / inexact.

//#define DEBUG 1

void report_error(const char* fn_name, double cumulative_error)
{
    #ifdef DEBUG
        std::cout << fn_name << " cumulative error = " << cumulative_error
                  << ", test failed!" << std::endl;
    #endif

}

void report_delta(
    const char* fn_name, double argument, double expected, double actual)
{
    #ifdef DEBUG
        std::cout << fn_name << '(' << argument << ") expected = " << expected
                  << ", actual = " << actual << std::endl;
    #endif
}

// a test case which uses hc_math, which overrides math functions in the global namespace
template<typename T, std::size_t grid_sz, typename F, typename G>
bool test_math_fn(const char* name, F f, G ref_f)
{   // TODO: ideally this should be refactored to use proper approximate
    //       equality / comparison for floating-point types.
    using namespace hc;

    array_view<T> table(grid_sz);

    parallel_for_each(table.get_extent(), [=](const index<1>& idx) __HC__ {
       table[idx] = f(static_cast<T>(idx[0] + 1));
    });

    double error = 0.0;
    for (auto i = 0; i != table.get_extent().size(); ++i) {
        T actual = table[i];
        T expected = ref_f(static_cast<T>(i + 1));
        T delta = fabs(static_cast<double>(actual) - static_cast<double>(expected));

        if (ERROR_THRESHOLD < delta) report_delta(name, i + 1, expected, actual);
        error += delta;
    }
    if (ERROR_THRESHOLD < error) report_error(name, error);

    return error <= ERROR_THRESHOLD;
}

template<typename T, std::size_t grid_sz>
bool test()
{   // TODO: ideally this should be refactored to use iteration through the
    //       collection of tested functions, as opposed to this verbose form.
    using namespace hc;

    return test_math_fn<T, grid_sz>(
        "sqrt",
        [](T x) __HC__ { return sqrt(x); }, [](T x) { return std::sqrt(x); })
        && test_math_fn<T, grid_sz>(
        "fabs",
        [](T x) __HC__ { return fabs(x); }, [](T x) { return std::fabs(x); })
        && test_math_fn<T, grid_sz>(
        "cbrt",
        [](T x) __HC__ { return cbrt(x); }, [](T x) { return std::cbrt(x); })
        && test_math_fn<T, grid_sz>(
        "log",
        [](T x) __HC__ { return log(x); }, [](T x) { return std::log(x); })
        && test_math_fn<T, grid_sz>(
        "ilogb",
        [](T x) __HC__ { return ilogb(x); }, [](T x) { return std::ilogb(x); })
        && test_math_fn<T, grid_sz>(
        "isnormal",
        [](T x) __HC__ { return isnormal(x); },
        [](T x) { return std::isnormal(x); })
        && test_math_fn<T, grid_sz>(
        "cospi",
        [](T x) __HC__ { return cospi(x); },
        [](T x) { return std::cos(static_cast<T>(M_PI) * x); })
        && test_math_fn<T, grid_sz>(
        "sinpi",
        [](T x) __HC__ { return sinpi(x); },
        [](T x) { return std::sin(static_cast<T>(M_PI) * x); })
        && test_math_fn<T, grid_sz>(
        "rsqrt",
        [](T x) __HC__ { return rsqrt(x); },
        [](T x) { return static_cast<T>(1) / std::sqrt(x); });
}

int main()
{
  bool ret = true;

  //ret &= test<int,16>();
  ret &= test<float,16>();
  ret &= test<double,16>();

  return ret == false;
}

