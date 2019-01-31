// TODO: Assess whether this test considers valid behaviour without C++AMP.
// RUN: %cxxamp -c %s

#define TEST(a,b) static_assert(std::is_same<a,b>::value, "Test failed, type of \"" #a "\" != type of \"" #b "\".")
struct cpu_t
{
        operator bool() [[cpu, hc]]; // Req'd to define in 'if' condition
};
struct amp_t
{
        operator bool() [[cpu, hc]]; // Req'd to define in 'if' condition
        int i; // Req'd to satisfy alignment
};

cpu_t f() [[cpu]];
amp_t f() [[hc]];

auto test_trt_2() [[hc]] -> decltype(f()); // expect: amp_t test_trt_2() [[hc]]

void test_trt_2_verify() [[hc]]
{
        amp_t r = test_trt_2(); // verify                              // Error
        // since the auto & trailing return type of test_trt_2 is cpu_t
}
