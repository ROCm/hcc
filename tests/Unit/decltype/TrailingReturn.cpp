// RUN: %cxxamp -c %s

#define TEST(a,b) static_assert(std::is_same<a,b>::value, "Test failed, type of \"" #a "\" != type of \"" #b "\".")
struct cpu_t
{
        operator bool() restrict(cpu,amp); // Req'd to define in 'if' condition
};
struct amp_t
{
        operator bool() restrict(cpu,amp); // Req'd to define in 'if' condition
        int i; // Req'd to satisfy alignment
};

cpu_t f() restrict(cpu);
amp_t f() restrict(amp);

auto test_trt_2() restrict(amp) -> decltype(f()); // expect: amp_t test_trt_2() restrict(amp)

void test_trt_2_verify() restrict(amp)
{
        amp_t r = test_trt_2(); // verify                              // Error
        // since the auto & trailing return type of test_trt_2 is cpu_t
}
