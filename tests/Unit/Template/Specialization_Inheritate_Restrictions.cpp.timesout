// RUN: %cxxamp -c %s

template <typename T> T   tf_c_1(T) [[cpu, hc]];
void f_cpu_amp() [[cpu, hc]]
{
  tf_c_1(1.f);  // Expect tf_c_1 [[cpu, hc]] here
}
