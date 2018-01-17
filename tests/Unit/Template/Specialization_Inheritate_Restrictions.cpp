// RUN: %cxxamp -c %s

template <typename T> T   tf_c_1(T) restrict(cpu, amp);
void f_cpu_amp() restrict(cpu, amp)
{
  tf_c_1(1.f);  // Expect tf_c_1 restrict(cpu,amp) here
}
