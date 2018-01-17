// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify erronous decltype expressions in a function trailing return type.</summary>
#include "../expression_common.h"

auto f_cpu_1() restrict(cpu) -> TEST_CPU_1;
//#Expects: Error: test\.cpp\(10\) : error C3930:.*(\bf_4\b)

auto f_amp_1() restrict(amp)	-> TEST_AMP_1;
auto f_amp_2() restrict(amp)	-> TEST_AMP_2;
auto f_amp_3() restrict(amp)	-> TEST_AMP_3;
auto f_amp_4() restrict(amp)	-> TEST_AMP_4;
auto f_amp_5() restrict(amp)	-> TEST_AMP_5;
auto f_amp_6() restrict(amp)	-> TEST_AMP_6;
auto f_amp_7() restrict(amp)	-> TEST_AMP_7;
auto f_amp_8() restrict(amp)	-> TEST_AMP_8;
auto f_amp_9() restrict(amp)	-> TEST_AMP_9;
auto f_amp_10() restrict(amp)	-> TEST_AMP_10;
auto f_amp_11() restrict(amp)	-> TEST_AMP_11;
auto f_amp_12() restrict(amp)	-> TEST_AMP_12;
auto f_amp_13() restrict(amp)	-> TEST_AMP_13;
auto f_amp_14() restrict(amp)	-> TEST_AMP_14;
auto f_amp_15() restrict(amp)	-> TEST_AMP_15;
auto f_amp_16() restrict(amp)	-> TEST_AMP_16;
auto f_amp_17() restrict(amp)	-> TEST_AMP_17;
auto f_amp_18() restrict(amp)	-> TEST_AMP_18;
auto f_amp_19() restrict(amp)	-> TEST_AMP_19;
auto f_amp_20() restrict(amp)	-> TEST_AMP_20;
auto f_amp_21(int k) restrict(amp)	-> TEST_AMP_21(k);
auto f_amp_22(int k) restrict(amp)	-> TEST_AMP_22(k);
auto f_amp_23(int k) restrict(amp)	-> TEST_AMP_23(k);
auto f_amp_24(int k) restrict(amp)	-> TEST_AMP_24(k);
auto f_amp_25(int k) restrict(amp)	-> TEST_AMP_25(k);
auto f_amp_26(int k) restrict(amp)	-> TEST_AMP_26(k);
auto f_amp_27() restrict(amp)	-> TEST_AMP_27;
auto f_amp_28() restrict(amp)	-> TEST_AMP_28;
auto f_amp_29() restrict(amp)	-> TEST_AMP_29;
auto f_amp_30() restrict(amp)	-> TEST_AMP_30;
// wontfix,381215	#Expects: Error: test\.cpp\(8\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(9\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(10\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(11\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(12\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(18\) : error C3930:.*(\boperator new\b)
// wontfix,381215	#Expects: Error: test\.cpp\(14\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(15\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(16\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(17\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(23\) : error C2446:.*(\bchar\b).*(\bobj_conv_T\b)
// wontfix,381215	#Expects: Error: test\.cpp\(19\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(25\) : error C3581:.*(\bchar \(void\) restrict\(amp\))
// wontfix,381215	#Expects: Error: test\.cpp\(21\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(22\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(23\) : error C3595
// wontfix,381215	#Expects: Error: test\.cpp\(24\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(30\) : error C3581:.*(\bobj_T_m<T> \(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(31\) : error C3581:.*(\bobj_T_m_def_char<> \(void\) restrict\(amp\))
// wontfix,381215	#Expects: Error: test\.cpp\(27\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(28\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(29\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(30\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(31\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(32\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(33\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(34\) : error C3586.*'global_char'
// wontfix,381215	#Expects: Error: test\.cpp\(35\) : error C3586.*'global_int'
// wontfix,381215	#Expects: Error: test\.cpp\(36\) : error C3594
//#Expects: Error: test\.cpp\(42\) : error C3930:.*(\bf_3\b)

auto f_cpu_amp_cpu_1() restrict(cpu,amp)	-> TEST_CPU_1;
//#Expects: Error: test\.cpp\(74\) : error C3930:.*(\bf_4\b)

auto f_cpu_amp_amp_1() restrict(cpu,amp)	-> TEST_AMP_1;
auto f_cpu_amp_amp_2() restrict(cpu,amp)	-> TEST_AMP_2;
auto f_cpu_amp_amp_3() restrict(cpu,amp)	-> TEST_AMP_3;
auto f_cpu_amp_amp_4() restrict(cpu,amp)	-> TEST_AMP_4;
auto f_cpu_amp_amp_5() restrict(cpu,amp)	-> TEST_AMP_5;
auto f_cpu_amp_amp_6() restrict(cpu,amp)	-> TEST_AMP_6;
auto f_cpu_amp_amp_7() restrict(cpu,amp)	-> TEST_AMP_7;
auto f_cpu_amp_amp_8() restrict(cpu,amp)	-> TEST_AMP_8;
auto f_cpu_amp_amp_9() restrict(cpu,amp)	-> TEST_AMP_9;
auto f_cpu_amp_amp_10() restrict(cpu,amp)	-> TEST_AMP_10;
auto f_cpu_amp_amp_11() restrict(cpu,amp)	-> TEST_AMP_11;
auto f_cpu_amp_amp_12() restrict(cpu,amp)	-> TEST_AMP_12;
auto f_cpu_amp_amp_13() restrict(cpu,amp)	-> TEST_AMP_13;
auto f_cpu_amp_amp_14() restrict(cpu,amp)	-> TEST_AMP_14;
auto f_cpu_amp_amp_15() restrict(cpu,amp)	-> TEST_AMP_15;
auto f_cpu_amp_amp_16() restrict(cpu,amp)	-> TEST_AMP_16;
auto f_cpu_amp_amp_17() restrict(cpu,amp)	-> TEST_AMP_17;
auto f_cpu_amp_amp_18() restrict(cpu,amp)	-> TEST_AMP_18;
auto f_cpu_amp_amp_19() restrict(cpu,amp)	-> TEST_AMP_19;
auto f_cpu_amp_amp_20() restrict(cpu,amp)	-> TEST_AMP_20;
auto f_cpu_amp_amp_21(int k) restrict(cpu,amp)	-> TEST_AMP_21(k);
auto f_cpu_amp_amp_22(int k) restrict(cpu,amp)	-> TEST_AMP_22(k);
auto f_cpu_amp_amp_23(int k) restrict(cpu,amp)	-> TEST_AMP_23(k);
auto f_cpu_amp_amp_24(int k) restrict(cpu,amp)	-> TEST_AMP_24(k);
auto f_cpu_amp_amp_25(int k) restrict(cpu,amp)	-> TEST_AMP_25(k);
auto f_cpu_amp_amp_26(int k) restrict(cpu,amp)	-> TEST_AMP_26(k);
auto f_cpu_amp_amp_27() restrict(cpu,amp)	-> TEST_AMP_27;
auto f_cpu_amp_amp_28() restrict(cpu,amp)	-> TEST_AMP_28;
auto f_cpu_amp_amp_29() restrict(cpu,amp)	-> TEST_AMP_29;
auto f_cpu_amp_amp_30() restrict(cpu,amp)	-> TEST_AMP_30;
// wontfix,381215	#Expects: Error: test\.cpp\(72\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(73\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(74\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(75\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(76\) : error C3581.*'char'
// wontfix,397594	#Expects: Error: test\.cpp\(77\) : error C3930.*'operator new'
// wontfix,381215	#Expects: Error: test\.cpp\(78\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(79\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(80\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(81\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(87\) : error C2446:.*(\bchar\b).*(\bobj_conv_T\b)
// wontfix,381215	#Expects: Error: test\.cpp\(83\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(89\) : error C3581:.*(\bchar \(void\) restrict\(cpu, amp\))
// wontfix,381215	#Expects: Error: test\.cpp\(85\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(86\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(87\) : error C3595
// wontfix,381215	#Expects: Error: test\.cpp\(88\) : error C3581.*'char'
//#Expects: Error: test\.cpp\(94\) : error C3581:.*(\bobj_T_m<T> \(void\) restrict\(cpu, amp\))
//#Expects: Error: test\.cpp\(95\) : error C3581:.*(\bobj_T_m_def_char<> \(void\) restrict\(cpu, amp\))
// wontfix,381215	#Expects: Error: test\.cpp\(91\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(92\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(93\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(94\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(95\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(96\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(97\) : error C3581.*'char'
// wontfix,381215	#Expects: Error: test\.cpp\(98\) : error C3586.*'global_char'
// wontfix,381215	#Expects: Error: test\.cpp\(99\) : error C3586.*'global_int'
// wontfix,381215	#Expects: Error: test\.cpp\(100\) : error C3594
//#Expects: Error: test\.cpp\(106\) : error C3930:.*(\bf_3\b)

auto f_cpu_amp_1() restrict(cpu,amp)		-> TEST_CPU_AMP_1;
auto f_cpu_amp_2() restrict(cpu,amp)		-> TEST_CPU_AMP_2;
auto f_cpu_amp_3() restrict(cpu,amp)		-> TEST_CPU_AMP_3;
auto f_cpu_amp_4() restrict(cpu,amp)		-> TEST_CPU_AMP_4;
auto f_cpu_amp_5() restrict(cpu,amp)		-> TEST_CPU_AMP_5;
auto f_cpu_amp_6() restrict(cpu,amp)		-> TEST_CPU_AMP_6;
auto f_cpu_amp_7() restrict(cpu,amp)		-> TEST_CPU_AMP_7;
auto f_cpu_amp_8() restrict(cpu,amp)		-> TEST_CPU_AMP_8;
auto f_cpu_amp_9() restrict(cpu,amp)		-> TEST_CPU_AMP_9;
auto f_cpu_amp_10() restrict(cpu,amp)		-> TEST_CPU_AMP_10;
//#Expects: Error: test\.cpp\(138\) : error C3556:.*(\bf_1\b)
//#Expects: Error: test\.cpp\(139\) : error C3556:.*(\bf_2\b)
//#Expects: Error: test\.cpp\(140\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(141\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
// wontfix,388265	#Expects: Error: test\.cpp\(137\)
//#Expects: Error: test\.cpp\(143\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(144\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(145\) : error C2785:.*(\bT tf_2<float>\(T\)).*(\bint tf_2\(float\) restrict\(amp\))
//#Expects: Error: test\.cpp\(146\) : error C3930:.*(\bf_3\b)
//#Expects: Error: test\.cpp\(147\) : error C3930:.*(\bf_4\b)

