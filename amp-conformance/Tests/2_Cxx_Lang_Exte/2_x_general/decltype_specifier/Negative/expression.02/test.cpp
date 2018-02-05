// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify erronous decltype expressions in a lambda expression.</summary>
#include "../expression_common.h"


void f()
{
	[]
	{
		TEST_CPU_1;
	};
//#Expects: Error: test\.cpp\(15\) : error C3930:.*(\bf_4\b)

	[]() restrict(amp)
	{
		int k;

		TEST_AMP_1;
		TEST_AMP_2;
		TEST_AMP_3;
		TEST_AMP_4;
		TEST_AMP_5;
		TEST_AMP_6;
		TEST_AMP_7;
		TEST_AMP_8;
		TEST_AMP_9;
		TEST_AMP_10;
		TEST_AMP_11;
		TEST_AMP_12;
		TEST_AMP_13 test_amp_13;
		TEST_AMP_14;
		TEST_AMP_15;
		TEST_AMP_16;
		TEST_AMP_17;
		TEST_AMP_18;
		TEST_AMP_19;
		TEST_AMP_20;
		TEST_AMP_21(k);
		TEST_AMP_22(k);
		TEST_AMP_23(k);
		TEST_AMP_24(k);
		TEST_AMP_25(k);
		TEST_AMP_26(k);
		TEST_AMP_27;
		TEST_AMP_28;
		TEST_AMP_29;
		TEST_AMP_30;
	};
//#Expects: Error: test\.cpp\(23\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(24\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(25\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(26\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(27\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(28\) : error C3930:.*(\boperator new\b)
//#Expects: Error: test\.cpp\(29\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(30\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(31\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(32\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(33\) : error C2446:.*(\bchar\b).*(\bobj_conv_T\b)
//#Expects: Error: test\.cpp\(34\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(35\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(36\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(37\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(38\) : error C3595
//#Expects: Error: test\.cpp\(39\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(40\) : error C3581:.*(\bobj_T_m<T>)
//#Expects: Error: test\.cpp\(41\) : error C3581:.*(\bobj_T_m_def_char<>)
//#Expects: Error: test\.cpp\(42\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(43\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(44\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(45\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(46\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(47\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(48\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(49\) : error C3586:.*(\bglobal_char\b)
//#Expects: Error: test\.cpp\(50\) : error C3586:.*(\bglobal_int\b)
//#Expects: Error: test\.cpp\(51\) : error C3594
//#Expects: Error: test\.cpp\(52\) : error C3930:.*(\bf_3\b)

	[]() restrict(cpu,amp)
	{
		int k;

		TEST_CPU_1;

		TEST_AMP_1;
		TEST_AMP_2;
		TEST_AMP_3;
		TEST_AMP_4;
		TEST_AMP_5;
		TEST_AMP_6;
		TEST_AMP_7;
		TEST_AMP_8;
		TEST_AMP_9;
		TEST_AMP_10;
		TEST_AMP_11;
		TEST_AMP_12;
		TEST_AMP_13 test_amp_13;
		TEST_AMP_14;
		TEST_AMP_15;
		TEST_AMP_16;
		TEST_AMP_17;
		TEST_AMP_18;
		TEST_AMP_19;
		TEST_AMP_20;
		TEST_AMP_21(k);
		TEST_AMP_22(k);
		TEST_AMP_23(k);
		TEST_AMP_24(k);
		TEST_AMP_25(k);
		TEST_AMP_26(k);
		TEST_AMP_27;
		TEST_AMP_28;
		TEST_AMP_29;
		TEST_AMP_30;

		TEST_CPU_AMP_1;
		TEST_CPU_AMP_2;
		TEST_CPU_AMP_3;
		TEST_CPU_AMP_4;
		TEST_CPU_AMP_5;
		TEST_CPU_AMP_6;
		TEST_CPU_AMP_7;
		TEST_CPU_AMP_8;
		TEST_CPU_AMP_9;
		TEST_CPU_AMP_10;
	};
//#Expects: Error: test\.cpp\(89\) : error C3930:.*(\bf_4\b)

//#Expects: Error: test\.cpp\(91\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(92\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(93\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(94\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(95\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(96\) : error C3930:.*(\boperator new\b)
//#Expects: Error: test\.cpp\(97\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(98\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(99\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(100\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(101\) : error C2446:.*(\bchar\b).*(\bobj_conv_T\b)
//#Expects: Error: test\.cpp\(102\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(103\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(104\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(105\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(106\) : error C3595
//#Expects: Error: test\.cpp\(107\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(108\) : error C3581:.*(\bobj_T_m<T>)
//#Expects: Error: test\.cpp\(109\) : error C3581:.*(\bobj_T_m_def_char<>)
//#Expects: Error: test\.cpp\(110\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(111\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(112\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(113\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(114\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(115\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(116\) : error C3581:.*(\bchar\b)
//#Expects: Error: test\.cpp\(117\) : error C3586:.*(\bglobal_char\b)
//#Expects: Error: test\.cpp\(118\) : error C3586:.*(\bglobal_int\b)
//#Expects: Error: test\.cpp\(119\) : error C3594
//#Expects: Error: test\.cpp\(120\) : error C3930:.*(\bf_3\b)

//#Expects: Error: test\.cpp\(122\) : error C3556:.*(\bf_1\b)
//#Expects: Error: test\.cpp\(123\) : error C3556:.*(\bf_2\b)
//#Expects: Error: test\.cpp\(124\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(125\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
// wontfix,388265	#Expects: Error: test\.cpp\(121\)
//#Expects: Error: test\.cpp\(127\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(128\) : error C2785:.*(\bint f_2\(void\)).*(\bfloat f_2\(void\) restrict\(amp\))
//#Expects: Error: test\.cpp\(129\) : error C2785:.*(\bT tf_2<float>\(T\)).*(\bint tf_2\(float\) restrict\(amp\))
//#Expects: Error: test\.cpp\(130\) : error C3930:.*(\bf_3\b)
//#Expects: Error: test\.cpp\(131\) : error C3930:.*(\bf_4\b)

}

