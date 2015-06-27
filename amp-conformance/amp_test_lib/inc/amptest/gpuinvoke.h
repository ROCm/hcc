// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#pragma once

#include <amp.h>
#include <amptest/runall.h>
#include <amptest/logging.h>

namespace Concurrency {
	namespace Test {

		#pragma region run_test_on_cpu_and_gpu

		namespace details {
			inline void log_run_test_result(const std::string& func_desc, const std::string& on, runall_result result) {
				Log(result)
					<< func_desc << " on " << on << ": " << result << "."
					<< std::endl << std::flush;
			}

			// Note: 'Tout out' is the output we got from running the test function.
			template <typename Tout>
			void log_run_test_result(const std::string& func_desc, const std::string& on, runall_result result, bool log_out_on_pass, const Tout& out) {
				auto& log = Log(result);
				log << func_desc << " on " << on << ": " << result << ".";
				if(log_out_on_pass || !result.get_is_pass()) {
					log << " output value: " << format_as_code(out);
					// Also print the hex value for arithmetic types for more info
					if(std::is_arithmetic<Tout>::value) {
						log << " (0x" << std::hex << out << std::dec << ")";
					}
				}
				log << std::endl << std::flush;
			}
		}

		/// Runs a test functor (functor or lambda) on the CPU and GPU
		/// and provides simple reporting of the result on each.
		template <typename TestFunc>
		runall_result run_test_on_cpu_and_gpu(const concurrency::accelerator_view& av, const std::string& func_desc, const TestFunc& func) {
			/* Invoke on cpu */
			runall_result cpu_result = func();

			/* Invoke on gpu */
			runall_result gpu_result;
			concurrency::array_view<runall_result> gpu_resultsv(1, &gpu_result);
            gpu_resultsv.discard_data();
			concurrency::parallel_for_each(av, gpu_resultsv.get_extent()
				, [=](concurrency::index<1> idx) restrict(amp) {
					gpu_resultsv[idx] = func();
			});
			gpu_resultsv.synchronize();

			// Log the results
			details::log_run_test_result(func_desc, "CPU", cpu_result);
			details::log_run_test_result(func_desc, "GPU", gpu_result);

			return cpu_result & gpu_result;
		}

		/// Runs a test functor (functor or lambda) on the CPU and GPU
		/// and provides simple reporting of the result on each.
		/// This overload allows you to specify an out parameter in your
		/// lambda func (or functor). This output will be written to the log.
		/// The lambda signature should look something like:
		///    [](Tout& r) -> runall_result { r = ...; return result; }
		template <typename Tout, typename TestFunc>
		runall_result run_test_on_cpu_and_gpu(const concurrency::accelerator_view& av, const std::string& func_desc, const TestFunc& func, bool log_out_on_pass = false) {
			/* Invoke on cpu */
			Tout cpu_out;
			runall_result cpu_result = func(cpu_out);

			/* Invoke on gpu */
			Tout gpu_out;
			concurrency::array_view<Tout> gpu_outv(1, &gpu_out);
            gpu_outv.discard_data();
			runall_result gpu_result;
			concurrency::array_view<runall_result> gpu_resultsv(1, &gpu_result);
            gpu_resultsv.discard_data();
			concurrency::parallel_for_each(av, gpu_resultsv.get_extent()
				, [=](concurrency::index<1> idx) restrict(amp) {
					gpu_resultsv[idx] = func(gpu_outv[idx]);
			});
			gpu_outv.synchronize();
			gpu_resultsv.synchronize();

			// Log the results
			details::log_run_test_result(func_desc, "CPU", cpu_result, log_out_on_pass, cpu_out);
			details::log_run_test_result(func_desc, "GPU", gpu_result, log_out_on_pass, gpu_out);

			return cpu_result & gpu_result;
		}

		#pragma endregion

	}
}

// Evaluates a test expression on the GPU. The expression should be implicitly convertable
// to a runall_result. IOW, the expression should be a valid 'test'.
// e.g. EVALUATE_TEST_ON_CPU_AND_GPU(av, test1());
// e.g. EVALUATE_TEST_ON_CPU_AND_GPU(av, test1<int>());
// e.g. EVALUATE_TEST_ON_CPU_AND_GPU(av, (test1<int, 2>()));  // Note since there's a comma in the template args you must surround the function call with parenthesese.
#define EVALUATE_TEST_ON_CPU_AND_GPU(_av, _expr) Concurrency::Test::run_test_on_cpu_and_gpu(_av, #_expr, [&]() restrict(cpu,amp) { return _expr; })

/*
* This macro can be used to invoke a function (gpui_func) on the GPU and return the result.
* gpui_acclv - The accelerator_view on which to run.
* gpui_func_Tresult - The return type of gpui_func.
*      Advanced: this just needs to be a type that is implicitly castable from the actual return type of gpui_func.
* gpui_func - The function to invoke. The result of this function will be assigned to gpui_result.
*/
#define GPU_INVOKE(gpui_acclv, gpui_func_Tresult, gpui_func, ...) [&]() mutable { \
	gpui_func_Tresult gpui_result; \
	concurrency::array_view<gpui_func_Tresult, 1> gpui_resultv(1, &(gpui_result)); \
    gpui_resultv.discard_data(); \
	concurrency::parallel_for_each(gpui_acclv, gpui_resultv.get_extent(), [=](concurrency::index<1> gpui__idx) restrict(amp) { \
		gpui_resultv[gpui__idx] = (gpui_func)(__VA_ARGS__); \
	}); \
	return gpui_resultv[0]; \
}()


// Evaluates a boolean expression and tests that its result is true. It executes on both the cpu and gpu, reporting the results of each
// and returns a boolean value indicating whether both cpu and gpu evaluations were true.
#define EVALUATE_IS_TRUE_ON_CPU_AND_GPU(_av, _expression) [&]() -> runall_result { \
	/* Invoke on cpu */ \
	bool cpu_result = (_expression); \
	concurrency::Test::report_result("EVALUATE_IS_TRUE: " #_expression " on CPU", cpu_result); \
	\
	/* Invoke on gpu */ \
	int gpu_resulti; \
	concurrency::array_view<int, 1> gpu_resultiv(1, &gpu_resulti); \
    gpu_resultiv.discard_data(); \
	concurrency::parallel_for_each(av, gpu_resultiv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		bool res = (_expression); \
		gpu_resultiv[idx] = res ? 1 : 0; \
	}); \
	bool gpu_result = gpu_resultiv[0] != 0; \
	concurrency::Test::report_result("EVALUATE_IS_TRUE: " #_expression " on GPU", gpu_result); \
	\
	return cpu_result && gpu_result; \
}()




// Invokes a test function returning a test result (i.e. runall_result, bool or runall exit code) on both the cpu and gpu, reporting the results of each.
#define INVOKE_TEST_FUNC_ON_CPU_AND_GPU(_av, _func, ...) [&]() -> runall_result { \
	/* Invoke on cpu */ \
	runall_result cpu_result = _func(__VA_ARGS__); \
	concurrency::Test::report_result(#_func "(" #__VA_ARGS__ ") on CPU", cpu_result); \
	\
	/* Invoke on gpu */ \
	runall_result gpu_result; \
	concurrency::array_view<runall_result, 1> gpu_resultv(1, &gpu_result); \
    gpu_resultv.discard_data(); \
	concurrency::parallel_for_each(av, gpu_resultv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		gpu_resultv[idx] = _func(__VA_ARGS__); \
	}); \
	gpu_resultv.synchronize(); \
	concurrency::Test::report_result(#_func "(" #__VA_ARGS__ ") on GPU", gpu_result); \
	\
	return cpu_result & gpu_result; \
}()



#define INVOKE_TEST_FUNC_ON_CPU_AND_GPU_1T(_av, _func, _T1, ...) [&]() -> runall_result { \
	/* Invoke on cpu */ \
	runall_result cpu_result = _func<_T1>(__VA_ARGS__); \
	concurrency::Test::report_result(#_func "<" #_T1 ">(" #__VA_ARGS__ ") on CPU", cpu_result); \
	\
	/* Invoke on gpu */ \
	runall_result gpu_result; \
	concurrency::array_view<runall_result, 1> gpu_resultv(1, &gpu_result); \
    gpu_resultv.discard_data(); \
	concurrency::parallel_for_each(av, gpu_resultv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		gpu_resultv[idx] = _func<_T1>(__VA_ARGS__); \
	}); \
	gpu_resultv.synchronize(); \
	concurrency::Test::report_result(#_func "<" #_T1 ">(" #__VA_ARGS__ ") on GPU", gpu_result); \
	\
	return cpu_result & gpu_result; \
}()

#define INVOKE_TEST_FUNC_ON_CPU_AND_GPU_2T(_av, _func, _T1, _T2, ...) [&]() -> runall_result { \
	/* Invoke on cpu */ \
	runall_result cpu_result = _func<_T1, _T2>(__VA_ARGS__); \
	concurrency::Test::report_result(#_func "<" #_T1 "," #_T2 ">(" #__VA_ARGS__ ") on CPU", cpu_result); \
	\
	/* Invoke on gpu */ \
	runall_result gpu_result; \
	concurrency::array_view<runall_result, 1> gpu_resultv(1, &gpu_result); \
    gpu_resultv.discard_data(); \
	concurrency::parallel_for_each(av, gpu_resultv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		gpu_resultv[idx] = _func<_T1, _T2>(__VA_ARGS__); \
	}); \
	gpu_resultv.synchronize(); \
	concurrency::Test::report_result(#_func "<" #_T1 "," #_T2 ">(" #__VA_ARGS__ ") on GPU", gpu_result); \
	\
	return cpu_result & gpu_result; \
}()

#define INVOKE_TEST_FUNC_ON_CPU_AND_GPU_3T(_av, _func, _T1, _T2, _T3, ...) [&]() -> runall_result { \
	/* Invoke on cpu */ \
	runall_result cpu_result = _func<_T1, _T2, _T3>(__VA_ARGS__); \
	concurrency::Test::report_result(#_func "<" #_T1 "," #_T2 "," #_T3 ">(" #__VA_ARGS__ ") on CPU", cpu_result); \
	\
	/* Invoke on gpu */ \
	runall_result gpu_result; \
	concurrency::array_view<runall_result, 1> gpu_resultv(1, &gpu_result); \
    gpu_resultv.discard_data(); \
	concurrency::parallel_for_each(av, gpu_resultv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		gpu_resultv[idx] = _func<_T1, _T2, _T3>(__VA_ARGS__); \
	}); \
	gpu_resultv.synchronize(); \
	concurrency::Test::report_result(#_func "<" #_T1 "," #_T2 "," #_T3 ">(" #__VA_ARGS__ ") on GPU", gpu_result); \
	\
	return cpu_result & gpu_result; \
}()

#define INVOKE_TEST_FUNC_ON_CPU_AND_GPU_4T(_av, _func, _T1, _T2, _T3, _T4, ...) [&]() -> runall_result { \
	/* Invoke on cpu */ \
	runall_result cpu_result = _func<_T1, _T2, _T3, _T4>(__VA_ARGS__); \
	concurrency::Test::report_result(#_func "<" #_T1 "," #_T2 "," #_T3 "," #_T4 ">(" #__VA_ARGS__ ") on CPU", cpu_result); \
	\
	/* Invoke on gpu */ \
	runall_result gpu_result; \
	concurrency::array_view<runall_result, 1> gpu_resultv(1, &gpu_result); \
    gpu_resultv.discard_data(); \
	concurrency::parallel_for_each(av, gpu_resultv.get_extent() \
		, [=](concurrency::index<1> idx) restrict(amp) { \
		gpu_resultv[idx] = _func<_T1, _T2, _T3, _T4>(__VA_ARGS__); \
	}); \
	gpu_resultv.synchronize(); \
	concurrency::Test::report_result(#_func "<" #_T1 "," #_T2 "," #_T3 "," #_T4 ">(" #__VA_ARGS__ ") on GPU", gpu_result); \
	\
	return cpu_result & gpu_result; \
}()



