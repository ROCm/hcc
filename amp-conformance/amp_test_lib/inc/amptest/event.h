#pragma once
#include <atomic>

#ifndef AMP_TEST_PLATFORM_MSVC
namespace Concurrency {
	namespace Test {
		class event
		{
		public:
			event() : flag(false) {}
			void set() { flag = true; }
			void wait() { while(!flag){} }

		private:
			std::atomic<bool> flag;
		};
	} // namespace Test
} // namespace concurrency
#endif