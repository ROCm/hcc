// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include "amptest.h"
#include "amptest_main.h"
#include <amptest/event.h>
#include <typeinfo>

#define RANGE 40
#define VALUE 10

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayToArray(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	array<_type, _rank> destArray(dataExtent, destDevice.create_view());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArray, destArray).then([&]() {
		Log(LogType::Info, true) << "Verifying destArray" << std::endl;
		if(VerifyAllSameValue(destArray, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayToArrayView(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());

	array<_type, _rank> dataArray(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView(dataArray);

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArray, destArrayView).then([&]() {
		Log(LogType::Info, true) << "Verifying destArrayView" << std::endl;
		if(VerifyAllSameValue(destArrayView, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewToArray(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> dataArray(dataExtent, data.begin(), srcDevice.create_view());
	array<_type, _rank> srcArrayView(dataArray);
	array<_type, _rank> destArray(dataExtent, destDevice.create_view());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArrayView, destArray).then([&]() {
		Log(LogType::Info, true) << "Verifying destArray" << std::endl;
		if(VerifyAllSameValue(destArray, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewConstToArray(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> dataArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view<const _type, _rank> srcArrayView(dataArray);

	array<_type, _rank> destArray(dataExtent, destDevice.create_view());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArrayView, destArray).then([&]() {
		Log(LogType::Info, true) << "Verifying destArray" << std::endl;
		if(VerifyAllSameValue(destArray, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewToArrayView(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view< _type, _rank> srcArrayView(srcArray);

	array<_type, _rank> dataArray(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView(dataArray);

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArrayView, destArrayView).then([&]() {
		Log(LogType::Info, true) << "Verifying destArrayView" << std::endl;
		if(VerifyAllSameValue(destArrayView, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayToIter(Concurrency::accelerator& srcDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());

	std::vector<_type> destCont(dataExtent.size());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArray, destCont.begin()).then([&]() {
		Log(LogType::Info, true) << "Verifying destCont" << std::endl;
		if(VerifyAllSameValue(destCont, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewConstToArrayView(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view<const _type, _rank> srcArrayView(srcArray);

	array<_type, _rank> dataArray(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView(dataArray);

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArrayView, destArrayView).then([&]() {
		Log(LogType::Info, true) << "Verifying destArrayView" << std::endl;
		if(VerifyAllSameValue(destArrayView, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIterToArray(Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> srcCont(dataExtent.size(), expected_value);
	array<_type, _rank> destArray(dataExtent, destDevice.create_view());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcCont.begin(), destArray).then([&]() {
		Log(LogType::Info, true) << "Verifying destArray" << std::endl;
		if(VerifyAllSameValue(destArray, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIter2ToArray(Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> srcCont(dataExtent.size(), expected_value);
	array<_type, _rank> destArray(dataExtent, destDevice.create_view());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcCont.begin(), srcCont.end(), destArray).then([&]() {
		Log(LogType::Info, true) << "Verifying destArray" << std::endl;
		if(VerifyAllSameValue(destArray, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewToIter(Concurrency::accelerator& srcDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> data(dataExtent.size(), expected_value);
	array<_type, _rank> dataArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view<_type, _rank> srcArrayView(dataArray);

	std::vector<_type> destCont(dataExtent.size());

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcArrayView, destCont.begin()).then([&]() {
		Log(LogType::Info, true) << "Verifying destConst" << std::endl;
		if(VerifyAllSameValue(destCont, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIterToArrayView(Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> srcCont(dataExtent.size(), expected_value);
	array<_type, _rank> dataArray(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView(dataArray);

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcCont.begin(), destArrayView).then([&]() {
		Log(LogType::Info, true) << "Verifying destArrayView" << std::endl;
		if(VerifyAllSameValue(destArrayView, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIter2ToArrayView(Concurrency::accelerator& destDevice)
{
	using namespace concurrency::Test;

	extent<_rank> dataExtent = CreateRandomExtent<_rank>(RANGE);
	static const _type expected_value = static_cast<_type>(VALUE);

	std::vector<_type> srcCont(dataExtent.size(), expected_value);
	array<_type, _rank> dataArray(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView(dataArray);

	long flag = 0;

	// Wait event set when continuation finishes verification.
	event waitEvent;

	copy_async(srcCont.begin(), srcCont.end(), destArrayView).then([&]() {
		Log(LogType::Info, true) << "Verifying destArrayView" << std::endl;
		if(VerifyAllSameValue(destArrayView, expected_value) == -1)
		{
			flag = 1;
		}

		waitEvent.set();
	});

	waitEvent.wait();
	return (flag == 1);
}