// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include <amptest.h>

class Udt
{
public:
	Udt()
	{
		i = 10;
		l = 20;
		f = 30;
	}

	const Udt operator=(Udt* const obj)
	{
		i = obj->i;
		l = obj->l;
		f = obj->f;

		return *this;
	}

	int i;
	long l;
	float f;
};

bool VerifyUdt(std::vector<Udt> inputVector)
{
	using namespace Concurrency::Test;
	for(std::vector<Udt>::size_type m = 0; m < inputVector.size(); m++)
	{
		if(inputVector[m].i != 10 || inputVector[m].l != 20 || inputVector[m].f != 30)
		{
			Log(LogType::Error, true) << "Data mismatch for element: " << m << std::endl;
			Log(LogType::Error, true) << " i = " << inputVector[m].i << " l = " << inputVector[m].l << " f = " << inputVector[m].f << std::endl;
			return false;
		}
	}

	return true;
}

bool CopyBetweenArrayAndArray(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{

	using namespace Concurrency;
	using namespace Concurrency::Test;

	Concurrency::extent<2> arrayExtent = CreateRandomExtent<2>(5);

	// Create source array
	std::vector<Udt> srcCont(arrayExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), Udt());
	array<Udt, 2> srcArray(arrayExtent, srcCont.begin(), srcDevice.get_default_view());
	Log(LogType::Info, true) << "Created array on source device of " << srcArray.get_extent() << std::endl;

	// Create destination array
	array<Udt, 2> destArray(arrayExtent, destDevice.get_default_view());
	Log(LogType::Info, true) << "Created array on destination device of " << destArray.get_extent()  << std::endl;

	copy(srcArray, destArray);

	std::vector<Udt> temp = destArray;

	// Verify data
	if(!VerifyUdt(temp))
	{
		Log(LogType::Error, true) << "FAILED" << std::endl;
		return false;
	}

	Log(LogType::Info, true) << "PASSED" << std::endl;
	return true;
}

bool CopyBetweenArrayViewAndArrayView(Concurrency::accelerator& srcDevice, Concurrency::accelerator& destDevice)
{
	using namespace Concurrency;
	using namespace Concurrency::Test;

	Concurrency::extent<2> arrayExtent = CreateRandomExtent<2>(5);

	// Create source array view
	std::vector<Udt> srcCont(arrayExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), Udt());
	array<Udt, 2> tempArray(arrayExtent, srcCont.begin(), srcDevice.get_default_view());
	array_view<Udt, 2> srcArrayView(tempArray);
	Log(LogType::Info, true) << "Created array view on source device of " << srcArrayView.get_extent() << std::endl;

	// Create destination array view
	array<Udt, 2> destDataArray(arrayExtent, destDevice.get_default_view());
	array_view<Udt, 2> destArrayView(destDataArray);
	Log(LogType::Info, true) << "Created array view on destination device of " << destArrayView.get_extent() << std::endl;

	copy(srcArrayView, destArrayView);

	std::vector<Udt> temp = destDataArray;

	// Verify data
	if(!VerifyUdt(temp))
	{
		Log(LogType::Error, true) << "FAILED" << std::endl;
		return false;
	}

	Log(LogType::Info, true) << "PASSED" << std::endl;
	return true;
}

bool CopyBetweenArrayViewAndStdCont(Concurrency::accelerator& srcDevice)
{
	using namespace Concurrency;
	using namespace Concurrency::Test;

	Concurrency::extent<2> arrayExtent = CreateRandomExtent<2>(5);

	// Create source array view
	std::vector<Udt> srcCont(arrayExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), Udt());
	array<Udt, 2> tempArray(arrayExtent, srcCont.begin(), srcDevice.get_default_view());
	array_view<Udt, 2> srcArrayView(tempArray);
	Log(LogType::Info, true) << "Created array view on source device of " << srcArrayView.get_extent() << std::endl;

	// Create destination standard container
	std::vector<Udt> destCont(arrayExtent.size());
	Log(LogType::Info, true) << "Created std::vector with size " << destCont.size() << std::endl;

	copy(srcArrayView, destCont.begin());

	// Verify data
	if(!VerifyUdt(destCont))
	{
		Log(LogType::Error, true) << "FAILED" << std::endl;
		return false;
	}

	Log(LogType::Info, true) << "PASSED";
	return true;
}

bool CopyBetweenStdContAndArrayView(Concurrency::accelerator& destDevice)
{
	using namespace Concurrency;
	using namespace Concurrency::Test;

	Concurrency::extent<2> arrayExtent = CreateRandomExtent<2>(5);

	// Create source array view
	std::vector<Udt> srcCont(arrayExtent.size());
		std::fill(srcCont.begin(), srcCont.end(), Udt());
	Log(LogType::Info, true) << "Created std container with size " << srcCont.size() << std::endl;

	// Create destination array view
	array<Udt, 2> destDataArray(arrayExtent, srcCont.begin(), destDevice.get_default_view());
	array_view<Udt, 2> destArrayView(destDataArray);
	Log(LogType::Info, true) << "Created array view on destination device of " << destArrayView.get_extent() << std::endl;

	copy(srcCont.begin(), destArrayView);

	std::vector<Udt> temp = destDataArray;

	// Verify data
	if(!VerifyUdt(temp))
	{
		Log(LogType::Error, true) << "FAILED" << std::endl;
		return false;
	}

	Log(LogType::Info, true) << "PASSED" << std::endl;
	return true;
}
