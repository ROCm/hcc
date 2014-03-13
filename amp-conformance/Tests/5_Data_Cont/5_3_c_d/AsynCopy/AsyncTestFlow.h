// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include "amptest.h"
#include "amptest_main.h"
#include <typeinfo>

//
//**************************************** BEGIN: Copy between data containers *************************************
//

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayToArray(concurrency::accelerator& srcDevice, concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayToArray(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	array<_type, _rank> destArray1(dataExtent, destDevice.create_view());
	array<_type, _rank> destArray2(dataExtent, destDevice.create_view());
	
	std::shared_future<void> w1 = copy_async(srcArray, destArray1);
	std::shared_future<void> w2 = copy_async(srcArray, destArray2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl;
	if(VerifyAllSameValue(srcArray, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray1" << std::endl;
	if(VerifyAllSameValue(destArray1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray2" << std::endl;
	if(VerifyAllSameValue(destArray2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayToArrayView(concurrency::accelerator& srcDevice, concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayToArrayView(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	
	array<_type, _rank> dataArray1(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView1(dataArray1);
	
	array<_type, _rank> dataArray2(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView2(dataArray2);
	
	std::shared_future<void> w1 = copy_async(srcArray, destArrayView1);
	std::shared_future<void> w2 = copy_async(srcArray, destArrayView2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl << std::endl;
	if(VerifyAllSameValue(srcArray, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED";
		return false;
	}
	
	Log() << "Verifying destArrayView1" << std::endl;
	if(VerifyAllSameValue(destArrayView1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView2" << std::endl;
	if(VerifyAllSameValue(destArrayView2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewToArray(concurrency::accelerator& srcDevice, concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayViewToArray(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> dataArray(dataExtent, data.begin(), srcDevice.create_view());
	array<_type, _rank> srcArrayView(dataArray);
	
	array<_type, _rank> destArray1(dataExtent, destDevice.create_view());
	array<_type, _rank> destArray2(dataExtent, destDevice.create_view());
	
	std::shared_future<void> w1 = copy_async(srcArrayView, destArray1);
	std::shared_future<void> w2 = copy_async(srcArrayView, destArray2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl;
	if(VerifyAllSameValue(srcArrayView, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray1" << std::endl;
	if(VerifyAllSameValue(destArray1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray2" << std::endl;
	if(VerifyAllSameValue(destArray2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewConstToArray(concurrency::accelerator& srcDevice, concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayViewConstToArray(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> dataArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view<const _type, _rank> srcArrayView(dataArray);
	
	array<_type, _rank> destArray1(dataExtent, destDevice.create_view());
	array<_type, _rank> destArray2(dataExtent, destDevice.create_view());
	
	std::shared_future<void> w1 = copy_async(srcArrayView, destArray1);
	std::shared_future<void> w2 = copy_async(srcArrayView, destArray2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl;
	if(VerifyAllSameValue(dataArray, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray1" << std::endl;
	if(VerifyAllSameValue(destArray1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray2" << std::endl;
	if(VerifyAllSameValue(destArray2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewToArrayView(concurrency::accelerator& srcDevice, concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayViewToArrayView(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view< _type, _rank> srcArrayView(srcArray);
	
	array<_type, _rank> dataArray1(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView1(dataArray1);
	
	array<_type, _rank> dataArray2(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView2(dataArray2);
	
	std::shared_future<void> w1 = copy_async(srcArrayView, destArrayView1);
	std::shared_future<void> w2 = copy_async(srcArrayView, destArrayView2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl;
	if(VerifyAllSameValue(srcArray, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView1" << std::endl;
	if(VerifyAllSameValue(destArrayView1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView2" << std::endl;
	if(VerifyAllSameValue(destArrayView2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayToIter(concurrency::accelerator& srcDevice)
{
	LogStream() << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayToIter(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	
	std::vector<_type> destCont1(dataExtent.size());
	std::vector<_type> destCont2(dataExtent.size());
	
	std::shared_future<void> w1 = copy_async(srcArray, destCont1.begin());
	std::shared_future<void> w2 = copy_async(srcArray, destCont2.begin());
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl;
	if(VerifyAllSameValue(srcArray, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destCont1" << std::endl;
	if(VerifyAllSameValue(destCont1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destCont2" << std::endl;
	if(VerifyAllSameValue(destCont2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewConstToArrayView(concurrency::accelerator& srcDevice, concurrency::accelerator& destDevice)
{
	LogStream() << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayViewConstToArrayView(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> srcArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view<const _type, _rank> srcArrayView(srcArray);
	
	array<_type, _rank> dataArray1(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView1(dataArray1);
	
	array<_type, _rank> dataArray2(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView2(dataArray2);
	
	std::shared_future<void> w1 = copy_async(srcArrayView, destArrayView1);
	std::shared_future<void> w2 = copy_async(srcArrayView, destArrayView2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArray" << std::endl;
	if(VerifyAllSameValue(srcArray, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView1" << std::endl;
	if(VerifyAllSameValue(destArrayView1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView2" << std::endl;
	if(VerifyAllSameValue(destArrayView2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIterToArray(concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyIterToArray(...)" << std::endl;
	Log() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> srcCont(dataExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), expected_value);
		
	array<_type, _rank> destArray1(dataExtent, destDevice.create_view());
	array<_type, _rank> destArray2(dataExtent, destDevice.create_view());
	
	std::shared_future<void> w1 = copy_async(srcCont.begin(), destArray1);
	std::shared_future<void> w2 = copy_async(srcCont.begin(), destArray2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcCont" << std::endl;
	if(VerifyAllSameValue(srcCont, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray1" << std::endl;
	if(VerifyAllSameValue(destArray1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray2" << std::endl;
	if(VerifyAllSameValue(destArray2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIter2ToArray(concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyIter2ToArray(...)" << std::endl;
	Log() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> srcCont(dataExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), expected_value);
		
	array<_type, _rank> destArray1(dataExtent, destDevice.create_view());
	array<_type, _rank> destArray2(dataExtent, destDevice.create_view());
	
	std::shared_future<void> w1 = copy_async(srcCont.begin(), srcCont.end(), destArray1);
	std::shared_future<void> w2 = copy_async(srcCont.begin(), srcCont.end(), destArray2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcCont" << std::endl;
	if(VerifyAllSameValue(srcCont, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray1" << std::endl;
	if(VerifyAllSameValue(destArray1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArray2" << std::endl;
	if(VerifyAllSameValue(destArray2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyArrayViewToIter(concurrency::accelerator& srcDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyArrayViewToIter(...)" << std::endl;
	Log() << "Source device: " << srcDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> data(dataExtent.size());
	std::fill(data.begin(), data.end(), expected_value);
	array<_type, _rank> dataArray(dataExtent, data.begin(), srcDevice.create_view());
	array_view<_type, _rank> srcArrayView(dataArray);
	
	std::vector<_type> destCont1(dataExtent.size());
	std::vector<_type> destCont2(dataExtent.size());
	
	std::shared_future<void> w1 = copy_async(srcArrayView, destCont1.begin());
	std::shared_future<void> w2 = copy_async(srcArrayView, destCont2.begin());
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcArrayView" << std::endl;
	if(VerifyAllSameValue(srcArrayView, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destCont1" << std::endl;
	if(VerifyAllSameValue(destCont1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destCont2" << std::endl;
	if(VerifyAllSameValue(destCont2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIterToArrayView(concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyIterToArrayView(...)" << std::endl;
	Log() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> srcCont(dataExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), expected_value);
		
	array<_type, _rank> dataArray1(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView1(dataArray1);

	array<_type, _rank> dataArray2(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView2(dataArray2);
	
	std::shared_future<void> w1 = copy_async(srcCont.begin(), destArrayView1);
	std::shared_future<void> w2 = copy_async(srcCont.begin(), destArrayView2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcCont" << std::endl;
	if(VerifyAllSameValue(srcCont, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView1" << std::endl;
	if(VerifyAllSameValue(destArrayView1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView2" << std::endl;
	if(VerifyAllSameValue(destArrayView2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

template<typename _type, int _rank>
bool AsyncCopyAndVerifyIter2ToArrayView(concurrency::accelerator& destDevice)
{
	LogStream() << std::endl << std::endl;
	Log() << "Invoking AsyncCopyAndVerifyIter2ToArrayView(...)" << std::endl;
	Log() << " Destination device: " << destDevice.get_device_path() << std::endl;
	Log() << "type: " << get_type_name<_type>() << std::endl;
	
	extent<_rank> dataExtent = CreateRandomExtent<_rank>(5);
	static const _type expected_value = (_type)10;
	
	std::vector<_type> srcCont(dataExtent.size());
	std::fill(srcCont.begin(), srcCont.end(), expected_value);
		
	array<_type, _rank> dataArray1(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView1(dataArray1);

	array<_type, _rank> dataArray2(dataExtent, destDevice.create_view());
	array_view<_type, _rank> destArrayView2(dataArray2);
	
	std::shared_future<void> w1 = copy_async(srcCont.begin(), srcCont.end(), destArrayView1);
	std::shared_future<void> w2 = copy_async(srcCont.begin(), srcCont.end(), destArrayView2);
	
	w1.wait();
	w2.wait();
	
	Log() << "Verifying srcCont" << std::endl;
	if(VerifyAllSameValue(srcCont, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView1" << std::endl;
	if(VerifyAllSameValue(destArrayView1, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
	
	Log() << "Verifying destArrayView2" << std::endl;
	if(VerifyAllSameValue(destArrayView2, expected_value) != -1)
	{
		Log(LogType::Error) << "FAILED" << std::endl;
		return false;
	}
		
	Log() << "PASSED" << std::endl;
	return true;
}

//
//**************************************** END: Copy between data containers *************************************
//






