// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#pragma once

#include <amptest.h>
#include <amptest/coordinates.h>
#include <map>

#define MODIFY_VALUE 5.5f
#define INIT_VALUE 0xDEAD

#pragma region utility methods

int get_max_dim(int rank)
{
	if(rank == 1) { return 512; }
	if(rank == 2) { return 256; }
	if(rank == 3) { return 128; }

	return 20;
}

typedef std::vector<std::tuple<concurrency::access_type, concurrency::access_type>> access_list;

void print_access_type_tuple(std::tuple<concurrency::access_type, concurrency::access_type>& tup)
{
	Concurrency::Test::Log(Concurrency::Test::LogType::Info, true) << "CPU Access Types: (" << std::get<0>(tup) << ", " << std::get<1>(tup) << ")" << std::endl;
}

void compute_access_type_list(access_list& access_types_vec, concurrency::accelerator& gpu_acc, concurrency::access_type def_acc_type)
{
	if(gpu_acc.get_supports_cpu_shared_memory())
	{
		//Concurrency::Test::WLog(LogType::Info, true) << "Accelerator " << gpu_acc.get_description() << " supports zero copy" << std::endl;

		// Set the default cpu access type for this accelerator
		gpu_acc.set_default_cpu_access_type(def_acc_type);

		concurrency::access_type a_t_list[] = { concurrency::access_type_none,
												concurrency::access_type_read,
												concurrency::access_type_write,
												concurrency::access_type_read_write };

		for(int i = 0; i < 4; i++)
		{
			for(int j = 0; j < 4; j++)
			{
				access_types_vec.push_back(std::make_tuple(a_t_list[i], a_t_list[j]));
			}
		}
	}
	else
	{
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_auto, concurrency::access_type_auto));
	}
}

void compute_access_type_list(
				access_list& access_types_vec,
				concurrency::accelerator& gpu_acc1,
				concurrency::accelerator& gpu_acc2,
				concurrency::access_type def_acc_type1,
				concurrency::access_type def_acc_type2)
{
	if(!gpu_acc1.get_supports_cpu_shared_memory() && !gpu_acc2.get_supports_cpu_shared_memory())
	{
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_auto, concurrency::access_type_auto));
	}
	else if(gpu_acc1.get_supports_cpu_shared_memory() && !gpu_acc2.get_supports_cpu_shared_memory())
	{
		//Concurrency::Test::WLog(LogType::Info, true) << "Accelerator " << gpu_acc1.get_description() << " supports zero copy" << std::endl;

		// Set the default cpu access type for this accelerator
		gpu_acc1.set_default_cpu_access_type(def_acc_type1);

		access_types_vec.push_back(std::make_tuple(concurrency::access_type_none, concurrency::access_type_auto));
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_read, concurrency::access_type_auto));
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_write, concurrency::access_type_auto));
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_read_write, concurrency::access_type_auto));
	}
	else if(!gpu_acc1.get_supports_cpu_shared_memory() && gpu_acc2.get_supports_cpu_shared_memory())
	{
		//Concurrency::Test::WLog(LogType::Info, true) << "Accelerator " << gpu_acc2.get_description() << " supports zero copy" << std::endl;

		// Set the default cpu access type for this accelerator
		gpu_acc2.set_default_cpu_access_type(def_acc_type2);

		access_types_vec.push_back(std::make_tuple(concurrency::access_type_auto, concurrency::access_type_auto));
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_auto, concurrency::access_type_read));
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_auto, concurrency::access_type_write));
		access_types_vec.push_back(std::make_tuple(concurrency::access_type_auto, concurrency::access_type_read_write));
	}
	else
	{
		//Concurrency::Test::WLog(LogType::Info, true) << "Accelerator " << gpu_acc1.get_description() << " supports zero copy" << std::endl;
		//Concurrency::Test::WLog(LogType::Info, true) << "Accelerator " << gpu_acc2.get_description() << " supports zero copy" << std::endl;

		// Set the default cpu access type for these accelerators
		gpu_acc1.set_default_cpu_access_type(def_acc_type1);
		gpu_acc2.set_default_cpu_access_type(def_acc_type2);

		concurrency::access_type a_t_list[] = { concurrency::access_type_none,
												concurrency::access_type_read,
												concurrency::access_type_write,
												concurrency::access_type_read_write };

		for(int i = 0; i < 4; i++)
		{
			for(int j = 0; j < 4; j++)
			{
				access_types_vec.push_back(std::make_tuple(a_t_list[i], a_t_list[j]));
			}
		}
	}
}


// We cannot invoke p_f_e for an accelerator_view on CPU. This method is invoked
// to modify array and array_view on accelerator_view(s) on CPU
template<typename _type, int _rank, template<typename, int> class _amp_container_type>
void ModifyOnCpu(_amp_container_type<_type, _rank>& amp_container, _type value)
{
	using namespace concurrency::Test;

	index_iterator<_rank> idx_iter(amp_container.get_extent());

	for(index_iterator<_rank> iter = idx_iter.begin(); iter != idx_iter.end(); iter++)
	{
		amp_container[*iter] += value;
	}
}

template<typename _type, int _rank>
void ModifyOnAcceleratorView(const concurrency::accelerator_view& av, concurrency::array<_type, _rank>& arr, _type value)
{
	// If the accelerator view on CPU, we cannot invoke p_f_e
	if(av.get_accelerator().get_device_path() == concurrency::accelerator::cpu_accelerator)
	{
		ModifyOnCpu<_type, _rank, array>(arr, value);
	}
	else
	{
		parallel_for_each(av, arr.get_extent(), [&,value](index<_rank> idx) restrict(amp)
		{
			arr[idx] += value;
		});
	}
}

template<typename _type, int _rank>
void ModifyOnAcceleratorView(const concurrency::accelerator_view& av, concurrency::array_view<_type, _rank>& arr_v, _type value)
{
	// If the accelerator view on CPU cannot invoke p_f_e
	if(av.get_accelerator().get_device_path() == concurrency::accelerator::cpu_accelerator)
	{
		ModifyOnCpu<_type, _rank, array_view>(arr_v, value);
	}
	else
	{
		parallel_for_each(av, arr_v.get_extent(), [=](index<_rank> idx) restrict(amp)
		{
			arr_v[idx] += value;
		});
	}
}

#pragma endregion

#pragma region copy from array<T, N> methods

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayToArray(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
								concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);

	// Copy: array (src_av) -> target array (target_av)
	array<_type, _rank> target_arr(src_arr.get_extent(), target_av, target_access_type);
	copy(src_arr, target_arr);

	// Modify target array on target accelerator view
	ModifyOnAcceleratorView(target_av, target_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array (target_av) -> array (src_av)
	array<_type, _rank> dest_arr(src_arr.get_extent(), src_av, dest_access_type);
	copy(target_arr, dest_arr);

	return VerifyDataOnCpu(src_arr, dest_arr, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayToArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);

	// Copy: array (src_av) -> target array_view (target_av)
	array<_type, _rank> target_data_arr(src_arr.get_extent(), target_av, target_access_type);
	array_view<_type, _rank> target_arr_v(target_data_arr);
	copy(src_arr, target_arr_v);

	// Modify target array_view on target accelerator view
	ModifyOnAcceleratorView(target_av, target_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array_view (target_av) -> array (src_av)
	array<_type, _rank> dest_arr(src_arr.get_extent(), src_av, dest_access_type);
	copy(target_arr_v, dest_arr);

	return VerifyDataOnCpu(src_arr, dest_arr, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayToStagingArray(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_arr = CreateArrayAndFillData<_type, _rank>(arr_av, get_max_dim(_rank), src_access_type);

	// Copy: array (src_av) -> staging array (stg_arr_av)
	array<_type, _rank> stg_arr(src_arr.get_extent(), cpu_av, stg_arr_av);
	copy(src_arr, stg_arr);

	// Modify staging array on cpu accelertor_view
	ModifyOnAcceleratorView(cpu_av, stg_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: staging array (stg_arr_av) -> array (src_av)
	array<_type, _rank> dest_arr(src_arr.get_extent(), arr_av, dest_access_type);
	copy(stg_arr, dest_arr);

	return VerifyDataOnCpu(src_arr, dest_arr, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayToNonContiguousArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	array<_type, _rank> data_arr = CreateArrayAndFillData<_type, _rank>(target_av, get_max_dim(_rank), target_access_type);
	array_view<_type, _rank> non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(data_arr);

	// Create source data
	std::vector<_type> data(non_contig_arr_v.get_extent().size(), static_cast<_type>(INIT_VALUE));
	array<_type, _rank> src_arr(non_contig_arr_v.get_extent(), data.begin(), src_av, src_access_type);

	// Copy: array (src_av) -> non-contiguous array_view (target_av)
	copy(src_arr, non_contig_arr_v);

	// Modify non-contiguous array_view on target_av
	ModifyOnAcceleratorView(target_av, non_contig_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: non-contiguous array_view (target_av) -> array (src_av)
	array<_type, _rank> dest_arr(src_arr.get_extent(), src_av, dest_access_type);
	copy(non_contig_arr_v, dest_arr);

	return VerifyDataOnCpu(src_arr, dest_arr, static_cast<_type>(MODIFY_VALUE));
}

#pragma endregion

#pragma region copy from array_view<T, N> methods

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewToArray(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> target array (target_av)
	array<_type, _rank> target_arr(src_arr_v.get_extent(), target_av, target_access_type);
	copy(src_arr_v, target_arr);

	// Modify target array on target accelerator_view
	ModifyOnAcceleratorView(target_av, target_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array (target_av) -> array_view (src_av)
	array<_type, _rank> dest_data_arr(src_arr_v.get_extent(), src_av, dest_access_type);
	array_view<_type, _rank> dest_arr_v(dest_data_arr);
	copy(target_arr, dest_arr_v);

	return VerifyDataOnCpu(src_arr_v, dest_arr_v, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewToArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> target array_view (target_av)
	array<_type, _rank> target_data_arr(src_arr_v.get_extent(), target_av, target_access_type);
	array_view<_type, _rank> target_arr_v(target_data_arr);
	copy(src_arr_v, target_arr_v);

	// Modify target array_view on target accelerator_view
	ModifyOnAcceleratorView(target_av, target_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array_view (target_av) -> array_view (src_av)
	array<_type, _rank> dest_data_arr(src_arr_v.get_extent(), src_av, dest_access_type);
	array_view<_type, _rank> dest_arr_v(dest_data_arr);
	copy(target_arr_v, dest_arr_v);

	return VerifyDataOnCpu(src_arr_v, dest_arr_v, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank, template<typename T, typename=std::allocator<T>> class _stl_cont>
bool CopyAndVerifyFromArrayViewToIterator(const concurrency::accelerator_view& av,   concurrency::access_type src_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_arr_v(src_data_arr);

	// Copy to STL container
	_stl_cont<_type> src_stl_cont(src_arr_v.get_extent().size(), static_cast<_type>(_rank));
	copy(src_arr_v, src_stl_cont.begin());

	return VerifyDataOnCpu(src_arr_v, src_stl_cont);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewToStagingArray(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_v_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(arr_v_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> staging array (stg_arr_av)
	array<_type, _rank> stg_arr(src_arr_v.get_extent(), cpu_av, stg_arr_av);
	copy(src_arr_v, stg_arr);

	// Modify staging array on cpu accelerator_view
	ModifyOnAcceleratorView(cpu_av, stg_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: staging array (stg_arr_av) -> array_view (src_av)
	array<_type, _rank> dest_data_arr(src_arr_v.get_extent(), arr_v_av, dest_access_type);
	array_view<_type, _rank> dest_arr_v(dest_data_arr);
	copy(stg_arr, dest_arr_v);

	return VerifyDataOnCpu(src_arr_v, dest_arr_v, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewToNonContiguousArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type,
							  	concurrency::access_type dest_access_type)
{
	using namespace concurrency::Test;

	array<_type, _rank> data_arr = CreateArrayAndFillData<_type, _rank>(target_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(data_arr);

	// Create source data
	std::vector<_type> data(non_contig_arr_v.get_extent().size(), static_cast<_type>(INIT_VALUE));
	array<_type, _rank> src_data_arr(non_contig_arr_v.get_extent(), data.begin(), src_av, target_access_type);
	array_view<_type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> non-contiguous array_view (target_av)
	copy(src_arr_v, non_contig_arr_v);

	// Modify non-contiguous array_view on target accelerator_view
	ModifyOnAcceleratorView(target_av, non_contig_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: non-contiguous array_view (target_av) -> array_view (src_av)
	array<_type, _rank> dest_data_arr(non_contig_arr_v.get_extent(), src_av, dest_access_type);
	array_view<_type, _rank> dest_arr_v(dest_data_arr);
	copy(non_contig_arr_v, dest_arr_v);

	return VerifyDataOnCpu(src_arr_v, dest_arr_v, static_cast<_type>(MODIFY_VALUE));
}

#pragma endregion

#pragma region copy from non-contiguous array_view<T, N> methods
// Copy from non-contiguous array_view only copies from source accelerator_view
// to target accelerator_view. The reverse is not done as it is covered in other
// tests. For e.g. In function CopyAndVerifyFromNonContigArrayViewToArray(), the reverse is
// already covered in test function CopyAndVerifyFromArrayToNonContiguousArrayView().
// The function CopyAndVerifyFromArrayToNonContiguousArrayView() does not cover copying
// from a freashly created non-contiguous array_view


template<typename _type, int _rank>
bool CopyAndVerifyFromNonContigArrayViewToArray(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(src_data_arr);

	// Copy: non-contiguous array_view (src_av) -> array (target_av)
	array<_type, _rank> target_arr(src_non_contig_arr_v.get_extent(), target_av, target_access_type);
	copy(src_non_contig_arr_v, target_arr);

	return VerifyDataOnCpu(src_non_contig_arr_v, target_arr);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromNonContigArrayViewToArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(src_data_arr);

	// Copy: non-contiguous array_view (src_av) -> array_view (target_av)
	array<_type, _rank> target_data_arr(src_non_contig_arr_v.get_extent(), target_av, target_access_type);
	array_view<_type, _rank> target_arr_v(target_data_arr);
	copy(src_non_contig_arr_v, target_arr_v);

	return VerifyDataOnCpu(src_non_contig_arr_v, target_arr_v);
}

template<typename _type, int _rank, template<typename T, typename=std::allocator<T>> class _stl_cont>
bool CopyAndVerifyFromNonContigArrayViewToIterator(const concurrency::accelerator_view& av,   concurrency::access_type src_access_type)
{

	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(src_data_arr);

	// Copy: non-contiguous array_view (src_av) -> target STL container
	_stl_cont<_type> target_stl_cont(src_non_contig_arr_v.get_extent().size(), static_cast<_type>(_rank));
	copy(src_non_contig_arr_v, target_stl_cont.begin());

	return VerifyDataOnCpu(src_non_contig_arr_v, target_stl_cont);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromNonContigArrayViewToStagingArray(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_v_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type src_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(arr_v_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(src_data_arr);

	// Copy: non-contiguous array_view (src_av) -> staging array (stg_arr_v)
	array<_type, _rank> target_stg_arr(src_non_contig_arr_v.get_extent(), cpu_av, stg_arr_av);
	copy(src_non_contig_arr_v, target_stg_arr);

	return VerifyDataOnCpu(src_non_contig_arr_v, target_stg_arr);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromNonContigArrayViewToNonContigArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<_type, _rank> src_non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(src_data_arr);

	index<_rank> idx;
	for(int i = 0; i < _rank; i++) { idx[i] = 0; }

	// Copy: non-contiguous array_view (src_av) -> staging array (stg_arr_v)
	array<_type, _rank> target_data_arr(src_non_contig_arr_v.get_extent(), target_av, target_access_type);
	array_view<_type, _rank> target_non_contig_arr_v = target_data_arr.section(idx, src_non_contig_arr_v.get_extent());
	copy(src_non_contig_arr_v, target_non_contig_arr_v);

	return VerifyDataOnCpu(src_non_contig_arr_v, target_non_contig_arr_v);
}

#pragma endregion

#pragma region copy from array_view<const T, N> methods

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewConstToArray(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<const _type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> target array (target_av)
	array<_type, _rank> target_arr(src_arr_v.get_extent(), target_av, target_access_type);
	copy(src_arr_v, target_arr);

	return VerifyDataOnCpu(src_arr_v, target_arr);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewConstToArrayView(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<const _type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> target array_view (target_av)
	array<_type, _rank> target_data_arr(src_arr_v.get_extent(), target_av, target_access_type);
	array_view<_type, _rank> target_arr_v(target_data_arr);
	copy(src_arr_v, target_arr_v);

	return VerifyDataOnCpu(src_arr_v, target_arr_v);
}

template<typename _type, int _rank, template<typename U, typename=std::allocator<U>> class _stl_cont>
bool CopyAndVerifyFromArrayViewConstToIterator(const concurrency::accelerator_view& av,   concurrency::access_type src_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(av, get_max_dim(_rank), src_access_type);
	array_view<const _type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> target STL container
	_stl_cont<_type> target_stl_cont(src_arr_v.get_extent().size());
	copy(src_arr_v, target_stl_cont.begin());

	return VerifyDataOnCpu(src_arr_v, target_stl_cont);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromArrayViewConstToStagingArray(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_v_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type src_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(arr_v_av, get_max_dim(_rank), src_access_type);
	array_view<const _type, _rank> src_arr_v(src_data_arr);

	// Copy: array_view (src_av) -> target staging array (stg_arr_av)
	array<_type, _rank> target_stg_arr(src_arr_v.get_extent(), cpu_av, stg_arr_av);
	copy(src_arr_v, target_stg_arr);

	return VerifyDataOnCpu(src_arr_v, target_stg_arr);
}

template<typename _type, int _rank>
bool CopyAndVerifyFromNonContigArrayViewConstToArray(
								const concurrency::accelerator_view& src_av,
								const concurrency::accelerator_view& target_av,
							  	concurrency::access_type src_access_type,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_data_arr = CreateArrayAndFillData<_type, _rank>(src_av, get_max_dim(_rank), src_access_type);
	array_view<const _type, _rank> src_non_contig_arr_v = CreateNonContiguousArrayViewWithConstType<_type, _rank>(src_data_arr);

	// Copy: non-contiguous array_view (src_av) -> target array (stg_arr_av)
	array<_type, _rank> target_arr(src_non_contig_arr_v.get_extent(), target_av, target_access_type);
	copy(src_non_contig_arr_v, target_arr);

	return VerifyDataOnCpu(src_non_contig_arr_v, target_arr);
}

#pragma endregion

#pragma region copy from iterator methods

template<typename _type, int _rank, template<typename U, typename=std::allocator<U>> class _stl_cont>
bool CopyAndVerifyBetweenArrayAndIterator(const concurrency::accelerator_view& target_av,   concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	concurrency::extent<_rank> arr_extent = CreateRandomExtent<_rank>(get_max_dim(_rank));
	Log(LogType::Info, true) << "arr_extent = " << arr_extent << std::endl;

	// Create source data
	_stl_cont<_type> src_stl_cont(arr_extent.size(), static_cast<_type>(INIT_VALUE));

	// Copy: STL container -> target array (target_av)
	array<_type, _rank> target_arr(arr_extent, target_av, target_access_type);
	copy(src_stl_cont.begin(), target_arr);

	// Modify target array on target_av
	ModifyOnAcceleratorView(target_av, target_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array (target_av) -> STL container
	_stl_cont<_type> dest_stl_cont(arr_extent.size());
	copy(target_arr, dest_stl_cont.begin());

	return concurrency::Test::Equal(src_stl_cont.begin(),  src_stl_cont.end(),  dest_stl_cont.begin(), concurrency::Test::Difference<_type>(static_cast<_type>(MODIFY_VALUE)));
}

template<typename _type, int _rank, template<typename U, typename=std::allocator<U>> class _stl_cont>
bool CopyAndVerifyFromIteratorToArrayView(const concurrency::accelerator_view& target_av,   concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	concurrency::extent<_rank> arr_v_extent = CreateRandomExtent<_rank>(get_max_dim(_rank));
	Log(LogType::Info, true) << "arr_extent = " << arr_v_extent << std::endl;

	// Create source data
	_stl_cont<_type> src_stl_cont(arr_v_extent.size(), static_cast<_type>(_rank));

	// Copy: STL container -> target array_view (target_av)
	array<_type, _rank> target_data_arr(arr_v_extent, target_av, target_access_type);
	array_view<_type, _rank> target_arr_v(target_data_arr);
	concurrency::copy(src_stl_cont.begin(),src_stl_cont.end(), target_arr_v);

	// Modify target array_view on target_av
	ModifyOnAcceleratorView(target_av, target_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: STL container -> target array_view (target_av)
	_stl_cont<_type> dest_stl_cont(arr_v_extent.size());
	copy(target_arr_v, dest_stl_cont.begin());

	return concurrency::Test::Equal(src_stl_cont.begin(),  src_stl_cont.end(),  dest_stl_cont.begin(), concurrency::Test::Difference<_type>(static_cast<_type>(MODIFY_VALUE)));
}

template<typename _type, int _rank, template<typename U, typename=std::allocator<U>> class _stl_cont>
bool CopyAndVerifyFromIteratorToNonContigArrayView(const concurrency::accelerator_view& target_av,   concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	array<_type, _rank> target_data_arr = CreateArrayAndFillData<_type, _rank>(target_av, get_max_dim(_rank), target_access_type);
	array_view<_type, _rank> target_non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(target_data_arr);

	// Create source data
	_stl_cont<_type> src_stl_cont(target_non_contig_arr_v.get_extent().size(), static_cast<_type>(_rank));

	// Copy: STL container -> target non-contiguous array_view (target_av)
	concurrency::copy(src_stl_cont.begin(),src_stl_cont.end(), target_non_contig_arr_v);

	// Modify target non-contiguous array_view on target_av
	ModifyOnAcceleratorView(target_av, target_non_contig_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: target non-contiguous array_view (target_av) -> STL container
	_stl_cont<_type> dest_stl_cont(src_stl_cont.size());
	copy(target_non_contig_arr_v, dest_stl_cont.begin());

	return concurrency::Test::Equal(src_stl_cont.begin(),  src_stl_cont.end(),  dest_stl_cont.begin(), concurrency::Test::Difference<_type>(static_cast<_type>(MODIFY_VALUE)));
}

template<typename _type, int _rank, template<typename U, typename=std::allocator<U>> class _stl_cont>
bool CopyAndVerifyBetweenStagingArrayAndIterator(const concurrency::accelerator_view& cpu_av, const concurrency::accelerator_view& stg_arr_av)
{
	using namespace concurrency::Test;

	concurrency::extent<_rank> arr_extent = CreateRandomExtent<_rank>(get_max_dim(_rank));
	Log(LogType::Info, true) << "arr_extent = " << arr_extent << std::endl;

	// Create source data
	_stl_cont<_type> src_stl_cont(arr_extent.size(), static_cast<_type>(_rank));

	// Copy: STL container -> target staging array (stg_arr_av)
	array<_type, _rank> target_stg_arr(arr_extent, cpu_av, stg_arr_av);
	concurrency::copy(src_stl_cont.begin(), src_stl_cont.end(), target_stg_arr);

	// Modify target staging array on cpu accelerator_view
	ModifyOnAcceleratorView(cpu_av, target_stg_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: target staging array (stg_arr_av) -> STL container
	_stl_cont<_type> dest_stl_cont(arr_extent.size());
	copy(target_stg_arr, dest_stl_cont.begin());

	return concurrency::Test::Equal(src_stl_cont.begin(),  src_stl_cont.end(),  dest_stl_cont.begin(), concurrency::Test::Difference<_type>(static_cast<_type>(MODIFY_VALUE)));
}

#pragma endregion

#pragma region copy from staging array<T, N> methods

template<typename _type, int _rank>
bool CopyAndVerifyFromStagingArrayToArray(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_stg_arr = CreateStagingArrayAndFillData<_type, _rank>(cpu_av, stg_arr_av, get_max_dim(_rank));

	// Copy: staging array (stg_arr_av) -> target array (arr_av)
	array<_type, _rank> target_arr(src_stg_arr.get_extent(), arr_av, target_access_type);
	copy(src_stg_arr, target_arr);

	// Modify target array on arr_av
	ModifyOnAcceleratorView(arr_av, target_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array (arr_av) -> staging array (stg_arr_av)
	array<_type, _rank> dest_stg_arr(src_stg_arr.get_extent(), cpu_av, stg_arr_av);
	copy(target_arr, dest_stg_arr);

	return VerifyDataOnCpu(src_stg_arr, dest_stg_arr, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromStagingArrayToArrayView(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_v_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_stg_arr = CreateStagingArrayAndFillData<_type, _rank>(cpu_av, stg_arr_av, get_max_dim(_rank));

	// Copy: staging array (stg_arr_av) -> target array_view (arr_v_av)
	array<_type, _rank> data_arr(src_stg_arr.get_extent(), arr_v_av, target_access_type);
	array_view<_type, _rank> target_arr_v(data_arr);
	copy(src_stg_arr, target_arr_v);

	// Modify target array_view on arr_v_av
	ModifyOnAcceleratorView(arr_v_av, target_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: target array_view (arr_v_av) -> staging array (stg_arr_av)
	array<_type, _rank> dest_stg_arr(src_stg_arr.get_extent(), cpu_av, stg_arr_av);
	copy(target_arr_v, dest_stg_arr);

	return VerifyDataOnCpu(src_stg_arr, dest_stg_arr, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromStagingArrayToStagingArray(const concurrency::accelerator_view& src_cpu_av, const concurrency::accelerator_view& src_stg_arr_av, const concurrency::accelerator_view& target_cpu_av, const concurrency::accelerator_view& target_stg_arr_av)
{
	using namespace concurrency::Test;

	// Create source data
	array<_type, _rank> src_stg_arr = CreateStagingArrayAndFillData<_type, _rank>(src_cpu_av, src_stg_arr_av, get_max_dim(_rank));

	// Copy: staging array (src_stg_arr_av) -> target staging array (target_stg_arr_av)
	array<_type, _rank> target_stg_arr(src_stg_arr.get_extent(), target_cpu_av, target_stg_arr_av);
	copy(src_stg_arr, target_stg_arr);

	// Modify target stging array on cpu accelerator_view
	ModifyOnAcceleratorView(target_cpu_av, target_stg_arr, static_cast<_type>(MODIFY_VALUE));

	// Copy: target staging array (target_stg_arr_av) -> staging array (src_stg_arr_av)
	array<_type, _rank> dest_stg_arr(src_stg_arr.get_extent(), src_cpu_av, src_stg_arr_av);
	copy(target_stg_arr, dest_stg_arr);

	return VerifyDataOnCpu(src_stg_arr, dest_stg_arr, static_cast<_type>(MODIFY_VALUE));
}

template<typename _type, int _rank>
bool CopyAndVerifyFromStagingArrayToNonContigArrayView(
								const concurrency::accelerator_view& cpu_av,
								const concurrency::accelerator_view& arr_v_av,
								const concurrency::accelerator_view& stg_arr_av,
							  	concurrency::access_type target_access_type)
{
	using namespace concurrency::Test;

	array<_type, _rank> data_arr = CreateArrayAndFillData<_type, _rank>(arr_v_av, get_max_dim(_rank), target_access_type);
	array_view<_type, _rank> non_contig_arr_v = CreateNonContiguousArrayView<_type, _rank>(data_arr);

	// Create source data
	std::vector<_type> data(non_contig_arr_v.get_extent().size(), static_cast<_type>(INIT_VALUE));
	array<_type, _rank> src_stg_arr(non_contig_arr_v.get_extent(), data.begin(), cpu_av, stg_arr_av);

	// Copy: staging array (stg_arr_av) -> non-contiguously array_view (arr_v_av)
	copy(src_stg_arr, non_contig_arr_v);

	// Modify non-contiguous array_view on arr_v_av
	ModifyOnAcceleratorView(arr_v_av, non_contig_arr_v, static_cast<_type>(MODIFY_VALUE));

	// Copy: non-contiguously array_view (arr_v_av) -> staging array (stg_arr_av)
	array<_type, _rank> dest_stg_arr(src_stg_arr.get_extent(), cpu_av, stg_arr_av);
	copy(non_contig_arr_v, dest_stg_arr);

	return VerifyDataOnCpu(src_stg_arr, dest_stg_arr, static_cast<_type>(MODIFY_VALUE));
}

#pragma endregion
