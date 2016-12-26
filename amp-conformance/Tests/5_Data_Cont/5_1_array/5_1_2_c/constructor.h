/// This file contains template functions used by many array constructor tests

#include "./../dpc_array.h"

#define MIN_AVAILABLE               (0.80) // % of dedicated memory available - value should be between 0 - 1. Min of 80%
#define _KB_                        (1024)  // Used to convert KB to bytes
#define _AVAIALABLE_BYTES           (_KB_)  // Aliasing

// Validate array rank
template<typename _type, int _rank>
bool test_array_rank(int extval = _rank)
{
    int *data = new int[_rank];
    for (int i = 0; i < _rank; i++)
        data[i] = extval;

    extent<_rank> e(data);
    array<_type, _rank> a1(e);

    parallel_for_each(e, [&](index<_rank> idx) __GPU{
        a1[idx] = 1;
    });

    // is the rank correct
    if (a1.rank != _rank)
    {
        return false;
    }

    // verify data
    std::vector<_type> vdata = a1;
    for (unsigned int i = 0; i < e.size(); i++)
    {
        if (vdata[i] != 1)
            return false;
    }

    return true;
}

// testing supported types
template<typename _type, int _rank>
bool test_array_type_rank(int *data)
{
    extent<_rank> e(data);
    array<_type, _rank> a1(e);

    parallel_for_each(e, [&](index<_rank> idx) __GPU{
        a1[idx] = 1;
    });

    // verify data
    std::vector<_type> vdata = a1;
    for (unsigned int i = 0; i < e.size(); i++)
    {
        if (vdata[i] != 1)
            return false;
    }

    return true;
}

template<typename _type>
bool test_array_type()
{
    int data[5] = {1, 2, 3, 4, 5};

    return test_array_type_rank<_type, 1>(data) &&
        test_array_type_rank<_type, 2>(data) &&
        test_array_type_rank<_type, 5>(data);
}

template<typename _type, int _rank>
void kernel_userdefined(index<_rank>& idx, array<_type, _rank>& f) __GPU;

template<typename _type, int _rank>
bool verify_kernel_userdefined(array<_type, _rank>& arr);

template<typename _type, int _rank>
bool test_array_userdefined(int *data)
{
    extent<_rank> e(data);
    array<_type, _rank> a1(e);

    parallel_for_each(e, [&](index<_rank> idx) __GPU{
        // this is defined as part of test
        kernel_userdefined<_type, _rank>(idx, a1);
    });

    // this is defined as part of test
    return verify_kernel_userdefined<_type, _rank>(a1);
}

// Verify various ways of copy constructor usage
template<typename _type, int _rank>
bool test_array_copy_constructor()
{
	bool result = true;
    int *edata = new int[_rank];
    for (int i = 0; i < _rank; i++)
	{
        edata[i] = 3;
	}

    extent<_rank> e1(edata);
    std::vector<_type> data;

    for (unsigned int i = 0; i < e1.size(); i++)
	{
        data.push_back((_type)rand());
	}

    array<_type, _rank> src(e1, data.begin(), data.end());


    // Verify non-const creation
    {
        array<_type, _rank> dst (src);
        result &= VerifyDataOnCpu<_type, _rank>(src, dst);
    }

	delete[] edata;
    return result;
}

// Verify various ways of copy constructor usage
template<typename _type, int _rank >
bool test_array_copy_constructors_with_array_view()
{
	bool result = true;
    int *edata = new int[_rank];
    for (int i = 0; i < _rank; i++)
	{
        edata[i] = 3;
	}

    extent<_rank> e1(edata);
    std::vector<_type> data;

    for (unsigned int i = 0; i < e1.size(); i++)
	{
        data.push_back((_type)rand());
	}

    array_view<_type, _rank> src(e1, data);
    // Verify non-const creation
    {
        array<_type, _rank> dst(src);
        result &= VerifyDataOnCpu<_type, _rank>(src, dst);
    }

	delete[] edata;
    return result;
}

template<typename _type, int _rank , typename _accl>
bool test_array_copy_constructors_with_array_view(_accl device)
{
	bool result = true;
    int *edata = new int[_rank];
    for (int i = 0; i < _rank; i++)
	{
        edata[i] = 3;
	}

    extent<_rank> e1(edata);
    std::vector<_type> data;

    for (unsigned int i = 0; i < e1.size(); i++)
	{
        data.push_back((_type)rand());
	}

    array_view<_type, _rank> src(e1, data);
    // Verify non-const creation
    {
        array<_type, _rank> dst(src, device);
        result &= VerifyDataOnCpu<_type, _rank>(src, dst);
    }

	delete[] edata;
    return result;
}

template<typename _type, int _rank , typename _accl>
bool test_array_copy_constructors_with_array_view(_accl device1,_accl device2)
{
	bool result = true;
    int *edata = new int[_rank];
    for (int i = 0; i < _rank; i++)
	{
        edata[i] = 3;
	}

    extent<_rank> e1(edata);
    std::vector<_type> data;

    for (unsigned int i = 0; i < e1.size(); i++)
	{
        data.push_back((_type)rand());
	}

    array_view<_type, _rank> src(e1, data);
    // Verify non-const creation
    {
        array<_type, _rank> dst(src, device1, device2);
        result &= VerifyDataOnCpu<_type, _rank>(src, dst);
    }

	delete[] edata;
    return result;
}

// Validate size of 1D array
template<typename _type, int _rank, int _D0>
bool test_array_1d()
{
    {
        array<_type, _rank> src(_D0);

        if (src.get_extent().size() != _D0)
            return false;
    }

    return true;
}

// Validate size of 2D array
template<typename _type, int _rank, int _D0, int _D1>
bool test_array_2d()
{
    {
        array<_type, _rank> src(_D0, _D1);

        if (src.get_extent().size() != _D0 * _D1)
            return false;
    }

    return true;
}

// Validate size of 3D array
template<typename _type, int _rank, int _D0, int _D1, int _D2>
bool test_array_3d()
{
    {
        array<_type, _rank> src(_D0, _D1, _D2);

        if (src.get_extent().size() != _D0 * _D1 * _D2)
            return false;
    }

    return true;
}

// Testing extent and bounded iterator based constructor
template<typename _type, int _rank, typename _Iterator>
bool test_feature_itr(extent<_rank> _extent, _Iterator _first, _Iterator _last)
{
    // Init src1 array
    array<_type, _rank> src1(_extent, _first, _last);
    parallel_for_each(src1.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        src1(idx) = src1[idx]*2;
    });

    // compare with src1 and update src2 to be same
    array<_type, _rank> src2(_extent, _first, _last);
    parallel_for_each(src2.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != (src2(idx) * 2))
            src2(idx) = 0;
        else
            src2(idx) = src1(idx);
    });

    // compare with src1 and src2 - update result is dst
    array<_type, _rank> dst(_extent);
    parallel_for_each(dst.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != src2(idx))
        {
            dst(idx) = 1;
        }
        else
        {
            dst(idx) = 0;
        }
    });

    std::vector<_type> dst_data = dst;

    if (dst_data.size() != _extent.size())
    {
        return false;
    }

    for (size_t i = 0; i < dst_data.size(); i++)
    {
        if (dst_data[i] == 1)
        {
            return false;
        }
    }

    return true;
}

// Testing for 1D array and bounded iterator specialized constructor
template<typename _type, int _rank, int _D0, typename _Iterator>
bool test_feature_itr(_Iterator _first, _Iterator _last)
{
    // Init src1 array
    array<_type, _rank> src1(_D0, _first, _last);
    parallel_for_each(src1.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        src1(idx) = src1[idx]*2;
    });

    // compare with src1 and update src2 to be same
    array<_type, _rank> src2(_D0, _first);
    parallel_for_each(src2.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != (src2(idx) * 2))
            src2(idx) = 0;
        else
            src2(idx) = src1(idx);
    });

    // compare with src1 and src2 - update result is dst
    array<_type, _rank> dst(_D0);
    parallel_for_each(dst.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != src2(idx))
        {
            dst(idx) = 1;
        }
        else
        {
            dst(idx) = 0;
        }

    });

    std::vector<_type> dst_data = dst;
    if (dst_data.size() != _D0)
    {
        return false;
    }

    for (size_t i = 0; i < dst_data.size(); i++)
    {
        if (dst_data[i] == 1)
        {
            return false;
        }
    }

    return true;
}

// Testing for 2D array and bounded iterator specialized constructor
template<typename _type, int _rank, int _D0, int _D1, typename _Iterator>
bool test_feature_itr(_Iterator _first, _Iterator _last)
{
    // Init src1 array
    array<_type, _rank> src1(_D0, _D1, _first, _last);
    parallel_for_each(src1.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        src1(idx) = src1[idx]*2;
    });

    // compare with src1 and update src2 to be same
    array<_type, _rank> src2(_D0, _D1, _first);
    parallel_for_each(src2.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != (src2(idx) * 2))
            src2(idx) = 0;
        else
            src2(idx) = src1(idx);
    });

    // compare with src1 and src2 - update result is dst
    array<_type, _rank> dst(_D0, _D1);
    parallel_for_each(dst.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != src2(idx))
        {
            dst(idx) = 1;
        }
        else
        {
            dst(idx) = 0;
        }
    });

    std::vector<_type> dst_data = dst;
    if (dst_data.size() != (_D0*_D1))
        return false;
    for (size_t i = 0; i < dst_data.size(); i++)
        if (dst_data[i] == 1)
            return false;

    return true;
}

// Testing for 3D array and bounded iterator specialized constructor
template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _Iterator>
bool test_feature_itr(_Iterator _first, _Iterator _last)
{
    // Init src1 array
    array<_type, _rank> src1(_D0, _D1, _D2, _first, _last);
    parallel_for_each(src1.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        src1(idx) = src1[idx]*2;
    });

    // compare with src1 and update src2 to be same
    array<_type, _rank> src2(_D0, _D1, _D2, _first);
    parallel_for_each(src2.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != (src2(idx) * 2))
            src2(idx) = 0;
        else
            src2(idx) = src1(idx);
    });

    // compare with src1 and src2 - update result is dst
    array<_type, _rank> dst(_D0, _D1, _D2);
    parallel_for_each(dst.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        if (src1(idx) != src2(idx))
        {
            dst(idx) = 1;
        }
        else
        {
            dst(idx) = 0;
        }
    });

    std::vector<_type> dst_data = dst;
    if (dst_data.size() != _D0*_D1*_D2)
        return false;

    for (size_t i = 0; i < dst_data.size(); i++)
        if (dst_data[i] == 1)
            return false;

    return true;
}

// Testing extent and unbounded iterator based constructor
template<typename _type, int _rank, typename _BeginIterator>
bool test_feature_itr(extent<_rank> _e, _BeginIterator _first)
{
    array<_type, _rank> src(_e, _first);
    array<_type, _rank> dst(_e);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _e.size()) || (dst_data.size() != _e.size()))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 1D and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, typename _BeginIterator>
bool test_feature_itr(_BeginIterator _first)
{
    array<_type, _rank> src(_D0, _first);
    array<_type, _rank> dst(_D0);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0) || (dst_data.size() != _D0))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 2D and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, typename _BeginIterator>
bool test_feature_itr(_BeginIterator _first)
{
    array<_type, _rank> src(_D0, _D1, _first);
    array<_type, _rank> dst(_D0, _D1);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0*_D1) || (dst_data.size() != _D0*_D1))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 3D and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _BeginIterator>
bool test_feature_itr(_BeginIterator _first)
{
    array<_type, _rank> src(_D0, _D1, _D2, _first);
    array<_type, _rank> dst(_D0, _D1, _D2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0*_D1*_D2) || (dst_data.size() != _D0*_D1*_D2))
    {
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing accelerator/accelerator_view based array constructor
template<typename _type, int _rank, typename _accl>
bool test_accl_constructor(_accl device)
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;

    {
        extent<_rank> e1(edata);
        array<_type, _rank> src(e1, device);

        // let the kernel initialize data;
        parallel_for_each(e1, [&](index<_rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if (opt[i] != _rank)
                return false;
        }
    }

    if (_rank > 0) // for rank 1
    {
        const int rank = 1;
        array<_type, rank> src(edata[0], device);

        // let the kernel initialize data;
        extent<1> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if (opt[i] != _rank)
                return false;
        }
    }

    if (_rank > 1) // for rank 2
    {
        const int rank = 2;
        array<_type, rank> src(edata[0], edata[1], device);

        // let the kernel initialize data;
        extent<rank> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if (opt[i] != _rank)
                return false;
        }
    }

    if (_rank > 2) // for rank 3
    {
        const int rank = 3;
        array<_type, rank> src(edata[0], edata[1], edata[2], device);

        // let the kernel initialize data;
        extent<rank> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if (opt[i] != _rank)
                return false;
        }
    }

    return true;
}

template<typename _type, int _rank, typename _accl>
bool test_accl_constructor_diff_array_same_kernel(_accl device1, _accl device2)
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;

    {
        extent<_rank> e1(edata);
        array<_type, _rank> src1(e1, device1);
        array<_type, _rank> src2(e1, device2);

        // let the kernel initialize data;
        parallel_for_each(e1, [&](index<_rank> idx) __GPU_ONLY
        {
            src1[idx] = _rank;
            src2[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt1(e1.size());
        opt1 = src1;
        vector<_type> opt2(e1.size());
        opt2 = src2;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if ((opt1[i] != _rank) || (opt2[i] != _rank))
                return false;
        }
    }

    if (_rank > 0)
    {
        const int rank = 1;
        array<_type, rank> src1(edata[0], device1);
        array<_type, rank> src2(edata[0], device2);

        // let the kernel initialize data;
        extent<1> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src1[idx] = _rank;
            src2[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt1(e1.size());
        opt1 = src1;
        vector<_type> opt2(e1.size());
        opt2 = src2;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if ((opt1[i] != _rank) || (opt2[i] != _rank))
                return false;
        }
    }
    if (_rank > 1)
    {
        const int rank = 2;
        array<_type, rank> src1(edata[0], edata[1], device1);
        array<_type, rank> src2(edata[0], edata[1], device2);

        // let the kernel initialize data;
        extent<rank> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src1[idx] = _rank;
            src2[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt1(e1.size());
        opt1 = src1;
        vector<_type> opt2(e1.size());
        opt2 = src2;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if ((opt1[i] != _rank) || (opt2[i] != _rank))
                return false;
        }
    }
    if (_rank > 2)
    {
        const int rank = 3;
        array<_type, rank> src1(edata[0], edata[1], edata[2], device1);
        array<_type, rank> src2(edata[0], edata[1], edata[2], device2);

        // let the kernel initialize data;
        extent<rank> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src1[idx] = _rank;
            src2[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt1(e1.size());
        opt1 = src1;
        vector<_type> opt2(e1.size());
        opt2 = src2;

        for (unsigned int i = 0; i < e1.size(); i++)
        {
            if ((opt1[i] != _rank) || (opt2[i] != _rank))
                return false;
        }
    }

    return true;
}

template<typename _type, int _rank, typename _staging>
bool test_accl_staging_buffer_constructor(_staging device1, _staging device2)
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;

	bool passed = true;

    { // Extent based staging constructor
		Log(LogType::Info, true) << "  Verifying array<" << get_type_name<_type>() << ", " << _rank << ">..." << std::endl;
        extent<_rank> e1(edata);
        array<_type, _rank> src(e1, device1, device2);

        // let the kernel initialize data;
        parallel_for_each(e1, [&](index<_rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

		passed &= REPORT_RESULT(VerifyAllSameValue(opt, static_cast<_type>(_rank)) == -1);
    }

    if (_rank > 0) // for rank 1
    {
        const int rank = 1;
		Log(LogType::Info, true) << "  Verifying array<" << get_type_name<_type>() << ", " << rank << ">..." << std::endl;
        array<_type, rank> src(edata[0], device1, device2);

        // let the kernel initialize data;
        extent<1> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

		passed &= REPORT_RESULT(VerifyAllSameValue(opt, static_cast<_type>(_rank)) == -1);
    }

    if (_rank > 1) // for rank 2
    {
        const int rank = 2;
		Log(LogType::Info, true) << "  Verifying array<" << get_type_name<_type>() << ", " << rank << ">..." << std::endl;
        array<_type, rank> src(edata[0], edata[1], device1, device2);

        // let the kernel initialize data;
        extent<rank> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

		passed &= REPORT_RESULT(VerifyAllSameValue(opt, static_cast<_type>(_rank)) == -1);
    }

    if (_rank > 2)
    {
        const int rank = 3;
		Log(LogType::Info, true) << "  Verifying array<" << get_type_name<_type>() << ", " << rank << ">..." << std::endl;
        array<_type, rank> src(edata[0], edata[1], edata[2], device1, device2);

        // let the kernel initialize data;
        extent<rank> e1(edata);
        parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
        {
            src[idx] = _rank;
        });

        // Copy data to CPU
        vector<_type> opt(e1.size());
        opt = src;

		passed &= REPORT_RESULT(VerifyAllSameValue(opt, static_cast<_type>(_rank)) == -1);
    }

    return passed;
}

// Testing extent, device/view and bounded iterator based constructor
template<typename _type, int _rank, typename _accl, typename _Iterator>
bool test_feature_itr(extent<_rank> _extent, _Iterator _first, _Iterator _last, _accl device)
{
    array<_type, _rank> src(_extent, _first, _last, device);
    array<_type, _rank> dst(_extent, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing 1D, device/view and bounded iterator based constructor
template<typename _type, int _rank, int _D0, typename _accl, typename _Iterator>
bool test_feature_itr(_Iterator _first, _Iterator _last, _accl device)
{
    array<_type, _rank> src(_D0, _first, _last, device);
    array<_type, _rank> dst(_D0, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing 2D, device/view and bounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, typename _accl, typename _Iterator>
bool test_feature_itr(_Iterator _first, _Iterator _last, _accl device)
{
    array<_type, _rank> src(_D0, _D1, _first, _last, device);
    array<_type, _rank> dst(_D0, _D1, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing 3D, device/view and bounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _accl, typename _Iterator>
bool test_feature_itr(_Iterator _first, _Iterator _last, _accl device)
{
    array<_type, _rank> src(_D0, _D1, _D2, _first, _last, device);
    array<_type, _rank> dst(_D0, _D1, _D2, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing extent, device/view and unbounded iterator based constructor
template<typename _type, int _rank, typename _accl, typename _BeginIterator>
bool test_feature_itr(extent<_rank> _e, _BeginIterator _first, _accl device)
{
    array<_type, _rank> src(_e, _first, device);
    array<_type, _rank> dst(_e, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _e.size()) || (dst_data.size() != _e.size()))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 1D, device/view and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, typename _accl, typename _BeginIterator>
bool test_feature_itr(_BeginIterator _first, _accl device)
{
    array<_type, _rank> src(_D0, _first, device);
    array<_type, _rank> dst(_D0, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0) || (dst_data.size() != _D0))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 2D, device/view and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, typename _accl, typename _BeginIterator>
bool test_feature_itr(_BeginIterator _first, _accl device)
{
    array<_type, _rank> src(_D0, _D1, _first, device);
    array<_type, _rank> dst(_D0, _D1, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0*_D1) || (dst_data.size() != _D0*_D1))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 3D, device/view and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _accl, typename _BeginIterator>
bool test_feature_itr(_BeginIterator _first, _accl device)
{
    array<_type, _rank> src(_D0, _D1, _D2, _first, device);
    array<_type, _rank> dst(_D0, _D1, _D2, device);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0*_D1*_D2) || (dst_data.size() != _D0*_D1*_D2))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing extent, device/view, staging and bounded iterator based constructor
template<typename _type, int _rank, typename _accl, typename _Iterator>
bool test_feature_staging_itr(extent<_rank> _extent, _Iterator _first, _Iterator _last, _accl device1, _accl device2)
{
    array<_type, _rank> src(_extent, _first, _last, device1, device2);
    array<_type, _rank> dst(_extent, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing 1D, device/view, staging and bounded iterator based constructor
template<typename _type, int _rank, int _D0, typename _accl, typename _Iterator>
bool test_feature_staging_itr(_Iterator _first, _Iterator _last, _accl device1, _accl device2)
{
    array<_type, _rank> src(_D0, _first, _last, device1, device2);
    array<_type, _rank> dst(_D0, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing 2D, device/view, staging and bounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, typename _accl, typename _Iterator>
bool test_feature_staging_itr(_Iterator _first, _Iterator _last, _accl device1, _accl device2)
{
    array<_type, _rank> src(_D0, _D1, _first, _last, device1, device2);
    array<_type, _rank> dst(_D0, _D1, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing 3D, device/view, staging and bounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _accl, typename _Iterator>
bool test_feature_staging_itr(_Iterator _first, _Iterator _last, _accl device1, _accl device2)
{
    array<_type, _rank> src(_D0, _D1, _D2, _first, _last, device1, device2);
    array<_type, _rank> dst(_D0, _D1, _D2, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst(idx) = src[idx];
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    return Verify(dst_data, src_data);
}

// Testing extent, device/view, staging and unbounded iterator based constructor
template<typename _type, int _rank, typename _accl, typename _BeginIterator>
bool test_feature_staging_itr(extent<_rank> _e, _BeginIterator _first, _accl device1, _accl device2)
{
    array<_type, _rank> src(_e, _first, device1, device2);
    array<_type, _rank> dst(_e, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _e.size()) || (dst_data.size() != _e.size()))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 1D, device/view, staging and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, typename _accl, typename _BeginIterator>
bool test_feature_staging_itr(_BeginIterator _first, _accl device1, _accl device2)
{
    array<_type, _rank> src(_D0, _first, device1, device2);
    array<_type, _rank> dst(_D0, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0) || (dst_data.size() != _D0))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 2D, device/view, staging and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, typename _accl, typename _BeginIterator>
bool test_feature_staging_itr(_BeginIterator _first, _accl device1, _accl device2)
{
    array<_type, _rank> src(_D0, _D1, _first, device1, device2);
    array<_type, _rank> dst(_D0, _D1, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0*_D1) || (dst_data.size() != _D0*_D1))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

// Testing 3D, device/view, staging and unbounded iterator based constructor
template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _accl, typename _BeginIterator>
bool test_feature_staging_itr(_BeginIterator _first, _accl device1, _accl device2)
{
    array<_type, _rank> src(_D0, _D1, _D2, _first, device1, device2);
    array<_type, _rank> dst(_D0, _D1, _D2, device1, device2);
    parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
    {
        dst[idx] = src(idx);
    });

    std::vector<_type> src_data = src;
    std::vector<_type> dst_data = dst;

    if ((src_data.size() != _D0*_D1*_D2) || (dst_data.size() != _D0*_D1*_D2))
    {
        std::cout << "Invalid size : src - " << src_data.size() << " dst - " << dst_data.size() << std::endl;
        return false;
    }

    return Verify(dst_data, src_data);
}

