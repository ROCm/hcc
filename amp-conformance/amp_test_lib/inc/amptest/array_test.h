
#pragma once

#include <algorithm>
#include <amp.h>
#include <assert.h>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <amptest/coordinates.h>
#include <amptest/array_view_test.h>
#include <amptest.h>

namespace Concurrency
{
namespace Test
{
    namespace details
    {
	    // ARRAY OVERLOADED
        template<typename value_type, int rank>
        value_type gpu_read(array<value_type, rank> &src, index<rank> idx)
        {
            value_type result_buf[1];
            array_view<value_type, 1> result(extent<1>(1), result_buf);
            parallel_for_each(result.get_extent(), [&src,result,idx](index<1>) restrict(amp) {
                result[0] = src[idx];
            });

            return result[0];
        }

		// ARRAY OVERLOADED
        template<typename value_type, int rank>
        void gpu_write(array<value_type, rank> &dest, index<rank> idx, value_type value)
        {
            parallel_for_each(extent<1>(1), [&dest,idx,value](index<1>) restrict(amp) {
                dest[idx] = value;
            });
        }
    }

    // forward declaration
    template<typename _value_type, int _rank, int _original_rank = _rank + 1>
    class ProjectedArrayTest;

    // forward declaration
    template<typename _value_type, int _rank>
    class ViewAsArrayTest;

    ///<summary>A common test class for positive tests that exercise indexing and other array operations</summary>
    ///<remarks>
    /// This class holds an array, as well as a map of known-values. A test should use this
    /// class to create an array, and then use the reference returned by arr() to get/set
    /// values.
    ///
    /// Adding data via set_known_value() opts in to validation and logging via the pass() and fail()
    /// methods.
    ///
    ///</remarks>
    template<typename _value_type, int _rank = 1, int _data_rank = _rank>
    class ArrayTest
    {
    public:

        static const int rank = _rank;

        ///<summary> the type of the index for this array</summary>
        typedef index<_rank> index_type;

        ///<summary>The type of value_type </summary>
        typedef _value_type value_type;

        ///<summary>The type of value_type with const removed</summary>
        // typedef typename std::remove_const<_value_type>::type value_type;

        ///<summary>The structure used to hold known-values</summary>
        typedef typename std::unordered_map<index<_data_rank>, value_type, details::IndexHash<_data_rank>> known_values_store;

        ///<summary>Creates a new ArrayTest -- creating a vector of the given size, and an array from it</summary>
        ArrayTest(extent<rank> extent) :
        _coordinates(new extent_coordinate_nest<rank>(extent)),
        _data(new std::vector<value_type>(extent.size())),
        _known_values(new known_values_store()),
        _arr(extent, (*_data.get()).begin())
        {
            Log(LogType::Info, true) << "Created Array of: " << extent << std::endl;
        }

        ///<summary>Creates a new ArrayTest -- with initial data, and an array from it</summary>
        ArrayTest(extent<rank> extent, std::vector<value_type> &data) :
        _coordinates(new extent_coordinate_nest<rank>(extent)),
        _data(new std::vector<value_type>(data)),
        _known_values(new known_values_store()),
        _arr(extent, (*_data.get()).begin())
        {
            assert(_data.get()->size() == extent.size());

            Log(LogType::Info, true) << "Created Array of: " << extent << std::endl;
            Log(LogType::Info, true) << "Initial data: ";
            std::ostream_iterator<value_type> os_iter(LogStream(), ", ");
            std::copy(_data.get()->begin(), _data.get()->end(), os_iter);
            LogStream() << std::endl;
        }

        ///<summary>Creates a new ArrayTest</summary>
        ArrayTest(
            std::shared_ptr<coordinate_nest<rank, _data_rank>> coordinates,
            std::shared_ptr<std::vector<value_type>> data,
            std::shared_ptr<known_values_store> known_values,
            array_view<value_type, rank> view) :
            _coordinates(coordinates),
            _data(data),
            _known_values(known_values),
            _arr(view)
        {
        };

		///<summary>Creates a new ArrayTest</summary>
        ArrayTest(
            std::shared_ptr<coordinate_nest<rank, _data_rank>> coordinates,
            std::shared_ptr<std::vector<value_type>> data,
            std::shared_ptr<known_values_store> known_values,
            array<value_type, rank>& arr) :
            _coordinates(coordinates),
            _data(data),
            _known_values(known_values),
            _arr(arr)
        {
        };

        ///<summary>Creates a new ArrayTest -- with sequential data</summary>
        template<int initial_value>
        static ArrayTest sequential(extent<rank> extent)
        {
            std::vector<value_type> data(extent.size());

            value_type n = initial_value;
            std::generate(data.begin(), data.end(), [&n]() mutable { return n++; });
            return ArrayTest(extent, data);
        }

        ///<summary>Registers a section of original array</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank - 1, _data_rank> projection(int i)
        {
            return projection(_arr[i], i);
        }

        ///<summary>Registers a section of original array</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank - 1, _data_rank> projection(array_view<value_type, rank - 1> other, int i)
        {
            Log(LogType::Info, true) << "Creating projection on: " << i << std::endl;

            std::shared_ptr<coordinate_nest<rank - 1, _data_rank>> p(new projected_coordinate_nest<rank - 1, rank, _data_rank>(_coordinates, index<1>(i)));
            return ArrayViewTest<value_type, rank - 1, _data_rank>(
                p,
                _data,
                _known_values,
                other);
        }

        ///<summary>Creates a section of original array</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(index<rank> offset)
        {
            return section(this->arr().section(offset), offset);
        };

        ///<summary>Creates a section of original array</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(extent<rank> ex)
        {
            return section(this->arr().section(ex), index<rank>());
        };

        ///<summary>Creates a section of original array</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(index<rank> origin, extent<rank> ex)
        {
            return section(this->arr().section(origin, ex), origin);
        };

        ///<summary>Registers a section of original array</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(array_view<value_type, rank> other, index<rank> origin)
        {
            Log(LogType::Info, true) << "Creating section: (origin: " << origin << " extent: "
                << other.get_extent() << ")" << std::endl;

			// make a copy
			std::shared_ptr<coordinate_nest<rank, _data_rank>> p(new offset_coordinate_nest<rank, _data_rank>(_coordinates, origin));
            ArrayViewTest<value_type, rank, _data_rank> otherTest(
                p,
				_data,
                _known_values,
                other);
            return otherTest;
        };

        template<int new_rank>
        ArrayViewTest<_value_type, new_rank, _data_rank> view_as(extent<new_rank> ex)
        {
            Log(LogType::Info, true) << "Creating reshaped view: (extent: " << ex << ")" << std::endl;
            std::shared_ptr<coordinate_nest<new_rank, _data_rank>> p(new reshaped_coordinate_nest<new_rank, _data_rank>(_coordinates, ex));
            return ArrayViewTest<_value_type, new_rank, _data_rank>(
                p,
                _data,
                _known_values,
                _arr.view_as(ex));
        }

        ///<summary>sets the given value in the array and known-values store</summary>
        void set_value(index<rank> i, value_type value)
        {
            set_known_value(i, value);

            details::gpu_write(_arr,i,value);
        }

        ///<summary>sets the given value in the known-values store</summary>
        void set_known_value(index<rank> i, value_type value)
        {
            Log(LogType::Info, true) << "Added known value at: " << i << " (this view) or " << _coordinates.get()->get_absolute(i) <<
            " (original array view) of: " << value << std::endl;

            (*_known_values)[_coordinates.get()->get_absolute(i)] = value;
        };

        typename std::vector<value_type>::iterator begin()
        {
            return _data.get()->begin();
        }

        typename std::vector<value_type>::iterator end()
        {
            return _data.get()->end();
        }

        coordinate_nest<rank, _data_rank>& coordinates() const
        {
            return *_coordinates.get();
        }

        ///<summary>returns a reference to the underlying data</summary>
        value_type* data()
        {
            return _data.get()->data();
        }

        ///<summary>returns a reference to the array</summary>
        array<value_type, rank>& arr()
        {
            return _arr;
        }

        const known_values_store& known_values()
        {
            return *_known_values.get();
        }

        ///<summary>uses the known-values to verify the underlying data, then returns runall_pass</summary>
        int pass()
        {
			copy(_arr,(*_data.get()).begin()); // Copying out the data

            for(auto iter = _known_values.get()->begin(); iter != _known_values.get()->end(); iter++)
            {
				// Only cause logging to occur if there are failures.
                if(this->data()[_coordinates.get()->get_linear(iter->first)] != iter->second)
                {
                    return fail();
                }
            }
            Log(LogType::Info, true) << "Pass" << std::endl;
            return runall_pass;
        };

        ///<summary>logs information, then returns runall_fail</summary>
        int fail()
        {
		    copy(_arr,(*_data.get()).begin()); // Copying out the data
            for(auto iter = _known_values.get()->begin(); iter != _known_values.get()->end(); iter++)
            {
				// Only log the failing elements
				if(this->data()[_coordinates.get()->get_linear(iter->first)] != iter->second)
				{
					Log(LogType::Error, true) << "Known value at: " << iter->first << " should be: " << iter->second << " was: "
						<< this->data()[_coordinates.get()->get_linear(iter->first)] << std::endl;
				}
            }

            Log(LogType::Info, true) << "Raw data: ";
            std::ostream_iterator<value_type> os_iter(LogStream(), ", ");
            std::copy(_data.get()->begin(), _data.get()->end(), os_iter);
            LogStream() << std::endl;
            Log(LogType::Error, true) << "Fail" << std::endl;
            return runall_fail;
        };

    private:
        std::shared_ptr<coordinate_nest<rank, _data_rank>> _coordinates;
        std::shared_ptr<std::vector<value_type>> _data;
        std::shared_ptr<known_values_store> _known_values;
        array<value_type, rank> _arr;
    };

	 // ARRAY OVERLOADED
    template<typename value_type, int rank>
    value_type gpu_read(array<value_type, rank> &src, index<rank> idx)
	{
		return details::gpu_read(src,idx);
	}

	template<typename value_type, int rank>
    void gpu_write(array<value_type, rank> &dest, index<rank> idx, value_type value)
	{
		details::gpu_write(dest,idx,value);
	}

    template<typename value_type, int rank>
    bool TestSection(ArrayTest<value_type, rank> &original, index<rank> origin)
    {
        ArrayViewTest<value_type, rank> section = original.section(origin);
        return TestSection(original, section, origin);

    }

    template<typename value_type, int rank>
    bool TestSection(ArrayTest<value_type, rank> &original, index<rank> origin, extent<rank> ex)
    {
        ArrayViewTest<value_type, rank> section = original.section(origin, ex);
        return TestSection(original, section, origin);
    }

    template<typename value_type, int rank>
    bool TestSection(ArrayTest<value_type, rank> &original, array_view<value_type, rank> &section_av, index<rank> origin)
    {
        ArrayViewTest<value_type, rank> section = original.section(section_av, origin);
        return TestSection(original, section, origin);
    }

    template<typename value_type, int rank>
    bool TestSection(ArrayTest<value_type, rank> &original, ArrayViewTest<value_type, rank> &section, index<rank> origin)
    {
		type_comparer<value_type> comparer;

        // now choose random points in the section
        // relative to the original
        index<rank> set_original_on_gpu = details::random_index(origin, section.view().get_extent());

        // relative to the section
        index<rank> set_section_on_gpu = details::random_index(origin, section.view().get_extent()) - origin;
        index<rank> set_section_on_cpu = details::random_index(origin, section.view().get_extent()) - origin;

        // set a value in the section on the GPU
        value_type expected_value = static_cast<value_type>(rand());
        Log(LogType::Info, true) << "Setting a value in the section AV on the GPU" << std::endl;
        details::gpu_write(section.view(), set_section_on_gpu, expected_value);
        section.set_known_value(set_section_on_gpu, expected_value);
        value_type actual_value = details::gpu_read(original.arr(),set_section_on_gpu + origin);
        if (!comparer.are_equal(actual_value, expected_value))
        {
            Log(LogType::Error, true) << "Reading original (CPU) expected: " << expected_value << " actual: " << actual_value << std::endl;
            return false;
        }

        // set a value in the original on the GPU
        expected_value = static_cast<value_type>(rand());
        Log(LogType::Info, true) << "Setting a value in the original AV on the GPU" << std::endl;
        details::gpu_write(original.arr(), set_original_on_gpu, expected_value);
        original.set_known_value(set_original_on_gpu, expected_value);
        actual_value = details::gpu_read(section.view(), set_original_on_gpu - origin);
        if (!comparer.are_equal(actual_value, expected_value))
        {
            Log(LogType::Error, true) << "Reading section (GPU) expected: " << expected_value << " actual: " << actual_value << std::endl;
            return false;
        }

        // set a value in the section on the CPU
        expected_value = static_cast<value_type>(rand());
        Log(LogType::Info, true) << "Setting a value in the section AV on the CPU" << std::endl;
        section.view()[set_section_on_gpu] = expected_value;
        section.set_known_value(set_section_on_gpu, expected_value);
        actual_value = details::gpu_read(original.arr(), set_section_on_gpu + origin);
        if (!comparer.are_equal(actual_value, expected_value))
        {
            Log(LogType::Error, true) << "Reading original (GPU) expected: " << expected_value << " actual: " << actual_value << std::endl;
            return false;
        }

        return true;
    }
}
}
