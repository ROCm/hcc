//--------------------------------------------------------------------------------------
// File: array_view_test.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//

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
#include <amptest.h>

namespace Concurrency
{
namespace Test
{
    namespace details
    {
        ///<summary>A function-object for hashing an index<rank></summary>
        template<int rank>
        struct IndexHash : public std::unary_function<index<rank>, std::size_t>
        {
            size_t operator()(index<rank> index) const
            {
                size_t value = 0;
                for (int i = 0; i < rank; i++)
                {
                    value ^= index[i];
                }
                return value;
            }
        };

        ///<summary>Generates a random index within the given bounds</summary>
        template<int rank>
        index<rank> random_index(index<rank> origin, extent<rank> ex)
        {
            int subscripts[rank];
            for (int i = 0; i < rank; i++)
            {
                subscripts[i] = origin[i] + (rand() % ex[i]);
            }
            return index<rank>(subscripts);
        }

        template<typename value_type, int rank>
        value_type gpu_read(array_view<value_type, rank> &src, index<rank> idx)
        {
            value_type result_buf[1];
            array_view<value_type, 1> result(extent<1>(1), result_buf);
            parallel_for_each(result.get_extent(), [=](index<1>) restrict(amp) {
                result[0] = src[idx];
            });

            return result[0];
        }

        template<typename value_type, int rank>
        void gpu_write(array_view<value_type, rank> &dest, index<rank> idx, value_type value)
        {
            parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp) {
                dest[idx] = value;
            });
        }
    }

    // forward declaration
    template<typename _value_type, int _rank, int _original_rank = _rank + 1>
    class ProjectedArrayViewTest;

    // forward declaration
    template<typename _value_type, int _rank>
    class ViewAsArrayViewTest;

    ///<summary>A common test class for positive tests that exercise indexing and other AV operations</summary>
    ///<remarks>
    /// This class holds an array_view, as well as a map of known-values. A test should use this
    /// class to create an array_view, and then use the reference returned by view() to get/set
    /// values.
    ///
    /// Adding data via set_known_value() opts in to validation and logging via the pass() and fail()
    /// methods.
    ///
    /// For testing an array_view<const T>, the set_value() and data() members allow operations
    /// on the non-const backing memory.
    ///</remarks>
    template<typename _value_type, int _rank = 1, int _data_rank = _rank>
    class ArrayViewTest
    {
    public:

        static const int rank = _rank;

        ///<summary> the type of the index for this view</summary>
        typedef index<_rank> index_type;

        ///<summary>The type of value_type </summary>
        typedef _value_type value_type;

        ///<summary>The type of value_type with const removed</summary>
        typedef typename std::remove_const<_value_type>::type non_const;

        ///<summary>The structure used to hold known-values</summary>
        typedef typename std::unordered_map<index<_data_rank>, non_const, details::IndexHash<_data_rank>> known_values_store;

        ///<summary>Creates a new ArrayViewTest -- creating a vector of the given size, and an array_view around it</summary>
        ArrayViewTest(extent<rank> extent) :
        _coordinates(new extent_coordinate_nest<rank>(extent)),
        _data(new std::vector<non_const>(extent.size())),
        _known_values(new known_values_store()),
        _view(extent, *_data.get())
        {
            Log(LogType::Info, true) << "Created Array View of: " << extent << std::endl;
        }

        ///<summary>Creates a new ArrayViewTest -- with initial data, and an array_view around it</summary>
        ArrayViewTest(extent<rank> extent, std::vector<non_const> &data) :
        _coordinates(new extent_coordinate_nest<rank>(extent)),
        _data(new std::vector<non_const>(data)),
        _known_values(new known_values_store()),
        _view(extent, *_data.get())
        {
            assert(_data.get()->size() == extent.size());

            Log(LogType::Info, true) << "Created Array View of: " << extent << std::endl;
            Log(LogType::Info, true) << "Initial data: ";
            std::ostream_iterator<value_type> os_iter(LogStream(), ", ");
            std::copy(_data.get()->begin(), _data.get()->end(), os_iter);
            LogStream() << std::endl;
        }

        ///<summary>Creates a new ArrayViewTest</summary>
        ArrayViewTest(
            std::shared_ptr<coordinate_nest<rank, _data_rank>> coordinates,
            std::shared_ptr<std::vector<non_const>> data,
            std::shared_ptr<known_values_store> known_values,
            array_view<value_type, rank> view) :
            _coordinates(coordinates),
            _data(data),
            _known_values(known_values),
            _view(view)
        {
        };

        ///<summary>Creates a new ArrayViewTest -- with sequential data, and an array_view around it</summary>
        template<int initial_value>
        static ArrayViewTest sequential(extent<rank> extent)
        {
            std::vector<non_const> data(extent.size());

            non_const n = initial_value;
            std::generate(data.begin(), data.end(), [&n]() mutable { return n++; });
            return ArrayViewTest(extent, data);
        }

        ///<summary>Registers a section of original array_view</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank - 1, _data_rank> projection(int i)
        {
            return projection(_view[i], i);
        }

        ///<summary>Registers a section of original array_view</summary>
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

        ///<summary>Creates a section of original array_view</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(index<rank> offset)
        {
            return section(this->view().section(offset), offset);
        };

        ///<summary>Creates a section of original array_view</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(extent<rank> ex)
        {
            return section(this->view().section(ex), index<rank>());
        };

        ///<summary>Creates a section of original array_view</summary>
        ///<remarks>
        ///The nested section shares the underlying data and known-values store with the original.
        ///All operations on the nested section are translated to an absolute index into the original
        ///data set.
        ///</remarks>
        ArrayViewTest<value_type, rank, _data_rank> section(index<rank> origin, extent<rank> ex)
        {
            return section(this->view().section(origin, ex), origin);
        };

        ///<summary>Registers a section of original array_view</summary>
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
            ArrayViewTest<value_type, rank, _data_rank> otherTest = *this;
            otherTest._view = other;
            otherTest._coordinates.reset(new offset_coordinate_nest<rank, _data_rank>(_coordinates, origin));
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
                _view.view_as(ex));
        }

        ///<summary>sets the given value in the underlying data (backing-store vector) and known-values store</summary>
        void set_value(index<rank> i, value_type value)
        {
            set_known_value(i, value);

            unsigned int linear_index = _coordinates.get()->get_linear(i);
            this->data()[linear_index] = value;
        }

        ///<summary>sets the given value in the known-values store</summary>
        void set_known_value(index<rank> i, value_type value)
        {
            Log(LogType::Info, true) << "Added known value at: " << i << " (this view) or " << _coordinates.get()->get_absolute(i) <<
            " (original array view) of: " << value << std::endl;

            (*_known_values)[_coordinates.get()->get_absolute(i)] = value;
        };

        typename std::vector<non_const>::iterator begin()
        {
            return _data.get()->begin();
        }

        typename std::vector<non_const>::iterator end()
        {
            return _data.get()->end();
        }

        coordinate_nest<rank, _data_rank>& coordinates() const
        {
            return *_coordinates.get();
        }

        ///<summary>returns a reference to the underlying data</summary>
        non_const* data()
        {
            return _data.get()->data();
        }

        ///<summary>returns a reference to the array_view</summary>
        array_view<value_type, rank>& view()
        {
            return _view;
        }

        const known_values_store& known_values()
        {
            return *_known_values.get();
        }

        ///<summary>uses the known-values to verify the underlying data, then returns runall_pass</summary>
        int pass()
        {
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
        std::shared_ptr<std::vector<non_const>> _data;
        std::shared_ptr<known_values_store> _known_values;
        array_view<value_type, rank> _view;
    };

    template<typename value_type, int rank>
    bool TestSection(ArrayViewTest<value_type, rank> &original, index<rank> origin)
    {
        ArrayViewTest<value_type, rank> section = original.section(origin);
        return TestSection(original, section, origin);

    }

    template<typename value_type, int rank>
    bool TestSection(ArrayViewTest<value_type, rank> &original, index<rank> origin, extent<rank> ex)
    {
        ArrayViewTest<value_type, rank> section = original.section(origin, ex);
        return TestSection(original, section, origin);
    }

    template<typename value_type, int rank>
    bool TestSection(ArrayViewTest<value_type, rank> &original, array_view<value_type, rank> &section_av, index<rank> origin)
    {
        ArrayViewTest<value_type, rank> section = original.section(section_av, origin);
        return TestSection(original, section, origin);
    }

    template<typename value_type, int rank>
    bool TestSection(ArrayViewTest<value_type, rank> &original, ArrayViewTest<value_type, rank> &section, index<rank> origin)
    {
		type_comparer<value_type> comparer;

        // now choose random points in the section
        // relative to the original
        index<rank> set_original_on_gpu = details::random_index(origin, section.view().get_extent());
        index<rank> set_original_on_cpu = details::random_index(origin, section.view().get_extent());

        // relative to the section
        index<rank> set_section_on_gpu = details::random_index(origin, section.view().get_extent()) - origin;
        index<rank> set_section_on_cpu = details::random_index(origin, section.view().get_extent()) - origin;

        // set a value in the original on the CPU
        value_type expected_value = static_cast<value_type>(rand());
        Log(LogType::Info, true) << "Setting a value in the original AV on the CPU" << std::endl;
        original.view()[set_original_on_cpu] = expected_value;
        original.set_known_value(set_original_on_cpu, expected_value);
        value_type actual_value = section.view()[set_original_on_cpu - origin];
        if (!comparer.are_equal(actual_value, expected_value))
        {
            Log(LogType::Error, true) << "Reading section (CPU) expected: " << expected_value << " actual: " << actual_value << std::endl;
            return false;
        }

        // set a value in the section on the GPU
        expected_value = static_cast<value_type>(rand());
        Log(LogType::Info, true) << "Setting a value in the section AV on the GPU" << std::endl;
        details::gpu_write(section.view(), set_section_on_gpu, expected_value);
        section.set_known_value(set_section_on_gpu, expected_value);
        actual_value = original.view()[set_section_on_gpu + origin];
        if (!comparer.are_equal(actual_value, expected_value))
        {
            Log(LogType::Error, true) << "Reading original (CPU) expected: " << expected_value << " actual: " << actual_value << std::endl;
            return false;
        }

        // set a value in the original on the GPU
        expected_value = static_cast<value_type>(rand());
        Log(LogType::Info, true) << "Setting a value in the original AV on the GPU" << std::endl;
        details::gpu_write(original.view(), set_original_on_gpu, expected_value);
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
        actual_value = details::gpu_read(original.view(), set_section_on_gpu + origin);
        if (!comparer.are_equal(actual_value, expected_value))
        {
            Log(LogType::Error, true) << "Reading original (GPU) expected: " << expected_value << " actual: " << actual_value << std::endl;
            return false;
        }

        return true;
    }

    ///<summary>
    /// A test class for verifying that sections of an ArrayView either do (positive) or do not (overlap)
    ///</summary>
    template<typename value_type, int rank>
    class OverlapTest
    {
    public:
        OverlapTest(ArrayViewTest<value_type, rank> original) :
            _original(original)
        {
        };

        OverlapTest(extent<rank> original_extent) :
            _original(original_extent)
        {
        };

        ArrayViewTest<value_type, rank>& original()
        {
            return _original;
        }

        bool positive_test(index<rank> local_origin, extent<rank> local_extent, index<rank> remote_origin, extent<rank> remote_extent)
        {
            return positive_test(_original.section(local_origin, local_extent), _original.section(remote_origin, remote_extent));
        }

        ///<summary>Returns true if these sections are verified to overlap</summary>
        template<typename local_view_type, typename remote_view_type>
        bool positive_test(local_view_type local, remote_view_type remote)
        {
            // initialized a constant value for the whole array
            value_type local_value = rand();
            write_local(local_value);

            // create some pending writes on the GPU
            write_remote(remote);

            // now try to read locally (implicit synchronize)
            Log(LogType::Info, true) << "Performing implicit synchronize on the local section" << std::endl;
            local.view()[typename local_view_type::index_type()];

            // for the positive case, the local section should be updated with some of the known
            // values
            Log(LogType::Info, true) << "Looking for changes in local_view" << std::endl;
            int known_values_checked = 0;
            index_iterator<local_view_type::rank> iter(local.view().get_extent());
            auto unexpected_changes = std::count_if(iter.begin(), iter.end(), [=, &known_values_checked] (typename local_view_type::index_type i) {

                // use the underlying data to access
                value_type actual_value = _original.data()[local.coordinates().get_linear(i)];

                // if this is part of the overlapping region, it's a known values
                index<rank> absolute_index = local.coordinates().get_absolute(i);
                auto value_iter = _original.known_values().find(absolute_index);

                if (value_iter == _original.known_values().end())
                {
                    // this value should not have changed
                    if (local_value != actual_value)
                    {
                        Log(LogType::Error, true) << "Mismatch found at: " << i << " (Local section AV) " << absolute_index <<
                            " (original AV) Expected: " << local_value << " Actual: " << actual_value << std::endl;
                        return true;
                    }
                }
                else
                {
                    known_values_checked++;
                    value_type expected_value = value_iter->second;
                    // this value should have changed
                    if (expected_value != actual_value)
                    {
                        Log(LogType::Error, true) << "Mismatch found at: " << i << " (Local section AV) " << absolute_index <<
                            " (original AV) Expected: " << expected_value << " Actual: " << actual_value << std::endl;
                        return true;
                    }
                    else
                    {
                        Log(LogType::Info, true) << "Value of: " << expected_value << " at: " << i << " (Local section AV) was correctly copied" << std::endl;
                    }
                }

                return false;
            });

            Log(LogType::Info, true) << unexpected_changes << " unexpected changes found" << std::endl;

            if (known_values_checked == 0)
            {
                Log(LogType::Error, true) << "0 known values in the local view range, the views do not overlap" << std::endl;
            }
            else
            {
                Log(LogType::Info, true) << known_values_checked << " overlapping elements found" << std::endl;
            }

            // now refresh the remote view
            Log(LogType::Info, true) << "Performing implicit synchronize on the remote section" << std::endl;
            remote.view()[typename remote_view_type::index_type()];

            return unexpected_changes == 0 && known_values_checked > 0;
        };

        bool negative_test(index<rank> local_origin, extent<rank> local_extent, index<rank> remote_origin, extent<rank> remote_extent)
        {
			auto localsect = _original.section(local_origin, local_extent);
			auto remotesect = _original.section(remote_origin, remote_extent);
            return negative_test(localsect, remotesect);
        }

        ///<summary>Returns false if these sections are verified to not overlap</summary>
        template<typename local_view_type, typename remote_view_type>
        bool negative_test(local_view_type &local, remote_view_type &remote)
        {
            // initialized a constant value for the whole array
            value_type local_value = rand();
            write_local(local_value);

            // create some pending writes on the GPU
            write_remote(remote);

            // now try to read locally (implicit synchronize)
            Log(LogType::Info, true) << "Performing implicit synchronize on the local section" << std::endl;
            local.view()[typename local_view_type::index_type()];

            // for the negative case, no change should occur -- this will count the changes
            Log(LogType::Info, true) << "Looking for changes in local_view" << std::endl;
            index_iterator<local_view_type::rank> iter(local.view().get_extent());
            auto changes = std::count_if(iter.begin(), iter.end(), [=] (typename local_view_type::index_type i) {

                // use the underlying data to access
                value_type actual_value = _original.data()[local.coordinates().get_linear(i)];
                if (local_value != actual_value)
                {
                    Log(LogType::Error, true) << "Mismatch found at: " << i << " (Local section AV) Expected: "
                        << local_value << " Actual: " << actual_value << std::endl;
                    return true;
                }

                return false;
            });

            Log(LogType::Info, true) << changes << " changes found" << std::endl;

            // now refresh the remote view
            Log(LogType::Info, true) << "Performing implicit synchronize on the remote section" << std::endl;
            remote.view()[index<remote_view_type::rank>()];

            return changes == 0;
        }

        int pass()
        {
            return _original.pass();
        }

        int fail()
        {
            return _original.fail();
        }

    private:

        void write_local(value_type value)
        {
            Log(LogType::Info, true) << "writing a constant value of: " << value << " locally" << std::endl;
            std::fill(_original.begin(), _original.end(), value);
        }

        // write to the remote array_view on the GPU
        template<typename remote_view_type>
        void write_remote(remote_view_type &remote)
        {
            accelerator accel = require_device(device_flags::NOT_SPECIFIED);
            std::vector<value_type> random_data(remote.view().get_extent().size());
            Fill(random_data);

            array_view<value_type, remote_view_type::rank> random_av(remote.view().get_extent(), random_data);
            array_view<value_type, remote_view_type::rank> remote_av = remote.view();

            Log(LogType::Info, true) << "Writing random data to AV: " << remote_av.get_extent() << std::endl;
            parallel_for_each(accel.get_default_view(), remote_av.get_extent(), [=](typename remote_view_type::index_type i) restrict(amp) {
                remote_av[i] = random_av[i];
            });

            index_iterator<remote_view_type::rank> iter(remote.view().get_extent());
            for (auto i = iter.begin(); i != iter.end(); i++)
            {
                remote.set_known_value(*i,  random_av[*i]);
            }
        };

        ArrayViewTest<value_type, rank> _original;
    };
}
}
