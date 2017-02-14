
// RUN: %hc %s -o %t.out && %t.out

//----------------------------------------------------------------------------
// File: Reduction.cpp
// 
// Implements several different versions of reduction algorithm in C++ AMP.
// Each consequent version demonstrates optimization techniques to speed up
// the execution. All of the algorithms are tailored to 4-byte wide data and
// may require tuning when different element type is to be used.
// The content of the GPU-side data structures is lost during the algorithm
// processing.
// Note: these implementations are for demo purposes only and not recommended
// to use in production code.
//----------------------------------------------------------------------------

#define NOMINMAX
#include <hc.hpp>
#include <iostream>
#include <numeric>
#include <assert.h>
#include <climits>

using namespace hc;

//----------------------------------------------------------------------------
// Helper macro to tell whether the argument is a positive integer
// power of two.
//----------------------------------------------------------------------------
#define IS_POWER_OF_2(arg) (arg > 1 && (arg & (arg - 1)) == 0)

//----------------------------------------------------------------------------
// Helper function checking the common preconditions for tiled algorithms.
// If the conditions are not met, the implementations will fall back to CPU
// reduction after some number of iterations using parallel_for_each.
// As it may be unexpected, a warning message will be issued.
// Addressing these limitations is left as an exercise for the reader.
//----------------------------------------------------------------------------
inline bool check_tiled_precondition(unsigned tile_size, unsigned element_count)
{
    while ((element_count % tile_size) == 0)
    {
        element_count /= tile_size;
    }
    return element_count < tile_size;
}

//----------------------------------------------------------------------------
// This is a simple sequential implementation.
//----------------------------------------------------------------------------
float sequential_reduction(const std::vector<float>& source)
{
    return std::accumulate(source.begin(), source.end(), 0.f);
}

//----------------------------------------------------------------------------
// This is an implementation of the reduction algorithm using a simple
// parallel_for_each. Multiple kernel launches are required to synchronize
// memory access among threads in separate tiles.
//----------------------------------------------------------------------------
float reduction_simple_1(const std::vector<float>& source)
{
    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.
    if (element_count == 1)
    {
        return source[0];
    }

    // Using array, as we mostly need just temporary memory to store
    // the algorithm state between iterations and in the end we have to copy
    // back only the first element.
    array<float, 1> a(element_count, source.begin());

    // Takes care of odd input elements – we could completely avoid tail sum
    // if we would require source to have even number of elements.
    float tail_sum = (element_count % 2) ? source[element_count - 1] : 0;
    array_view<float, 1> av_tail_sum(1, &tail_sum);

    // Each thread reduces two elements.
    for (unsigned s = element_count / 2; s > 0; s /= 2)
    {
        parallel_for_each(extent<1>(s), [=, &a] (index<1> idx) __attribute((hc))
        {
            a[idx] = a[idx] + a[idx + s];

            // Reduce the tail in cases where the number of elements is odd.
            if ((idx[0] == s - 1) && (s & 0x1) && (s != 1))
            {
                av_tail_sum[0] += a[s - 1];
            }
        }).wait();
    }

    // Copy the results back to CPU.
    std::vector<float> result(1);
    copy(a.section(0, 1), result.begin());
    av_tail_sum.synchronize();

    return result[0] + tail_sum;
}

//----------------------------------------------------------------------------
// This is an improved implementation of the reduction algorithm using
// a simple parallel_for_each. Each thread is reducing more elements,
// decreasing the total number of memory accesses.
//----------------------------------------------------------------------------
float reduction_simple_2(const std::vector<float>& source)
{
    const unsigned window_width = 8;
    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.

    // Using array as temporary memory.
    array<float, 1> a(element_count, source.begin());

    // Takes care of the sum of tail elements.
    float tail_sum = 0.f;
    if((element_count % window_width) != 0 && element_count > window_width)
    {
        tail_sum = std::accumulate(source.begin() + ((element_count - 1) / window_width) * window_width, source.end(), 0.f);
    }
    array_view<float, 1> av_tail_sum(1, &tail_sum);

    // Each thread reduces window_width elements.
    unsigned prev_s = element_count;
    for (unsigned s = element_count / window_width; s > 0; s /= window_width)
    {
        parallel_for_each(extent<1>(s), [=, &a] (index<1> idx) __attribute((hc))
        {
            float sum = 0.f;
            for(unsigned i = 0; i < window_width; i++)
            {
                sum += a[idx + i * s];
            }
            a[idx] = sum;

            // Reduce the tail in cases where the number of elements is not divisible.
            // Note: execution of this section may negatively affect the performance.
            // In production code the problem size passed to the reduction should
            // be a power of the window_width. Please refer to the blog post for more
            // information.
            if ((idx[0] == s - 1) && ((s % window_width) != 0) && (s > window_width))
            {
                for(unsigned i = ((s - 1) / window_width) * window_width; i < s; i++)
                {
                    av_tail_sum[0] += a[i];
                }
            }
        }).wait();
        prev_s = s;
    }

    // Perform any remaining reduction on the CPU.
    std::vector<float> result(prev_s);
    copy(a.section(0, prev_s), result.begin());
    av_tail_sum.synchronize();

    return std::accumulate(result.begin(), result.end(), tail_sum);
}

//----------------------------------------------------------------------------
// This is an implementation of the reduction algorithm which uses tiling and 
// the shared memory.
//----------------------------------------------------------------------------
template <unsigned _tile_size>
float reduction_tiled_1(const std::vector<float>& source)
{
    static_assert(IS_POWER_OF_2(_tile_size), "Tile size must be a positive integer power of two!");

    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.

    if (!check_tiled_precondition(_tile_size, element_count))
    {
        std::cout << "Warning, reduction_tiled_1 is not designed for the current problem size." << std::endl;
    }

    // Using arrays as temporary memory.
    array<float, 1> arr_1(element_count, source.begin());
    array<float, 1> arr_2((element_count / _tile_size) ? (element_count / _tile_size) : 1);
    
    // array_views may be swapped after each iteration.
    array_view<float, 1> av_src(arr_1);
    array_view<float, 1> av_dst(arr_2);
    av_dst.discard_data();

    // Reduce using parallel_for_each as long as the sequence length
    // is evenly divisable to the number of threads in the tile.
    while ((element_count % _tile_size) == 0)
    {
        parallel_for_each(extent<1>(element_count).tile(_tile_size), [=] (tiled_index<1> tidx) __attribute((hc))
        {
            // Use tile_static as a scratchpad memory.
            tile_static float tile_data[_tile_size];

            unsigned local_idx = tidx.local[0];
            tile_data[local_idx] = av_src[tidx.global];
            tidx.barrier.wait();

            // Reduce within a tile using multiple threads.
            for(unsigned s = 1; s < _tile_size; s *= 2)
            {
                if (local_idx % (2 * s) == 0)
                {
                    tile_data[local_idx] += tile_data[local_idx + s];
                }
                
                tidx.barrier.wait();
            }

            // Store the tile result in the global memory.
            if (local_idx == 0)
            {
                av_dst[tidx.tile] = tile_data[0];
            }
        }).wait();

        // Update the sequence length, swap source with destination.
        element_count /= _tile_size;
        std::swap(av_src, av_dst);
        av_dst.discard_data();
    }

    // Perform any remaining reduction on the CPU.
    std::vector<float> result(element_count);
    copy(av_src.section(0, element_count), result.begin());
    return std::accumulate(result.begin(), result.end(), 0.f);
}

//----------------------------------------------------------------------------
// This is a version with a better resource utilization, as only a group
// of consequtive threads become active - minimizing the divergence within
// the scheduling units.
//----------------------------------------------------------------------------
template <unsigned _tile_size>
float reduction_tiled_2(const std::vector<float>& source)
{
    static_assert(IS_POWER_OF_2(_tile_size), "Tile size must be a positive integer power of two!");

    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.

    if (!check_tiled_precondition(_tile_size, element_count))
    {
        std::cout << "Warning, reduction_tiled_2 is not designed for the current problem size." << std::endl;
    }

    // Using arrays as temporary memory.
    array<float, 1> arr_1(element_count, source.begin());
    array<float, 1> arr_2((element_count / _tile_size) ? (element_count / _tile_size) : 1);

    // array_views may be swapped after each iteration.
    array_view<float, 1> av_src(arr_1);
    array_view<float, 1> av_dst(arr_2);
    av_dst.discard_data();

    // Reduce using parallel_for_each as long as the sequence length
    // is evenly divisable to the number of threads in the tile.
    while ((element_count % _tile_size) == 0)
    {
        parallel_for_each(extent<1>(element_count).tile(_tile_size), [=] (tiled_index<1> tidx) __attribute((hc))
        {
            // Use tile_static as a scratchpad memory.
            tile_static float tile_data[_tile_size];

            unsigned local_idx = tidx.local[0];
            tile_data[local_idx] = av_src[tidx.global];
            tidx.barrier.wait();

            // Reduce within a tile using multiple threads.
            for(unsigned s = 1; s < _tile_size; s *= 2)
            {
                unsigned index = 2 * s * local_idx;
                if (index < _tile_size)
                {
                    tile_data[index] += tile_data[index + s];
                }

                tidx.barrier.wait();
            }

            // Store the tile result in the global memory.
            if (local_idx == 0)
            {
                av_dst[tidx.tile] = tile_data[0];
            }
        }).wait();

        // Update the sequence length, swap source with destination.
        element_count /= _tile_size;
        std::swap(av_src, av_dst);
        av_dst.discard_data();
    }

    // Perform any remaining reduction on the CPU.
    std::vector<float> result(element_count);
    copy(av_src.section(0, element_count), result.begin());
    return std::accumulate(result.begin(), result.end(), 0.f);
}

//----------------------------------------------------------------------------
// This is a version without bank conflicts issue.
//----------------------------------------------------------------------------
template <unsigned _tile_size>
float reduction_tiled_3(const std::vector<float>& source)
{
    static_assert(IS_POWER_OF_2(_tile_size), "Tile size must be a positive integer power of two!");

    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.

    if (!check_tiled_precondition(_tile_size, element_count))
    {
        std::cout << "Warning, reduction_tiled_3 is not designed for the current problem size." << std::endl;
    }

    // Using arrays as temporary memory.
    array<float, 1> arr_1(element_count, source.begin());
    array<float, 1> arr_2((element_count / _tile_size) ? (element_count / _tile_size) : 1);

    // array_views may be swapped after each iteration.
    array_view<float, 1> av_src(arr_1);
    array_view<float, 1> av_dst(arr_2);
    av_dst.discard_data();

    // Reduce using parallel_for_each as long as the sequence length
    // is evenly divisable to the number of threads in the tile.
    while ((element_count % _tile_size) == 0)
    {
        parallel_for_each(extent<1>(element_count).tile(_tile_size), [=] (tiled_index<1> tidx) __attribute((hc))
        {
            // Use tile_static as a scratchpad memory.
            tile_static float tile_data[_tile_size];

            unsigned local_idx = tidx.local[0];
            tile_data[local_idx] = av_src[tidx.global];
            tidx.barrier.wait();

            // Reduce within a tile using multiple threads.
            for(unsigned s = _tile_size / 2; s > 0; s /= 2)
            {
                if (local_idx < s)
                {
                    tile_data[local_idx] += tile_data[local_idx + s];
                }

                tidx.barrier.wait();
            }

            // Store the tile result in the global memory.
            if (local_idx == 0)
            {
                av_dst[tidx.tile] = tile_data[0];
            }
        }).wait();

        // Update the sequence length, swap source with destination.
        element_count /= _tile_size;
        std::swap(av_src, av_dst);
        av_dst.discard_data();
    }

    // Perform any remaining reduction on the CPU.
    std::vector<float> result(element_count);
    copy(av_src.section(0, element_count), result.begin());
    return std::accumulate(result.begin(), result.end(), 0.f);
}

//----------------------------------------------------------------------------
// This is a version with a reduced number of stalled threads in the first
// iteration.
//----------------------------------------------------------------------------
template <unsigned _tile_size>
float reduction_tiled_4(const std::vector<float>& source)
{
    static_assert(IS_POWER_OF_2(_tile_size), "Tile size must be a positive integer power of two!");

    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.

    if (!check_tiled_precondition(_tile_size * 2, element_count))
    {
        std::cout << "Warning, reduction_tiled_4 is not designed for the current problem size." << std::endl;
    }

    // Using arrays as temporary memory.
    array<float, 1> arr_1(element_count, source.begin());
    array<float, 1> arr_2((element_count / _tile_size) ? (element_count / _tile_size) : 1);
    
    // array_views may be swapped after each iteration.
    array_view<float, 1> av_src(arr_1);
    array_view<float, 1> av_dst(arr_2);
    av_dst.discard_data();

    // Reduce using parallel_for_each as long as the sequence length
    // is evenly divisable to twice the number of threads in the tile
    // (note each tile works on a problem size twice its own size).
    while (element_count >= _tile_size
        && (element_count % (_tile_size * 2)) == 0)
    {
        parallel_for_each(extent<1>(element_count / 2).tile(_tile_size), [=] (tiled_index<1> tidx) __attribute((hc))
        {
            // Use tile_static as a scratchpad memory.
            tile_static float tile_data[_tile_size];

            unsigned local_idx = tidx.local[0];

            // Partition input data among tiles,
            // 2 * _tile_size because the number of threads spawned is halved.
            unsigned rel_idx = tidx.tile[0] * (_tile_size * 2) + local_idx;
            tile_data[local_idx] = av_src[rel_idx] + av_src[rel_idx + _tile_size];
            tidx.barrier.wait();

            // Reduce within a tile using multiple threads.
            for(unsigned s = _tile_size / 2; s > 0; s /= 2)
            {
                if (local_idx < s)
                {
                    tile_data[local_idx] += tile_data[local_idx + s];
                }

                tidx.barrier.wait();
            }

            // Store the tile result in the global memory.
            if (local_idx == 0)
            {
                av_dst[tidx.tile] = tile_data[0];
            }
        }).wait();

        // Update the sequence length, swap source with destination.
        element_count /= _tile_size * 2;
        std::swap(av_src, av_dst);
        av_dst.discard_data();
    }

    // Perform any remaining uneven reduction on the CPU.
    std::vector<float> result(element_count);
    copy(av_src.section(0, element_count), result.begin());
    return std::accumulate(result.begin(), result.end(), 0.f);
}

//----------------------------------------------------------------------------
// Here we take completely different approach by using algorithm cascading
// by combining sequential and parallel reduction.
//----------------------------------------------------------------------------
template <unsigned _tile_size, unsigned _tile_count>
float reduction_cascade(const std::vector<float>& source)
{
    static_assert(_tile_count > 0, "Tile count must be positive!");
    static_assert(IS_POWER_OF_2(_tile_size), "Tile size must be a positive integer power of two!");

    assert(source.size() <= UINT_MAX);
    unsigned element_count = static_cast<unsigned>(source.size());
    assert(element_count != 0); // Cannot reduce an empty sequence.

    unsigned stride = _tile_size * _tile_count * 2;

    // Reduce tail elements.
    float tail_sum = 0.f;
    unsigned tail_length = element_count % stride;
    if(tail_length != 0)
    {
        tail_sum = std::accumulate(source.end() - tail_length, source.end(), 0.f);
        element_count -= tail_length;
        if(element_count == 0)
        {
            return tail_sum;
        }
    }

    // Using arrays as a temporary memory.
    array<float, 1> a(element_count, source.begin());
    array<float, 1> a_partial_result(_tile_count);

    parallel_for_each(extent<1>(_tile_count * _tile_size).tile(_tile_size), [=, &a, &a_partial_result] (tiled_index<1> tidx) __attribute((hc))
    {
        // Use tile_static as a scratchpad memory.
        tile_static float tile_data[_tile_size];

        unsigned local_idx = tidx.local[0];

        // Reduce data strides of twice the tile size into tile_static memory.
        unsigned input_idx = (tidx.tile[0] * 2 * _tile_size) + local_idx;
        tile_data[local_idx] = 0;
        do
        {
            tile_data[local_idx] += a[input_idx] + a[input_idx + _tile_size]; 
            input_idx += stride;
        } while (input_idx < element_count);

        tidx.barrier.wait();

        // Reduce to the tile result using multiple threads.
        for (unsigned stride = _tile_size / 2; stride > 0; stride /= 2)
        {
            if (local_idx < stride)
            {
                tile_data[local_idx] += tile_data[local_idx + stride];
            }

            tidx.barrier.wait();
        }

        // Store the tile result in the global memory.
        if (local_idx == 0)
        {
            a_partial_result[tidx.tile[0]] = tile_data[0];
        }
    }).wait();

    // Reduce results from all tiles on the CPU.
    std::vector<float> v_partial_result(_tile_count);
    copy(a_partial_result, v_partial_result.begin());
    return std::accumulate(v_partial_result.begin(), v_partial_result.end(), tail_sum);
}

//----------------------------------------------------------------------------
// Helper function comparing floating point numbers within a given relative
// difference.
//----------------------------------------------------------------------------
bool fp_equal(float a, float b, float max_rel_diff)
{
    float diff = std::fabs(a - b);
    a = std::fabs(a);
    b = std::fabs(b);
    return diff <= std::max(a, b) * max_rel_diff;
}

//----------------------------------------------------------------------------
// Program entry point.
//----------------------------------------------------------------------------
int main()
{
    int ret = 0;
    accelerator default_device;
    std::wcout << "Using device : " << default_device.get_description() << std::endl;
//    if (default_device == accelerator(accelerator::direct3d_ref))
//        std::cout << "WARNING!! Running on very slow emulator! Only use this accelerator for debugging." << std::endl;

    // Make sure that elements can be split into tiles so the number of
    // tiles in any dimension is less than 65536. Here we we have
    // element_count == 16777216 so the number of tiles:
    // tile_count = element_count / tile_size == 32768 < 65536
    unsigned element_count = 16 * 1024 * 1024;

    std::vector<float> source(element_count);
    for (unsigned i = 0; i < element_count; ++i)
    {
        // Element range is limited to avoid overflow or underflow
        source[i] = (i & 0xf) * 0.01f;
    }

    // The data is generated in a pattern and its sum can be computed by the following formula
    const float expected_result = ((element_count / 16) * ((15 * 16) / 2)) * 0.01f;

    std::cout << "Running kernels..." << std::endl;

    const unsigned tile_size = 512;

    typedef float (*ReductionFunction)(const std::vector<float>&);
    typedef std::pair<ReductionFunction, std::string> user_pair;
    std::vector<user_pair> functions;
    functions.push_back(user_pair(sequential_reduction, "sequential_reduction"));
    functions.push_back(user_pair(reduction_simple_1, "reduction_simple_1"));
    functions.push_back(user_pair(reduction_simple_2, "reduction_simple_2"));
    functions.push_back(user_pair(reduction_tiled_1<tile_size>, "reduction_tiled_1"));
    functions.push_back(user_pair(reduction_tiled_2<tile_size>, "reduction_tiled_2"));
    functions.push_back(user_pair(reduction_tiled_3<tile_size>, "reduction_tiled_3"));
    functions.push_back(user_pair(reduction_tiled_4<tile_size>, "reduction_tiled_4"));
    functions.push_back(user_pair(reduction_cascade<tile_size, 128>, "reduction_cascade"));
    for(const auto& func : functions)
    {
        float result = func.first(source);

        if (fp_equal(result, expected_result, 0.05f))
        {
            std::cout << "SUCCESS: " << func.second << "." << std::endl;
        }
        else
        {
            std::cout << "FAILED: " << func.second << " expected " << expected_result << " but found " << result << "!" << std::endl;
            ret = 1;
        }
    }
    return ret;
}
