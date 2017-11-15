// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amptest\data.h
*
**********************************************************************************/

// Attach the dpctest.lib
#include <amptest/platform.h>

#include <random>
#include <vector>

namespace Concurrency
{
    namespace Test
    {

        // Details namespace serves as private namespace
        namespace details
        {
			template<typename T> inline T get_default_fill_min() { return (std::is_unsigned<T>::value) ? 0 : -(1 << 15); }
			template<typename T> inline T get_default_fill_max() { return  1 << 15; }

#pragma warning( push )
#pragma warning( disable:4309 ) // 'return' : truncation of constant value

			template<> inline char get_default_fill_min<char>(){ return 0x80; }
			template<> inline char get_default_fill_max<char>(){ return 0x7F; }

			template<> inline short get_default_fill_min<short>(){ return 0x8000; }
			template<> inline short get_default_fill_max<short>(){ return 0x7FFF; }

			template<> inline unsigned char get_default_fill_min<unsigned char>(){ return 0x00; }
			template<> inline unsigned char get_default_fill_max<unsigned char>(){ return 0xFF; }

			template<> inline unsigned short get_default_fill_min<unsigned short>(){ return 0x00; }
			template<> inline unsigned short get_default_fill_max<unsigned short>(){ return 0xFFFF; }

			// These min/max for floating-point types come from the original template specialization functions
			template<> inline float get_default_fill_min<float>(){ return -(1 << 15); }
			template<> inline float get_default_fill_max<float>(){ return  1 << 15; }

			template<> inline double get_default_fill_min<double>(){ return -(1 << 15); }
			template<> inline double get_default_fill_max<double>(){ return  1 << 15; }

#pragma warning( pop )

            template<typename T>
            void FillFloatingPoint(T *arr, size_t size, T min, T max)
            {
				static_assert(!std::is_const<T>::value, "Cannot use Fill with 'const' value type");

				std::mt19937 mersenne_twister_engine;
                std::uniform_real_distribution<T> uni(min, max);

                for(size_t i = 0; i < size; ++i)
                {
                    arr[i] = uni(mersenne_twister_engine);
                }
            }

            template<typename T>
            void FillIntegral(T *arr, size_t size, T min, T max)
            {
				static_assert(!std::is_const<T>::value, "Cannot use Fill with 'const' value type");

				std::mt19937 mersenne_twister_engine;
                std::uniform_int_distribution<T> uni(min, max);

                for(size_t i = 0; i < size; ++i)
                {
                    arr[i] = uni(mersenne_twister_engine);
                }
            }

        } // namespace details

        template<typename T>
        void PackRandom(T* packed, const size_t size, size_t bitsPerT, T min = 0, T max = 1 << 15)
        {
            if(bitsPerT != 8 && bitsPerT != 16 && bitsPerT != 32) throw;

            const unsigned int numPacked = (sizeof(T) * 8 ) / static_cast<unsigned int>(bitsPerT);

            const size_t unpackedSize = size * numPacked;
            T* unpacked = new T[unpackedSize];

            details::FillIntegral<T>(unpacked, unpackedSize, min, max);

            size_t p = 0, up = 0;
            while(up < unpackedSize)
            {
                 int packedVal = 0;
                 for(size_t i = 0; i < numPacked; i++)
                 {
                    packedVal <<= bitsPerT;
                    T next = unpacked[up++];
                    packedVal |= (next & max);
                 }
                 packed[p++] = packedVal;
            }

            delete[] unpacked;
        }

        template<typename T>
        void Unpack(T* unpacked, T* packed, size_t packedsize, size_t bitsPerT)
        {
            if(bitsPerT != 8 && bitsPerT != 16 && bitsPerT != 32) throw;

            const int numPacked = (sizeof(T) * 8 ) / static_cast<int>(bitsPerT);
            const int max = (1 << bitsPerT) - 1;
            const unsigned int unpackedSize = static_cast<int>(packedsize) * numPacked;

            size_t p = 0, up = 0;
            while(up < unpackedSize)
            {
                int packedVal = packed[p++];
                for(int j = numPacked - 1; j >= 0; j--)
                {
                    T next = (packedVal & max);
                    unpacked[up+j] = next;
                    packedVal >>= bitsPerT;
                }
                up+= numPacked;
            }
        }


        // Fill functions for c-style arrays
        template<typename T>
        inline void Fill(T *arr, size_t size, T min, T max)
        {
            return details::FillIntegral(arr, size, min, max);
        }

        template<>
        inline void Fill(float *arr, size_t size, float min, float max)
        {
            return details::FillFloatingPoint(arr, size, min, max);
        }

        template<>
        inline void Fill(double *arr, size_t size, double min, double max)
        {
            return details::FillFloatingPoint(arr, size, min, max);
        }

        template<typename T>
        inline void Fill(T *arr, size_t size)
        {
            T min_val = details::get_default_fill_min<T>();
            T max_val = details::get_default_fill_max<T>();
            return Fill(arr, size, min_val, max_val);
        }

        // End of Fill functions for c-style arrays

        // Fill functions for std::vector

        template<typename T>
        inline void Fill(std::vector<T> &arr, T min, T max)
        {
            return Fill(arr.data(), arr.size(), min, max);
        }

        template<typename T>
        inline void Fill(std::vector<T> &arr)
        {
            return Fill(arr.data(), arr.size());
        }
		
        // End of Fill functions for std::vector

		#pragma region amptest_static_cast<T>()
		// TODO: Create tests for this function. Ensure that the T=double functions only require limited_double_support.

		/// Template function that uses static_cast in a safe manner.
		/// The main purpose of this is to cast a value to a double from an
		/// integral data type w/o needing full double support on the GPU.
		/// NOTE: Do not ALWAYS use this function as it avoids casting from int to dbl, which will lower our coverage of this operation.
		/// This is intended to allow code to not require full-double support if the only full-double operation being done is
		/// converting an int/uint to double.
		template <typename T, typename Tsrc>
		inline T amptest_static_cast(const Tsrc& src) restrict(cpu,amp) {
			return static_cast<T>(src); // By default, just use the normal static_cast
		}

		// WARNING: Since we cast to a float first, the actual range of src is that of full integral
		// values of the float data type.
		template <> inline double amptest_static_cast<double, int>(const int& src) restrict(cpu,amp) {
			// On the GPU, casting from int/uint to double requires full double support.
			// The following here only requires limited double support:
			return static_cast<double>(static_cast<float>(src));
		}

		// WARNING: Since we cast to a float first, the actual range of src is that of full integral
		// values of the float data type.
		template <> inline double amptest_static_cast<double, unsigned int>(const unsigned int& src) restrict(cpu,amp) {
			// On the GPU, casting from int/uint to double requires full double support.
			// The following here only requires limited double support:
			return static_cast<double>(static_cast<float>(src));
		}

		// WARNING: Since we cast to a float first, the actual range of src is that of full integral
		// values of the float data type.
		template <> inline int amptest_static_cast<int, double>(const double& src) restrict(cpu,amp) {
			// On the GPU, casting from int/uint to double requires full double support.
			// The following here only requires limited double support:
			return static_cast<int>(static_cast<float>(src));
		}

		// WARNING: Since we cast to a float first, the actual range of src is that of full integral
		// values of the float data type.
		template <> inline unsigned int amptest_static_cast<unsigned int, double>(const double& src) restrict(cpu,amp) {
			// On the GPU, casting from int/uint to double requires full double support.
			// The following here only requires limited double support:
			return static_cast<unsigned int>(static_cast<float>(src));
		}

		#pragma endregion

	}
}

