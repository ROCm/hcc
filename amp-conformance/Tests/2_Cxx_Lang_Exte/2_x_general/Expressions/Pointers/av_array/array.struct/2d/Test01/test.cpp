// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>struct has array reference as members. 2d.</summary>

#include <cmath>
#include "../av.h"

template<typename type>
void init(vector<type> &a, vector<type> &b, vector<type> &c, vector<type> &fa, vector<type> &fb, vector<type> &fc, vector<type> &ref_c, vector<int> &flag)
{
    srand(2010);
    size_t SIZE = a.size();

    Fill<type>(a, 0, SIZE - 1);
    Fill<type>(b, 0, SIZE - 1);

    for (size_t i = 0; i < SIZE; i++)
    {
        fa[i] = a[i] - 1;
        fb[i] = b[i] - 1;
        ref_c[i] = std::modf(a[i], &b[i]) * LOCAL_SIZE * LOCAL_SIZE; // Because in kernel_local, the results have been added up. So here it needs multiplication.
    }

    flag[0] = 10;
    flag[1] = 12;
    flag[2] = 20;
    flag[3] = 22;
    flag[4] = 0;
    flag[5] = 2;
    flag[6] = 0;
}

template<typename type>
void cf_test(type *pa, type *pb, type *pc, array_view<int, 1> &flag) __GPU_ONLY
{
    for (int i = flag[0]; i < flag[1]; i++)
    {
        for (int j = flag[2]; j < flag[3]; j++)
        {
            switch (flag[4])
            {
            case 1:
                {
                    pa++;
                }
                break;
            default:
                {
                    switch (flag[5])
                    {
                    case 2:
                        {
                            switch (flag[6])
                            {
                            case 1:
                                {
                                    pa++;
                                }
                                break;
                            default:
                                {
                                    *pc = precise_math::modf(*pa, pb);
                                    return;
                                }
                                break;
                            }
                        }
                        break;
                    default:
                        pa++;
                        break;
                    }
                }
                break;

            }
        }
    }
    *pc = 0;

}

