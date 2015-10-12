// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test pointer emulation with array_view and control flow, 3d. Use pointer to test CF. Use pointer to struct. av is in a struct. Use c array of struct which has av.
// It tests local, shared and global memory. cf: do do switch switch switch</summary>

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
        ref_c[i] = std::max(a[i], b[i] + 4) * LOCAL_SIZE * LOCAL_SIZE * LOCAL_SIZE; // Because in kernel_local, the results have been added up. So here it needs multiplication.
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
    int i4 = flag[0];
    do
    {
        i4++;
        int i5 = flag[2];
        do
        {
            i5++;
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
                                    (*pb)++;
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
        } while (i5 < flag[3]);
    } while (i4 < flag[1]);

    atomic_fetch_max(pa, *pb);
    *pc = *pa;
}

