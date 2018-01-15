// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Multiple restrict spec on a recursive template lead to base-case instantiation fail from within a parallel_for_each</summary>

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;
using namespace std;

template< int N >
int recurse() restrict( cpu, amp ) {
    return recurse< N-1 >();
}

template<>
int recurse< 0 >() restrict( cpu, amp ) {
    return 0x600DF00D;
}

runall_result test_main() {

    vector<int> result;
    Concurrency::array< int, 1 > compute_result( 1 );

    Concurrency::parallel_for_each( compute_result.get_extent(), [ &compute_result ]( Concurrency::index< 1 > idx ) restrict( amp ) {
        compute_result[ idx ] = recurse< 20 >();
    });
    result = compute_result;

    // Compile-time test. If the compile succeeds, test passes.
    return runall_pass;
}

