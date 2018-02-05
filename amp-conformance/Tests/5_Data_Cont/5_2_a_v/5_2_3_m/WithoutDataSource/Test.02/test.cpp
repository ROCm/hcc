// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test array_views without a data source</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <amp_short_vectors.h>

#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

// Helper methods for some tests that rely on Direct3D interop
// TODO: neither of these make sense with HCC, hence why they are disabled,
//       pending refactoring.
//HRESULT CopyOut(ID3D11Device *pDevice, ID3D11Buffer *pBuffer, void *pData)
//{
//    if ((pDevice == NULL) || (pBuffer == NULL) || (pData == NULL)) {
//        return E_FAIL;
//    }
//
//    D3D11_BUFFER_DESC bufferDescription;
//    pBuffer->GetDesc(&bufferDescription);
//
//    D3D11_BUFFER_DESC stagingBufferDescription;
//    ZeroMemory(&stagingBufferDescription, sizeof(D3D11_BUFFER_DESC));
//    stagingBufferDescription.ByteWidth = bufferDescription.ByteWidth;
//    stagingBufferDescription.Usage = D3D11_USAGE_STAGING;
//    stagingBufferDescription.BindFlags = 0;
//    stagingBufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE | D3D11_CPU_ACCESS_READ;
//    stagingBufferDescription.MiscFlags = 0;
//
//    ID3D11Buffer *pStagingBuffer = NULL;
//    if (pDevice->CreateBuffer(&stagingBufferDescription, NULL, &pStagingBuffer) != S_OK) {
//        return E_FAIL;
//    }
//
//    ID3D11DeviceContext *pContext = NULL;
//    pDevice->GetImmediateContext(&pContext);
//
//    D3D11_BOX box;
//    box.left = 0;
//    box.top = 0;
//    box.front = 0;
//    box.right = bufferDescription.ByteWidth;
//    box.bottom = 1;
//    box.back = 1;
//    pContext->CopySubresourceRegion(pStagingBuffer, 0, 0, 0, 0, pBuffer, 0, &box);
//
//    D3D11_MAPPED_SUBRESOURCE dOutBuf;
//
//    if (pContext->Map(pStagingBuffer, 0, D3D11_MAP_WRITE, 0, &dOutBuf) != S_OK) {
//        pStagingBuffer->Release();
//        pContext->Release();
//        return E_FAIL;
//    }
//
//    memcpy(pData, dOutBuf.pData, bufferDescription.ByteWidth);
//    pContext->Unmap(pStagingBuffer, 0);
//
//    pStagingBuffer->Release();
//    pContext->Release();
//
//    return S_OK;
//}

// Helper function to get the ID3D11Device pointer corresponding to a concurrency::accelerator_view object
//ID3D11Device *get_d3d11_device(accelerator_view &av)
//{
//    IUnknown *pTemp = direct3d::get_device(av);
//    ID3D11Device *pDevice = NULL;
//    pTemp->QueryInterface(__uuidof(ID3D11Device), reinterpret_cast<void**>(&pDevice));
//    pTemp->Release();
//
//    return pDevice;
//}

// Helper function to get the ID3D11Buffer pointer corresponding to a concurrency::array object
//template<typename T, int Rank>
//ID3D11Buffer *get_d3d11_buffer(array<T, Rank> &arr)
//{
//    IUnknown *pTemp = direct3d::get_buffer(arr);
//    ID3D11Buffer *pBuffer = NULL;
//    pTemp->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&pBuffer));
//    pTemp->Release();
//
//    return pBuffer;
//}

// Basic test for an array_view without a data source
bool Test1(const accelerator_view &av)
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);
    array_view<int, 2> arrViewSum(M, N);
    array_view<int, 2> arrViewDiff(M, N);
    parallel_for_each(av, arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i) {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test section for an array_view without a data source
// Sections created outside the p_f_e
bool Test2()
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);
    array_view<int, 2> arrViewOut(2 * M, 2 * N);
    array_view<int, 2> arrViewSum = arrViewOut.section(0, 0, M , N);
    array_view<int, 2> arrViewDiff = arrViewOut.section(0, N, M , N);
    array_view<int, 2> arrViewMul = arrViewOut.section(M, 0, M , N);
    array_view<int, 2> arrViewDiv = arrViewOut.section(M, N, M , N);

    parallel_for_each(arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
        arrViewMul[idx] = arrViewA[idx] * arrViewB[idx];
        arrViewDiv[idx] = arrViewA[idx] / (arrViewB[idx] + 1);
    });

    // Now verify the results
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewMul(i / N, i % N) != (vecA[i] * vecB[i])) {
            Log(LogType::Error, true) << "Mul(" << i / N << ", " << i % N << ") = " << arrViewMul(i / N, i % N) << ", Expected = " << (vecA[i] * vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiv(i / N, i % N) != (vecA[i] / (vecB[i] + 1))) {
            Log(LogType::Error, true) << "Div(" << i / N << ", " << i % N << ") = " << arrViewDiv(i / N, i % N) << ", Expected = " << (vecA[i] / (vecB[i] + 1)) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test section for an array_view without a data source
// Sections created inside the p_f_e
bool Test3(const accelerator_view &av)
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);

    // Let's first cache the inputs on "av"
    array<int, 2> tempArr(M, N, av);
    parallel_for_each(av, tempArr.get_extent(), [=, &tempArr](const index<2> &idx) restrict(amp) {
        tempArr[idx] = arrViewA[idx] + arrViewB[idx];
    });

    array_view<int, 2> arrViewOut(2 * M, 2 * N);
    parallel_for_each(extent<2>(M, N), [=](const index<2> &idx) restrict(amp)
    {
        array_view<int, 2> arrViewSum = arrViewOut.section(0, 0, M , N);
        array_view<int, 2> arrViewDiff = arrViewOut.section(0, N, M , N);
        array_view<int, 2> arrViewMul = arrViewOut.section(M, 0, M , N);
        array_view<int, 2> arrViewDiv = arrViewOut.section(M, N, M , N);

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
        arrViewMul[idx] = arrViewA[idx] * arrViewB[idx];
        arrViewDiv[idx] = arrViewA[idx] / (arrViewB[idx] + 1);
    });

    // Now verify the results
    array_view<int, 2> arrViewSum = arrViewOut.section(0, 0, M , N);
    array_view<int, 2> arrViewDiff = arrViewOut.section(0, N, M , N);
    array_view<int, 2> arrViewMul = arrViewOut.section(M, 0, M , N);
    array_view<int, 2> arrViewDiv = arrViewOut.section(M, N, M , N);
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewMul(i / N, i % N) != (vecA[i] * vecB[i])) {
            Log(LogType::Error, true) << "Mul(" << i / N << ", " << i % N << ") = " << arrViewMul(i / N, i % N) << ", Expected = " << (vecA[i] * vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiv(i / N, i % N) != (vecA[i] / (vecB[i] + 1))) {
            Log(LogType::Error, true) << "Div(" << i / N << ", " << i % N << ") = " << arrViewDiv(i / N, i % N) << ", Expected = " << (vecA[i] / (vecB[i] + 1)) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test projection for an array_view without a data source
// Projections created outside the p_f_e
bool Test4()
{
    const int size = 2047;

    std::vector<int> vecA(size);
    std::vector<int> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int> arrViewA(size, vecA);
    array_view<const int> arrViewB(size, vecB);
    array_view<int, 2> arrViewOut(4, size);
    array_view<int> arrViewSum = arrViewOut[0];
    array_view<int> arrViewDiff = arrViewOut[1];
    array_view<int> arrViewMul = arrViewOut[2];
    array_view<int> arrViewDiv = arrViewOut[3];

    parallel_for_each(arrViewSum.get_extent(), [=](const index<1> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
        arrViewMul[idx] = arrViewA[idx] * arrViewB[idx];
        arrViewDiv[idx] = arrViewA[idx] / (arrViewB[idx] + 1);
    });

    // Now verify the results
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i << ") = " << arrViewSum(i) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i << ") = " << arrViewDiff(i) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewMul(i) != (vecA[i] * vecB[i])) {
            Log(LogType::Error, true) << "Mul(" << i << ") = " << arrViewMul(i) << ", Expected = " << (vecA[i] * vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiv(i) != (vecA[i] / (vecB[i] + 1))) {
            Log(LogType::Error, true) << "Div(" << i << ") = " << arrViewDiv(i) << ", Expected = " << (vecA[i] / (vecB[i] + 1)) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test projection for an array_view without a data source
// Projections created outside the p_f_e
bool Test5(const accelerator_view &av)
{
    const int size = 2047;

    std::vector<int> vecA(size);
    std::vector<int> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int> arrViewA(size, vecA);
    array_view<const int> arrViewB(size, vecB);

    // Let's first cache the inputs on "av"
    array<int> tempArr(size, av);
    parallel_for_each(av, tempArr.get_extent(), [=, &tempArr](const index<1> &idx) restrict(amp) {
        tempArr[idx] = arrViewA[idx] + arrViewB[idx];
    });

    array_view<int, 2> arrViewOut(4, size);

    parallel_for_each(extent<1>(size), [=](const index<1> &idx) restrict(amp)
    {
        array_view<int> arrViewSum = arrViewOut[0];
        array_view<int> arrViewDiff = arrViewOut[1];
        array_view<int> arrViewMul = arrViewOut[2];
        array_view<int> arrViewDiv = arrViewOut[3];

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
        arrViewMul[idx] = arrViewA[idx] * arrViewB[idx];
        arrViewDiv[idx] = arrViewA[idx] / (arrViewB[idx] + 1);
    });

    // Now verify the results
    array_view<int> arrViewSum = arrViewOut[0];
    array_view<int> arrViewDiff = arrViewOut[1];
    array_view<int> arrViewMul = arrViewOut[2];
    array_view<int> arrViewDiv = arrViewOut[3];
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i << ") = " << arrViewSum(i) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i << ") = " << arrViewDiff(i) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewMul(i) != (vecA[i] * vecB[i])) {
            Log(LogType::Error, true) << "Mul(" << i << ") = " << arrViewMul(i) << ", Expected = " << (vecA[i] * vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiv(i) != (vecA[i] / (vecB[i] + 1))) {
            Log(LogType::Error, true) << "Div(" << i << ") = " << arrViewDiv(i) << ", Expected = " << (vecA[i] / (vecB[i] + 1)) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test view_as for an array_view without a data source
// view_as performed outside the p_f_e
bool Test6()
{
    const int M = 256;
    const int N = 129;
    const int size = M * N;

    std::vector<int> vecA(size);
    std::vector<int> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);

    array_view<int> tempArrViewOut(2 * size);
    array_view<int, 2> arrViewOut = tempArrViewOut.view_as(extent<2>(2 * M, N));

    parallel_for_each(extent<2>(M, N), [=](const index<2> &idx) restrict(amp)
    {
        array_view<int, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
        array_view<int, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    array_view<int, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
    array_view<int, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test view_as for an array_view without a data source
// view_as performed inside the p_f_e
bool Test7()
{
    const int M = 256;
    const int N = 129;
    const int size = M * N;

    std::vector<int> vecA(size);
    std::vector<int> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);

    array_view<int> tempArrViewOut(2 * size);
    array_view<int> tempArrViewSum = tempArrViewOut.section(0, size);
    array_view<int> tempArrViewDiff = tempArrViewOut.section(size, size);

    parallel_for_each(extent<2>(M, N), [=](const index<2> &idx) restrict(amp)
    {
        array_view<int, 2> arrViewSum = tempArrViewSum.view_as(extent<2>(M, N));
        array_view<int, 2> arrViewDiff = tempArrViewDiff.view_as(extent<2>(M, N));

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    array_view<int, 2> arrViewSum = tempArrViewOut.view_as(extent<2>(2 * M, N)).section(0, 0, M, N);
    array_view<int, 2> arrViewDiff = tempArrViewOut.view_as(extent<2>(2 * M, N)).section(M, 0, M, N);
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Overloaded std::ostream::operator<< for graphics::int_4 type
std::ostream& operator<<(std::ostream &outStream, const graphics::int_4 &val)
{
    outStream << "(" << val.get_x() << ", " << val.get_y() << ", " << val.get_z() << ", " << val.get_w() << ")";
    return outStream;
}

// Test reinterpret_as for an array_view without a data source
// reinterpret_as performed outside the p_f_e
bool Test8()
{
    const int M = 256;
    const int N = 129;
    const int size = M * N;

    std::vector<graphics::int_4> vecA(size);
    std::vector<graphics::int_4> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const graphics::int_4, 2> arrViewA(M, N, vecA);
    array_view<const graphics::int_4, 2> arrViewB(M, N, vecB);

    array_view<int> tempArrViewOut(2 * size * (sizeof(graphics::int_4) / sizeof(int)));
    array_view<graphics::int_4, 2> arrViewOut = tempArrViewOut.reinterpret_as<graphics::int_4>().view_as(extent<2>(2 * M, N));

    parallel_for_each(extent<2>(M, N), [=](const index<2> &idx) restrict(amp)
    {
        array_view<graphics::int_4, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
        array_view<graphics::int_4, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    array_view<graphics::int_4, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
    array_view<graphics::int_4, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test reinterpret_as for an array_view without a data source
// reinterpret_as performed inside the p_f_e
runall_result Test9()
{
    // This test requires limited_double support. Let's find an accelerator with the required
    // support and skip the test if we don't find such an accelerator

    // TODO: We may consider separating this out into another test
    // so that the require_device function can be used to find an
    // accelerator with the required double support and skip the test
    // if one is not found. It may not be a big deal though since the
    // direct3d_ref accelerator would mostly be present in the test
    // environment

    bool foundLimitedDoubleSupportAccl = false;
    accelerator_view av = accelerator().get_default_view();
    std::vector<accelerator> allAccls = accelerator::get_all();
    for (size_t i = 0; i < allAccls.size(); ++i)
    {
        if (allAccls[i].get_device_path() == accelerator::cpu_accelerator) {
            continue;
        }

        if (allAccls[i].get_supports_limited_double_precision()) {
            foundLimitedDoubleSupportAccl = true;
            av = allAccls[i].get_default_view();
            break;
        }
    }

    if (!foundLimitedDoubleSupportAccl) {
        return runall_skip;
    }

    const int M = 256;
    const int N = 129;
    const int size = M * N;

    std::vector<double> vecA(size);
    std::vector<double> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const double, 2> arrViewA(M, N, vecA);
    array_view<const double, 2> arrViewB(M, N, vecB);

    array_view<int> tempArrViewOut(2 * size * (sizeof(double) / sizeof(int)));

    parallel_for_each(av, extent<2>(M, N), [=](const index<2> &idx) restrict(amp)
    {
        array_view<double, 2> arrViewOut = tempArrViewOut.reinterpret_as<double>().view_as(extent<2>(2 * M, N));
        array_view<double, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
        array_view<double, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    array_view<double, 2> arrViewOut = tempArrViewOut.reinterpret_as<double>().view_as(extent<2>(2 * M, N));
    array_view<double, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
    array_view<double, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);
    runall_result passed = runall_pass;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = runall_fail;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = runall_fail;
        }
    }

    return passed;
}

// Test synchronize for an array_view without a data source
bool Test10()
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);
    array_view<int, 2> arrViewSum(M, N);
    array_view<int, 2> arrViewDiff(M, N);
    parallel_for_each(arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    arrViewSum.synchronize();
    arrViewDiff.synchronize();

    // Now verify the results
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i) {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test synchronize_async for an array_view without a data source
bool Test11()
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);
    array_view<int, 2> arrViewSum(M, N);
    array_view<int, 2> arrViewDiff(M, N);
    parallel_for_each(arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    auto fut1 = arrViewSum.synchronize_async();
    auto fut2 = arrViewDiff.synchronize_async();

    return (fut1.to_task() && fut2.to_task()).then([&]() {
        // Now verify the results
        bool passed = true;
        for (size_t i = 0; i < vecA.size(); ++i) {
            if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
                Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
                passed = false;
            }

            if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
                Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
                passed = false;
            }
        }

        return passed;
    }).get();
}

// Test refresh for an array_view without a data source
bool Test12()
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);
    array_view<int, 2> arrViewSum(M, N);
    array_view<int, 2> arrViewDiff(M, N);
    arrViewSum.refresh();
    arrViewDiff.refresh();

    parallel_for_each(arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i) {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test discard_data for an array_view without a data source
bool Test13()
{
    const int M = 256;
    const int N = 256;

    std::vector<int> vecA(M * N);
    std::vector<int> vecB(M * N);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);
    array_view<int, 2> arrViewSum(M, N);
    arrViewSum.discard_data();

    array_view<int, 2> arrViewDiff(M, N);
    parallel_for_each(arrViewDiff.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewDiff[idx] = 0;
    });

    arrViewDiff.discard_data();

    parallel_for_each(arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i) {
        if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Test data method for an array_view without a data source
// "data" used both inside and outside p_f_e
bool Test14()
{
    const int M = 256;
    const int N = 129;
    const int size = M * N;

    std::vector<int> vecA(size);
    std::vector<int> vecB(size);
    std::generate(vecA.begin(), vecA.end(), rand);
    std::generate(vecB.begin(), vecB.end(), rand);

    array_view<const int, 2> arrViewA(M, N, vecA);
    array_view<const int, 2> arrViewB(M, N, vecB);

    array_view<int> tempArrViewOut(2 * size);

    parallel_for_each(extent<2>(M, N), [=](const index<2> &idx) restrict(amp)
    {
        array_view<int, 2> arrViewOut(2 * M, N, tempArrViewOut.data());
        array_view<int, 2> arrViewSum = arrViewOut.section(0, 0, M, N);
        array_view<int, 2> arrViewDiff = arrViewOut.section(M, 0, M, N);

        arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
        arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
    });

    // Now verify the results
    const int *pArrViewSum = tempArrViewOut.data();
    const int *pArrViewDiff = pArrViewSum + (M * N);
    bool passed = true;
    for (size_t i = 0; i < vecA.size(); ++i)
    {
        if (pArrViewSum[i] != (vecA[i] + vecB[i])) {
            Log(LogType::Error, true) << "Sum(" << i << ") = " << pArrViewSum[i] << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
            passed = false;
        }

        if (pArrViewDiff[i] != (vecA[i] - vecB[i])) {
            Log(LogType::Error, true) << "Diff(" << i << ") = " << pArrViewDiff[i] << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

// Tests that when using an array_view without a data source
// the p_f_e target selected is always the one where the input
// data is pre-cached
// TODO: This makes no sense on HCC at the moment, hence why it is disabled,
//       pending refactoring.
//bool Test15()
//{
//    const int size = (1023 * 5);
//    accelerator_view av = accelerator().create_view();
//    ID3D11Device *pDevice = get_d3d11_device(av);
//
//    array<int> arrA(size, av), arrB(size, av);
//    ID3D11Buffer *pBufferA = get_d3d11_buffer(arrA);
//    ID3D11Buffer *pBufferB = get_d3d11_buffer(arrB);
//
//    parallel_for_each(arrA.get_extent(), [&](const index<1> &idx) restrict(amp) {
//        arrA[idx] = idx[0];
//    });
//
//    array_view<const int> arrViewA(arrA);
//    array_view<int> arrViewB(arrB);
//    arrViewB.discard_data();
//    array_view<int> arrViewC(size);
//    parallel_for_each(extent<1>(size), [=](const index<1> &idx) restrict(amp) {
//        arrViewB[idx] = arrViewA[idx] + idx[0];
//        arrViewC[idx] = arrViewA[idx] + idx[0];
//    });
//
//    bool passed = true;
//
//    // Now lets copy the contents of the ID3D11Buffer underlying the array source "arrB"
//    // without synchonizing to ensure that the p_f_e was indeed launched on "av"
//    std::vector<int> vec(arrB.get_extent().size(), 0);
//
//    if (CopyOut(pDevice, pBufferB, vec.data()) != S_OK) {
//        Log(LogType::Info, true) << "Failed to copy from D3D buffer to host!" << std::endl;
//        passed = false;
//    }
//    else {
//        // Verify the contents of pBufferB
//        for (size_t i = 0; i < vec.size(); ++i) {
//            if (vec[i] != (2 * i)) {
//                Log(LogType::Info, true) << "pBufferB[" << i << "] = " << vec[i] << ", Expected = " << (2 * i) << std::endl;
//                passed = false;
//            }
//        }
//    }
//
//    pDevice->Release();
//    pBufferA->Release();
//    pBufferB->Release();
//
//    return passed;
//}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    runall_result res;

#ifdef test_set1
	res &= REPORT_RESULT(Test1(av));
	res &= REPORT_RESULT(Test2());
	res &= REPORT_RESULT(Test3(av));
	res &= REPORT_RESULT(Test4());
#endif

#ifdef test_set2
	res &= REPORT_RESULT(Test5(av));
	res &= REPORT_RESULT(Test6());
	res &= REPORT_RESULT(Test7());
	res &= REPORT_RESULT(Test8());
#endif

#ifdef test_set3
	res &= REPORT_RESULT(Test9());
#endif

#ifdef test_set4
	res &= REPORT_RESULT(Test10());
	res &= REPORT_RESULT(Test11());
	res &= REPORT_RESULT(Test12());
	res &= REPORT_RESULT(Test13());
#endif

#ifdef test_set5
	res &= REPORT_RESULT(Test14());
	res &= REPORT_RESULT(Test15());
#endif

    return res;
}

