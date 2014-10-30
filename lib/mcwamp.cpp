//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <amp.h>

namespace Concurrency {

// initialize static class members
const wchar_t accelerator::gpu_accelerator[] = L"gpu";
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
const wchar_t accelerator::default_accelerator[] = L"default";

std::shared_ptr<accelerator> accelerator::_gpu_accelerator = std::make_shared<accelerator>(accelerator::gpu_accelerator);
std::shared_ptr<accelerator> accelerator::_cpu_accelerator = std::make_shared<accelerator>(accelerator::cpu_accelerator);
std::shared_ptr<accelerator> accelerator::_default_accelerator = nullptr;

} // namespace Concurrency

