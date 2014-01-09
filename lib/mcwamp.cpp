#include <amp.h>
namespace Concurrency {
accelerator_view *accelerator::default_view_ = NULL;
const wchar_t accelerator::gpu_accelerator[] = L"gpu";
wchar_t accelerator::default_accelerator[] = L"default";
const wchar_t accelerator::cpu_accelerator[] = L"cpu";
}

