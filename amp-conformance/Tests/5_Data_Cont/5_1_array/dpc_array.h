// This file contains coomon funcs, #define and consts used accross array data container

#include <amptest.h>
#include <vector>

#include <algorithm>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

accelerator _cpu_device(accelerator::cpu_accelerator);
accelerator _gpu_device(accelerator::default_accelerator); // This always doesnt gurantee hardware gpu - use if "is_gpu_hardware_available" is true;

bool is_gpu_hardware_available()
{
	return get_device(_gpu_device, device_flags::D3D11_GPU);
}

