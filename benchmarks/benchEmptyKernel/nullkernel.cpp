// RUN: %hc --amdgpu-target=gfx801 --amdgpu-target=gfx802 --amdgpu-target=gfx803 -fPIC -Wl,-Bsymbolic -shared %S/nullkernel.cpp -o %T/nullkernel
// RUN: HCC_HOME=%llvm_libs_dir/../../ %extractkernel -i %T/nullkernel

#include "hc.hpp"
#include "grid_launch.hpp"

__attribute__((hc_grid_launch))
void nullkernel(const grid_launch_parm lp, float* Ad) {
    if (Ad) {
        Ad[0] = 42;
    }
}
