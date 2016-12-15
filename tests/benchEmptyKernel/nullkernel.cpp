// RUN: %hc --amdgpu-target=AMD:AMDGPU:7:0:1 --amdgpu-target=AMD:AMDGPU:8:0:1 --amdgpu-target=AMD:AMDGPU:8:0:3 -fPIC -Wl,-Bsymbolic -shared %S/nullkernel.cpp -o %T/nullkernel
// RUN: extractkernel -i %T/nullkernel

#include "hc.hpp"
#include "grid_launch.hpp"

__attribute__((hc_grid_launch))
void nullkernel(const grid_launch_parm lp, float* Ad) {
    if (Ad) {
        Ad[0] = 42;
    }
}
