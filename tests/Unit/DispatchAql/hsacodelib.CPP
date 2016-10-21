#include <vector>
#include <fstream>
#include <assert.h>

#include <hc.hpp>
#include <grid_launch.h>

#include <hsa/hsa.h>

#include "hsacodelib.h"




// Load HSACO 
Kernel load_hsaco(hc::accelerator_view *av, const char * fileName, const char *kernelName)
{
    hsa_region_t systemRegion = *(hsa_region_t*)av->get_hsa_am_system_region();
    hsa_agent_t hsaAgent = *(hsa_agent_t*) av->get_hsa_agent();

    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("could not open code object '%s'\n", fileName);
        assert(0);
    };

    std::streamsize fsize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fsize);
    if (file.read(buffer.data(), fsize))
    {

        hsa_status_t status;

        hsa_code_object_t code_object = {0};
        status = hsa_code_object_deserialize(&buffer[0], fsize, NULL, &code_object);
        assert(status == HSA_STATUS_SUCCESS);

        hsa_executable_t hsaExecutable;
        status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                       NULL, &hsaExecutable);
        assert(status == HSA_STATUS_SUCCESS);

        // Load the code object.
        status = hsa_executable_load_code_object(hsaExecutable, hsaAgent, code_object, NULL);
        assert(status == HSA_STATUS_SUCCESS);

        // Freeze the executable.
        status = hsa_executable_freeze(hsaExecutable, NULL);
        assert(status == HSA_STATUS_SUCCESS);


        // Get symbol handle.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol(hsaExecutable, NULL, kernelName, hsaAgent, 0, &kernelSymbol);
        assert(status == HSA_STATUS_SUCCESS);

        Kernel k;


        status = hsa_executable_symbol_get_info(kernelSymbol,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   &k._kernelCodeHandle);
        assert(status == HSA_STATUS_SUCCESS);

        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &k._privateSegmentSize);
        assert(status == HSA_STATUS_SUCCESS);

        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &k._groupSegmentSize);
        assert(status == HSA_STATUS_SUCCESS);


        return k;


    } else {
        printf("could not open code object '%s'\n", fileName);
        assert(0);
    }
}


// Dispatch a GL packet - 
// Convert to AQL and call dispatch_hsa_kernel
void dispatch_glp_kernel(const grid_launch_parm *lp, const Kernel &k, void *args, int argSize)
{
    hsa_kernel_dispatch_packet_t aql;
    memset(&aql, 0, sizeof(aql));

    aql.completion_signal.handle = 0;

    aql.grid_size_x = lp->grid_dim.x;
    aql.grid_size_y = lp->grid_dim.y;
    aql.grid_size_z = lp->grid_dim.z;
    aql.workgroup_size_x = lp->grid_dim.x;
    aql.workgroup_size_y = lp->grid_dim.y;
    aql.workgroup_size_z = lp->grid_dim.z;

    aql.group_segment_size = lp->dynamic_group_mem_bytes; // TODO - should read static from code object.
    aql.private_segment_size = 0; // TODO - should read from code-object
    aql.kernarg_address = 0;
    aql.kernel_object = k._kernelCodeHandle;

    aql.header =   (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                        (1 << HSA_PACKET_HEADER_BARRIER) |
                        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

    aql.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    aql.private_segment_size = k._privateSegmentSize;
    aql.group_segment_size = k._groupSegmentSize;

    lp->av->dispatch_hsa_kernel(&aql, args, argSize, lp->cf);
}
