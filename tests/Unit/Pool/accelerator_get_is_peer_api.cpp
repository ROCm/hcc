
// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out

#include <hc.hpp>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

/**
 * Test if hc::accelerator::get_is_peer() works fine.
 * Create the default accelerator and check if others 
 * is peer of it.
 * accelerator is peer of itself.
 * FIXME: on current system, dGPU is peer of any each  
 * other, we should expect get_is_peer() return true.
 */
 

int main()
{
    hc::accelerator acc;

    const auto& all = hc::accelerator::get_all();

    if(0 == all.size())
        return -1;

    for(auto iter = all.begin(); iter != all.end(); iter++)
    {
        // Ignore CPU
        if(iter->get_is_emulated())
            continue;

        hsa_agent_t* s = static_cast<hsa_agent_t*>(iter->get_hsa_agent());
        hsa_amd_memory_pool_t* pool = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());

        hsa_amd_memory_pool_access_t access;
        hsa_status_t status = hsa_amd_agent_memory_pool_get_info(*s, *pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
        
        bool can_access = (access == HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT) || (access == HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT);
        if(can_access && !acc.get_is_peer(*iter))
            return -1;
    }

    return 0;
}
