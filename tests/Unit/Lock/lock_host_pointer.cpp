// XFAIL: Linux
// RUN: %hc %s -o %t.out %t.out

#include <hc.hpp>

int main()
{
    hc::accelerator acc;
    const auto& all = hc::accelerator::get_all();
    if(all.size() == 0){
        return -1;
    }
    std::vector<hsa_agent_t> agentVec;
    for(int i=0;i<all.size();i++)
    {
        agentVec.push_back(*static_cast<hsa_agent_t*>all[i].get_hsa_agent());
    }
    float *ptr = new float[1024*1024];
    if(AM_SUCCESS != hc::am_memory_host_lock(all[0], (void*)ptr, sizeof(float)*1024*1024, agentVec))
    {
        return 0;
    }
    return -1;
}
