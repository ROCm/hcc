
// RUN: %hc %s -lhc_am -o %t.out; %t.out

#include <hc.hpp>
#include <hc_am.hpp>

int main()
{
    hc::accelerator acc;
    const auto& all = hc::accelerator::get_all();
    if(all.size() == 0){
        return -1;
    }
    std::vector<hc::accelerator> accVec;
    int hsaAgentCount = 0;
    for(int i=0;i<all.size();i++)
    {
        if(all[i].get_hsa_agent() != NULL){
            accVec.push_back(all[i]);
            hsaAgentCount++;
	}
    }
    float *ptr = new float[1024*1024];
    acc = all[0];
    int ret = 0;
    if(AM_SUCCESS != hc::am_memory_host_lock(acc, (void*)ptr, sizeof(float)*1024*1024, &accVec[0], hsaAgentCount))
    {
        ret = -1;
    }
    delete[] ptr;
    return ret;
}
