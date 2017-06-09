#include "hc_am.hpp"

#include <cstdint>
#include <iomanip>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#define DB_TRACKER 0

#if DB_TRACKER 
#define mprintf( ...) {\
        fprintf (stderr, __VA_ARGS__);\
        };
#else
#define mprintf( ...) 
#endif

//=========================================================================================================
// Pointer Tracker Structures:
//=========================================================================================================
#include <map>
#include <iostream>

namespace hc {
AmPointerInfo & AmPointerInfo::operator= (const AmPointerInfo &other) 
{
    _hostPointer = other._hostPointer;
    _devicePointer = other._devicePointer;
    _sizeBytes = other._sizeBytes;
    _acc = other._acc;
    _isInDeviceMem = other._isInDeviceMem;
    _isAmManaged = other._isAmManaged;
    _appId = other._appId;
    _appAllocationFlags = other._appAllocationFlags;

    return *this;
}
}


//---
struct AmMemoryRange {
    const void * _basePointer;
    const void * _endPointer;
    AmMemoryRange(const void *basePointer, size_t sizeBytes) :
        _basePointer(basePointer), _endPointer((const unsigned char*)basePointer + sizeBytes - 1) {};
};

// Functor to compare ranges:
struct AmMemoryRangeCompare {
    // Return true is LHS range is less than RHS - used to order the 
    bool operator()(const AmMemoryRange &lhs, const AmMemoryRange &rhs) const
    {
        return lhs._endPointer < rhs._basePointer;
    }

};



// width to use when printing pointers:
const int PTRW=14;

void printShortPointerInfo(std::ostream &os, const hc::AmPointerInfo &ap)
{
    using namespace std;
    os << "#" << setw(6)  << ap._allocSeqNum
       << " " << setw(PTRW) << ap._hostPointer 
       << " " << setw(PTRW) << ap._devicePointer
       << " " << setw(12) <<  ap._sizeBytes
       << " " << setw(8) << fixed << setprecision(2) << (double)ap._sizeBytes/1024.0/1024.0
       << (ap._isInDeviceMem ? " DEV " : " HOST")
       << (ap._isAmManaged ? " ALLOC" : " REGIS")
       << " " << setw(5) << ap._appId 
       << " " << hex << setw(8) << ap._appAllocationFlags << dec
       ;
}


void printRocrPointerInfo(std::ostream &os, const void *ptr)
{
    hsa_amd_pointer_info_t info;
    hsa_status_t hsa_status;
    bool isLocked = false;
    info.size = sizeof(info);

    uint32_t peerAgentCnt=0;
    hsa_agent_t * peerAgents = nullptr;
    hsa_status = hsa_amd_pointer_info(const_cast<void*> (ptr), &info, malloc, &peerAgentCnt, &peerAgents);

    if(hsa_status == HSA_STATUS_SUCCESS) {

        for (uint32_t i=0; i<peerAgentCnt; i++) {
            os << " 0x" << std::hex << peerAgents[i].handle ;
               //<< "(" << hc::accelerator::get_seqnum_from_agent(peerAgents[i]) << ")" << std::dec;
        }

        if (peerAgents) {
            free (peerAgents);
        }
    }
    os << std::dec;
}


std::ostream &operator<<(std::ostream &os, const hc::AmPointerInfo &ap)
{
    os << "allocSeqNum:" << ap._allocSeqNum
       << " hostPointer:" << ap._hostPointer << " devicePointer:"<< ap._devicePointer << " sizeBytes:" << ap._sizeBytes
       << " isInDeviceMem:" << ap._isInDeviceMem  << " isAmManaged:" << ap._isAmManaged 
       << " appId:" << ap._appId << " appAllocFlags:" << ap._appAllocationFlags
       << std::left << " peers:" << std::right
       ;

    printRocrPointerInfo(os, ap._isInDeviceMem ? ap._devicePointer : ap._hostPointer);
    return os;
}

//-------------------------------------------------------------------------------------------------
// This structure tracks information for each pointer.
// Uses memory-range-based lookups - so pointers that exist anywhere in the range of hostPtr + size 
// will find the associated AmPointerInfo.
// The insertions and lookups use a self-balancing binary tree and should support O(logN) lookup speed.
// The structure is thread-safe - writers obtain a mutex before modifying the tree.  Multiple simulatenous readers are supported.
class AmPointerTracker {
typedef std::map<AmMemoryRange, hc::AmPointerInfo, AmMemoryRangeCompare> MapTrackerType;
public:

    void insert(void *pointer, hc::AmPointerInfo &p);
    int remove(void *pointer);

    MapTrackerType::iterator find(const void *hostPtr) ;
    
    MapTrackerType::iterator readerLockBegin() { _mutex.lock(); return _tracker.begin(); } ;
    MapTrackerType::iterator end() { return _tracker.end(); } ;
    void readerUnlock() { _mutex.unlock(); };


    size_t reset (const hc::accelerator &acc);
    void update_peers (const hc::accelerator &acc, int peerCnt, hsa_agent_t *peerAgents) ;

private:
    MapTrackerType  _tracker;
    std::mutex      _mutex;
    //std::shared_timed_mutex _mut;
    uint64_t        _allocSeqNum = 0;
};


//---
void AmPointerTracker::insert (void *pointer, hc::AmPointerInfo &p)
{
    std::lock_guard<std::mutex> l (_mutex);

    p._allocSeqNum = ++ this->_allocSeqNum;

    mprintf ("insert: %p + %zu\n", pointer, p._sizeBytes);
    _tracker.insert(std::make_pair(AmMemoryRange(pointer, p._sizeBytes), p));
}


//---
// Return 1 if removed or 0 if not found.
int AmPointerTracker::remove (void *pointer)
{
    std::lock_guard<std::mutex> l (_mutex);
    mprintf ("remove: %p\n", pointer);
    return _tracker.erase(AmMemoryRange(pointer,1));
}


//---
AmPointerTracker::MapTrackerType::iterator  AmPointerTracker::find (const void *pointer)
{
    std::lock_guard<std::mutex> l (_mutex);
    auto iter = _tracker.find(AmMemoryRange(pointer,1));
    mprintf ("find: %p\n", pointer);
    return iter;
}


//---
// Remove all tracked locations, and free the associated memory (if the range was originally allocated by AM).
// Returns count of ranges removed.
size_t AmPointerTracker::reset (const hc::accelerator &acc) 
{
    std::lock_guard<std::mutex> l (_mutex);
    mprintf ("reset: \n");

    size_t count = 0;
    // relies on C++11 (erase returns iterator)
    for (auto iter = _tracker.begin() ; iter != _tracker.end(); ) {
        if (iter->second._acc == acc) {
            if (iter->second._isAmManaged) {
                hsa_amd_memory_pool_free(const_cast<void*> (iter->first._basePointer));
            }
            count++;

            iter = _tracker.erase(iter);
        } else {
            iter++;
        }
    }

    return count;
}


//---
// Remove all tracked locations, and free the associated memory (if the range was originally allocated by AM).
// Returns count of ranges removed.
void AmPointerTracker::update_peers (const hc::accelerator &acc, int peerCnt, hsa_agent_t *peerAgents) 
{
    std::lock_guard<std::mutex> l (_mutex);

    // relies on C++11 (erase returns iterator)
    for (auto iter = _tracker.begin() ; iter != _tracker.end(); ) {
        if (iter->second._acc == acc) {
            hsa_amd_agents_allow_access(peerCnt, peerAgents, NULL, const_cast<void*> (iter->first._basePointer));
        } 
        iter++;
    }
}


//=========================================================================================================
// Global var defs:
//=========================================================================================================
AmPointerTracker g_amPointerTracker;  // Track all am pointer allocations.


//=========================================================================================================
// API Definitions.
//=========================================================================================================
//
//

namespace hc {

// Allocate accelerator memory, return NULL if memory could not be allocated:
auto_voidp am_alloc(size_t sizeBytes, hc::accelerator &acc, unsigned flags) 
{

    void *ptr = NULL;

    if (sizeBytes != 0 ) {
        if (acc.is_hsa_accelerator()) {
            hsa_agent_t *hsa_agent = static_cast<hsa_agent_t*> (acc.get_default_view().get_hsa_agent());
            hsa_amd_memory_pool_t *alloc_region;
            if (flags & amHostPinned) {
               alloc_region = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_system_region());
            } else if (flags & amHostCoherent) {
               alloc_region = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_finegrained_system_region());
            }else {
               alloc_region = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
            }

            if (alloc_region && alloc_region->handle != -1) {

                hsa_status_t s1 = hsa_amd_memory_pool_allocate(*alloc_region, sizeBytes, 0, &ptr);

                if (s1 != HSA_STATUS_SUCCESS) {
                    ptr = NULL;
                } else {
                    if (flags & (amHostPinned|amHostCoherent)) {
                        if (s1 != HSA_STATUS_SUCCESS) {
                            hsa_amd_memory_pool_free(ptr);
                            ptr = NULL;
                        } else {
                            hc::AmPointerInfo ampi(ptr/*hostPointer*/, ptr /*devicePointer*/, sizeBytes, acc, false/*isDevice*/, true /*isAMManaged*/);
                            g_amPointerTracker.insert(ptr,ampi);

                            // Host memory is always mapped to all possible peers:
                            auto accs = hc::accelerator::get_all();
                            auto s2 = am_map_to_peers(ptr, accs.size(), accs.data());
                            if (s2 != AM_SUCCESS) {
                                hsa_amd_memory_pool_free(ptr);
                                ptr = NULL;
                            }
                        }
                    } else {
                        hc::AmPointerInfo ampi(NULL/*hostPointer*/, ptr /*devicePointer*/, sizeBytes, acc, true/*isDevice*/, true /*isAMManaged*/);
                        g_amPointerTracker.insert(ptr,ampi);
                    }
                }
            }
        }
    }

    return ptr;
};


am_status_t am_free(void* ptr) 
{
    am_status_t status = AM_SUCCESS;

    if (ptr != NULL) {

        int numRemoved = g_amPointerTracker.remove(ptr) ;
        if (numRemoved == 0) {
            status = AM_ERROR_MISC;
        } else {
            // See also tracker::reset which can free memory.
            hsa_amd_memory_pool_free(ptr);
        }
    }
    return status;
}


am_status_t am_copy(void*  dst, const void*  src, size_t sizeBytes)
{
    am_status_t am_status = AM_ERROR_MISC;
    hsa_status_t err = hsa_memory_copy(dst, src, sizeBytes);

    if (err == HSA_STATUS_SUCCESS) {
        am_status = AM_SUCCESS;
    } else {
        am_status = AM_ERROR_MISC;
    }

    return am_status;
}


am_status_t am_memtracker_getinfo(hc::AmPointerInfo *info, const void *ptr)
{
    auto infoI = g_amPointerTracker.find(ptr);
    if (infoI != g_amPointerTracker.end()) {
        if (info) {
            *info = infoI->second;
        }
        return AM_SUCCESS;
    } else {
        return AM_ERROR_MISC;
    }
}


am_status_t am_memtracker_add(void* ptr, hc::AmPointerInfo &info)
{
    if ((ptr == NULL) || (info._sizeBytes == 0)) {
        return AM_ERROR_MISC;
    } else {
        g_amPointerTracker.insert(ptr, info);
        return AM_SUCCESS;
    };

}


am_status_t am_memtracker_update(const void* ptr, int appId, unsigned allocationFlags)
{
    auto iter = g_amPointerTracker.find(ptr);
    if (iter != g_amPointerTracker.end()) {
        iter->second._appId              = appId;
        iter->second._appAllocationFlags = allocationFlags;
        return AM_SUCCESS;
    } else {
        return AM_ERROR_MISC;
    }
}


am_status_t am_memtracker_remove(void* ptr)
{
    am_status_t status = AM_SUCCESS;

    int numRemoved = g_amPointerTracker.remove(ptr) ;
    if (numRemoved == 0) {
        status = AM_ERROR_MISC;
    }

    return status;
}

//---
void am_memtracker_print(void *targetAddress)
{
    const char *targetAddressP = static_cast<const char *> (targetAddress);
    std::ostream &os = std::cerr;

    uint64_t beforeD = std::numeric_limits<uint64_t>::max() ;
    uint64_t afterD =  std::numeric_limits<uint64_t>::max() ;
    auto closestBefore = g_amPointerTracker.end();
    auto closestAfter  = g_amPointerTracker.end();
    bool foundMatch = false;


    if (targetAddress) {
        for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
            const auto basePointer = static_cast<const char*> (iter->first._basePointer);
            const auto endPointer = static_cast<const char*> (iter->first._endPointer);
            if ((targetAddressP >= basePointer) && (targetAddressP < endPointer)) {
                ptrdiff_t offset = targetAddressP - basePointer;
                os << "db: memtracker found pointer:" << targetAddress << " offset:" << offset << " bytes inside this allocation:\n";
                os << "   " << iter->first._basePointer << "-" << iter->first._endPointer << "::  ";
                os << iter->second << std::endl;
                foundMatch = true;
                break;
            } else {
                if ((targetAddressP < basePointer) && (basePointer - targetAddressP < beforeD)) {
                    beforeD = (basePointer - targetAddressP);
                    closestBefore = iter;
                }
                if ((targetAddressP > endPointer) && (targetAddressP - endPointer < afterD)) {
                    afterD = (targetAddressP - endPointer);
                    closestAfter = iter;
                }
            };

        }

        if (!foundMatch) {
            os << "db: memtracker did not find pointer:" << targetAddress << ".  However, it is closest to the following allocations:\n";
            if (closestBefore != g_amPointerTracker.end()) {
                os << "db: closest before: " << beforeD << " bytes before base of: " << closestBefore->second << std::endl;
            }
            if (closestAfter != g_amPointerTracker.end()) {
                os << "db: closest after: " << afterD << " bytes after end of " << closestAfter->second << std::endl ;
            }
        }
    } else {
        using namespace std;
        os <<  setw(PTRW) << "base" << "-" << setw(PTRW) << "end" << ": ";
        os  << setw(6+1) << "#SeqNum"
            << setw(PTRW+1) << "HostPtr"
            << setw(PTRW+1) << "DevPtr"
            << setw(12+1) << "SizeBytes"
            << setw(8+1) << "SizeMB"
            << setw(5) << "Dev?"
            << setw(6) << "Reg?"
            << setw(6) << " AppId"
            << setw(7) << " AppFlags"
            << setw(12) << left << " Peers" << right
            << "\n";

        for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
            os << setw(PTRW) << iter->first._basePointer << "-" << setw(PTRW) << iter->first._endPointer << ": ";
            printShortPointerInfo(os, iter->second);
            printRocrPointerInfo(os, iter->first._basePointer);
            os << "\n";
        }
    }

    g_amPointerTracker.readerUnlock();
}


//---
void am_memtracker_sizeinfo(const hc::accelerator &acc, size_t *deviceMemSize, size_t *hostMemSize, size_t *userMemSize)
{
    *deviceMemSize = *hostMemSize = *userMemSize = 0;
    for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
        if (iter->second._acc == acc) {
            size_t sizeBytes = iter->second._sizeBytes;
            if (iter->second._isAmManaged) {
                if (iter->second._isInDeviceMem) {
                    *deviceMemSize += sizeBytes;
                } else {
                    *hostMemSize += sizeBytes;
                }
            } else {
                *userMemSize += sizeBytes;
            }
        }
    }

    g_amPointerTracker.readerUnlock();
}


//---
size_t am_memtracker_reset(const hc::accelerator &acc)
{
    return g_amPointerTracker.reset(acc);
}

void am_memtracker_update_peers (const hc::accelerator &acc, int peerCnt, hsa_agent_t *peerAgents) 
{
    return g_amPointerTracker.update_peers(acc, peerCnt, peerAgents);
}

am_status_t am_map_to_peers(void* ptr, size_t num_peer, const hc::accelerator* peers) 
{
    // check input
    if(nullptr == ptr || 0 == num_peer || nullptr == peers)
        return AM_ERROR_MISC;

    hc::accelerator ptrAcc;
    AmPointerInfo info(nullptr, nullptr, 0, ptrAcc, false, false);
    auto status = am_memtracker_getinfo(&info, ptr);
    if(AM_SUCCESS != status)
        return status;

        hsa_amd_memory_pool_t* pool = nullptr;
    if(info._isInDeviceMem)
    {
        // get accelerator and pool of device memory
        ptrAcc = info._acc;
        pool = static_cast<hsa_amd_memory_pool_t*>(ptrAcc.get_hsa_am_region());
    }
    else
    {
        //TODO: the ptr is host pointer, it might be allocated through am_alloc, 
        // or allocated by others, but add it to the tracker.
        // right now, only support host pointer which is allocated through am_alloc.
        if(info._isAmManaged)
        {
            // here, accelerator is the device, but used to query system memory pool
            ptrAcc = info._acc;
            pool = static_cast<hsa_amd_memory_pool_t*>(ptrAcc.get_hsa_am_system_region()); 
        }
        else
            return AM_ERROR_MISC;
    }

    const size_t max_agent = hc::accelerator::get_all().size();
    hsa_agent_t agents[max_agent];
  
    int peer_count = 0;

    for(auto i = 0; i < num_peer; i++)
    {
        // if pointer is device pointer, and the accelerator itself is included in the list, ignore it
        auto& a = peers[i];
        if(info._isInDeviceMem)
        {
            if(a == ptrAcc)
                continue;
        }

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(a.get_hsa_agent());

        if (agent) {

            hsa_amd_memory_pool_access_t access;
            hsa_status_t  status = hsa_amd_agent_memory_pool_get_info(*agent, *pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
            if(HSA_STATUS_SUCCESS != status)
                return AM_ERROR_MISC;

            // check access
            if(HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED == access)
                return AM_ERROR_MISC;

            bool add_agent = true;

            for(int ii = 0; ii < peer_count; ii++)
            {
                if(agent->handle == agents[ii].handle)
                    add_agent = false;
            } 

            if(add_agent)
            {
                 agents[peer_count] = *agent;
                 peer_count++;
            }    
        }
    }

    // allow access to the agents
    if(peer_count)
    {
        hsa_status_t status = hsa_amd_agents_allow_access(peer_count, agents, NULL, ptr);
        return status == HSA_STATUS_SUCCESS ? AM_SUCCESS : AM_ERROR_MISC;
    }
   
    return AM_SUCCESS;
}

am_status_t am_memory_host_lock(hc::accelerator &ac, void *hostPtr, size_t size, hc::accelerator *visible_ac, size_t num_visible_ac)
{
    am_status_t am_status = AM_ERROR_MISC;
    void *devPtr;
    std::vector<hsa_agent_t> agents;
    for(int i=0;i<num_visible_ac;i++)
    {
        agents.push_back(*static_cast<hsa_agent_t*>(visible_ac[i].get_hsa_agent()));
    }
    hsa_status_t hsa_status = hsa_amd_memory_lock(hostPtr, size, &agents[0], num_visible_ac, &devPtr);
    if(hsa_status == HSA_STATUS_SUCCESS)
    {
       hc::AmPointerInfo ampi(hostPtr, devPtr, size, ac, false, false);
       g_amPointerTracker.insert(hostPtr, ampi);
       am_status = AM_SUCCESS;
    }
    return am_status;
}

am_status_t am_memory_host_unlock(hc::accelerator &ac, void *hostPtr)
{
    am_status_t am_status = AM_ERROR_MISC;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, ac, 0, 0);
    am_status = am_memtracker_getinfo(&amPointerInfo, hostPtr);
    if(am_status == AM_SUCCESS)
    {
        hsa_status_t hsa_status = hsa_amd_memory_unlock(hostPtr);
        if (hsa_status == HSA_STATUS_SUCCESS) {
            am_status = am_memtracker_remove(hostPtr);
        } else {
            am_status = AM_ERROR_MISC;
        }
    }
    return am_status;
}

} // end namespace hc.
