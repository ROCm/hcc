#pragma once

#include <hc.hpp>
#include <initializer_list>

typedef int am_status_t;
#define AM_SUCCESS                           0
// TODO - provide better mapping of HSA error conditions to HC error codes.
#define AM_ERROR_MISC                       -1 /** Misellaneous error */

// Flags for am_alloc API:
#define amHostPinned 0x1


namespace hc {

// Info for each pointer in the memtry tracker:
struct AmPointerInfo {
    void *      _hostPointer;   ///< Host pointer.  If host access is not allowed, NULL.
    void *      _devicePointer; ///< Device pointer.  
    size_t      _sizeBytes;     ///< Size of allocation.
    hc::accelerator _acc;       ///< Device / Accelerator to use.
    bool        _isInDeviceMem;    ///< Memory is physically resident on a device (if false, memory is located on host)
    bool        _isAmManaged;   ///< Memory was allocated by AM and should be freed when am_reset is called.

    int         _appId;              ///< App-specific storage.  (Used by HIP to store deviceID.)
    unsigned    _appAllocationFlags; ///< App-specific allocation flags.  (Used by HIP to store allocation flags.)


    AmPointerInfo(void *hostPointer, void *devicePointer, size_t sizeBytes, hc::accelerator &acc, bool isInDeviceMem, bool isAmManaged) :
        _hostPointer(hostPointer),
        _devicePointer(devicePointer),
        _sizeBytes(sizeBytes),
        _acc(acc),
        _isInDeviceMem(isInDeviceMem),
        _isAmManaged(isAmManaged),
        _appId(-1),
        _appAllocationFlags(0)  {};

    AmPointerInfo & operator= (const AmPointerInfo &other);

};
}



namespace hc {


/**
 * Allocate a block of @p size bytes of memory on the specified @p acc.
 *
 * The contents of the newly allocated block of memory are not initialized.
 *
 * If @p size == 0, 0 is returned.
 *
 * Flags must be 0.
 *
 * @return : On success, pointer to the newly allocated memory is returned.
 * The pointer is typecast to the desired return type.
 *
 * If an error occurred trying to allocate the requested memory, 0 is returned.
 *
 * @see am_free, am_copy
 */
auto_voidp am_alloc(size_t size, hc::accelerator &acc, unsigned flags);

/**
 * Free a block of memory previously allocated with am_alloc.
 *
 * @return AM_SUCCESS
 * @see am_alloc, am_copy
 */
am_status_t am_free(void*  ptr);


/**
 * Copy @p size bytes of memory from @p src to @ dst.  The memory areas (src+size and dst+size) must not overlap.
 *
 * @return AM_SUCCESS on error or AM_ERROR_MISC if an error occurs.
 * @see am_alloc, am_free
 */
am_status_t am_copy(void*  dst, const void*  src, size_t size);



/**
 * Return information about tracked pointer.
 *
 * AM tracks pointers when they are allocated or added to tracker with am_track_pointer.
 * The tracker tracks the base pointer as well as the size of the allocation, and will
 * find the information for a pointer anywhere in the tracked range.
 *
 * @returns AM_ERROR_MISC if pointer is not currently being tracked.
 * @returns AM_SUCCESS if pointer is tracked and writes info to @p info.
 *
 * @see AM_memtracker_add, 
 */
am_status_t am_memtracker_getinfo(hc::AmPointerInfo *info, const void *ptr);


/**
 * Add a pointer to the memory tracker.
 *
 * @return AM_SUCCESS
 * @see am_memtracker_getinfo
 */
am_status_t am_memtracker_add(void* ptr, size_t sizeBytes, hc::accelerator &acc, bool isDeviceMem=false);


/*
 * Update info for an existing pointer in the memory tracker.
 *
 * @returns AM_ERROR_MISC if pointer is not found in tracker.  
 * @returns AM_SUCCESS if pointer is not found in tracker.  
 *
 * @see am_memtracker_getinfo, am_memtracker_add
 */
am_status_t am_memtracker_update(const void* ptr, int appId, unsigned allocationFlags);


/** 
 * Remove @ptr from the tracker structure.
 *
 * @p ptr may be anywhere in a tracked memory range.
 *
 * @returns AM_ERROR_MISC if pointer is not found in tracker.  
 * @returns AM_SUCCESS if pointer is not found in tracker.  
 *
 * @see am_memtracker_getinfo, am_memtracker_add
 */
am_status_t am_memtracker_remove(void* ptr);

/**
 * Remove all memory allocations associated with specified accelerator from the memory tracker.
 *
 * @returns Number of entries reset.
 * @see am_memtracker_getinfo
 */
size_t am_memtracker_reset(const hc::accelerator &acc);

/**
 * Print the entries in the memory tracker table.
 *
 * Intended primarily for debug purposes.
 * @see am_memtracker_getinfo
 **/
void am_memtracker_print();


/**
 * Return total sizes of device, host, and user memory allocated by the application
 *
 * User memory is registered with am_tracker_add.
 **/
void am_memtracker_sizeinfo(const hc::accelerator &acc, size_t *deviceMemSize, size_t *hostMemSize, size_t *userMemSize);


/*
 * Map device memory pointed to by @p ptr to the device's peers.
 * 
 * @p ptr pointer which points to device memory
 * @p begin iterator of the first accelerator
 * @p end iterator of the last accelerator
 * @return AM_SUCCESS if mapped successfully.
 * @return AM_ERROR_MISC if @p ptr is nullptr or @p list is empty.
 * @return AM_ERROR_MISC if @p ptr is not am managed.
 * @return AM_ERROR_MISC if @p is not found in the pointer tracker.
 * @return AM_ERROR_MISC if @p list incudes a bad peer.
 */
template<class Iterator>
am_status_t am_map_to_peers(void* ptr, Iterator begin, Iterator end) 
{
    // check input
    if(nullptr == ptr || begin == end)
        return AM_ERROR_MISC;

    hc::accelerator acc;
    AmPointerInfo info(nullptr, nullptr, 0, acc, false, false);
    auto status = am_memtracker_getinfo(&info, ptr);
    if(AM_SUCCESS != status)
        return status;

        hsa_amd_memory_pool_t* pool = nullptr;
    if(info._isInDeviceMem)
    {
        // get accelerator and pool of device memory
        acc = info._acc;
        pool = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
    }
    else
    {
        //TODO: the ptr is host pointer, it might be allocated through am_alloc, 
        // or allocated by others, but add it to the tracker.
        // right now, only support host pointer which is allocated through am_alloc.
        if(info._isAmManaged)
        {
            // here, accelerator is the device, but used to query system memory pool
            acc = info._acc;
            pool = static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_system_region()); 
        }
        else
            return AM_ERROR_MISC;
    }

    const size_t max_agent = hc::accelerator::get_all().size();
    hsa_agent_t agents[max_agent];
  
    int peer_count = 0;

    for(auto i = begin; i != end; i++)
    {
        // if pointer is device pointer, the accelerator itself is included in the list, ignore it
        auto& a = *i;
        if(info._isInDeviceMem)
        {
            if(a == acc)
                continue;
        }

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

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

    // allow access to the agents
    if(peer_count)
    {
        hsa_status_t status = hsa_amd_agents_allow_access(peer_count, agents, NULL, ptr);
        return status == HSA_STATUS_SUCCESS ? AM_SUCCESS : AM_ERROR_MISC;
    }
   
    return AM_SUCCESS;
}


}; // namespace hc

