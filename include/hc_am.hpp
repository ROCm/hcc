#pragma once

#include "hc.hpp"
#include <initializer_list>

typedef int am_status_t;
#define AM_SUCCESS                           0
// TODO - provide better mapping of HSA error conditions to HC error codes.
#define AM_ERROR_MISC                       -1 /** Misellaneous error */

// Flags for am_alloc API:
#define amHostPinned      0x1 ///< Allocate pinned host memory accessible from all GPUs.
#define amHostNonCoherent 0x1 ///< Allocate non-coherent pinned host memory accessible from all GPUs.
#define amHostCoherent    0x2 ///< Allocate coherent pinned host memory accessible from all GPUs.

namespace hc {

// Info for each pointer in the memtry tracker:
class AmPointerInfo {
public:
    void *      _hostPointer;   ///< Host pointer.  If host access is not allowed, NULL.
    void *      _devicePointer; ///< Device pointer.  
    size_t      _sizeBytes;     ///< Size of allocation.
    hc::accelerator _acc;       ///< Accelerator where allocation is physically located.
    bool        _isInDeviceMem; ///< Memory is physically resident on a device (if false, memory is located on host)
    bool        _isAmManaged;   ///< Memory was allocated by AM and should be freed when am_reset is called.
    uint64_t    _allocSeqNum;   ///< Sequence number of allocation.

    int         _appId;              ///< App-specific storage.  (Used by HIP to store deviceID)
    unsigned    _appAllocationFlags; ///< App-specific allocation flags.  (Used by HIP to store allocation flags)
    void *      _appPtr;             ///< App-specific pointer to additional information.


    AmPointerInfo(void *hostPointer, void *devicePointer, size_t sizeBytes, hc::accelerator &acc,  bool isInDeviceMem=false, bool isAmManaged=false) :
        _hostPointer(hostPointer),
        _devicePointer(devicePointer),
        _sizeBytes(sizeBytes),
        _acc(acc),
        _isInDeviceMem(isInDeviceMem),
        _isAmManaged(isAmManaged),
        _allocSeqNum(0),
        _appId(-1),
        _appAllocationFlags(0),
        _appPtr(nullptr)  {};

    AmPointerInfo & operator= (const AmPointerInfo &other);

};
}



struct hsa_agent_s;

namespace hc {


/**
 * Allocate a block of @p size bytes of memory on the specified @p acc.
 *
 * The contents of the newly allocated block of memory are not initialized.
 *
 * If @p size == 0, 0 is returned.
 *
 * Flags:
 *  amHostPinned : Allocated pinned host memory and map it into the address space of the specified accelerator.
 *
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
am_status_t am_copy(void*  dst, const void*  src, size_t size) __attribute__ (( deprecated ("use accelerator_view::copy instead (and note src/dst order reversal)" ))) ;



/**
 * Return information about tracked pointer.
 *
 * AM tracks pointers when they are allocated or added to tracker with am_track_pointer.
 * The tracker tracks the base pointer as well as the size of the allocation, and will
 * find the information for a pointer anywhere in the tracked range.
 *
 * @returns AM_ERROR_MISC if pointer is not currently being tracked.  In this case, @p info
 * is not modified.

 * @returns AM_SUCCESS if pointer is tracked and writes info to @p info. if @ info is NULL,
 * no info is written but the returned status indicates if the pointer was tracked.
 *
 * @see AM_memtracker_add 
 */
am_status_t am_memtracker_getinfo(hc::AmPointerInfo *info, const void *ptr);


/**
 * Add a pointer to the memory tracker.
 *
 * @return AM_ERROR_MISC : If @p ptr is NULL, or info._sizeBytes = 0, the info is not added to the tracker and AM_ERROR_MISC is returned.
 * @return AM_SUCCESS
 * @see am_memtracker_getinfo
 */
am_status_t am_memtracker_add(void* ptr, hc::AmPointerInfo &info);


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
void am_memtracker_print(void * targetAddress=nullptr);


/**
 * Return total sizes of device, host, and user memory allocated by the application
 *
 * User memory is registered with am_tracker_add.
 **/
void am_memtracker_sizeinfo(const hc::accelerator &acc, size_t *deviceMemSize, size_t *hostMemSize, size_t *userMemSize);


void am_memtracker_update_peers(const hc::accelerator &acc, int peerCnt, hsa_agent_s *agents);

/*
 * Map device memory or hsa allocated host memory pointed to by @p ptr to the peers.
 * 
 * @p ptr pointer which points to device memory or host memory
 * @p num_peer number of peers to map
 * @p peers pointer to peer accelerator list.
 * @return AM_SUCCESS if mapped successfully.
 * @return AM_ERROR_MISC if @p ptr is nullptr or @p num_peer is 0 or @p peers is nullptr.
 * @return AM_ERROR_MISC if @p ptr is not am managed.
 * @return AM_ERROR_MISC if @p ptr is not found in the pointer tracker.
 * @return AM_ERROR_MISC if @p peers incudes a non peer accelerator.
 */
am_status_t am_map_to_peers(void* ptr, size_t num_peer, const hc::accelerator* peers); 

/*
 * Locks a host pointer to a vector of agents
 * 
 * @p ac acclerator corresponding to current device
 * @p hostPtr pointer to host memory which should be page-locked
 * @p size size of hostPtr to be page-locked
 * @p visibleAc pointer to hcc accelerators to which the hostPtr should be visible
 * @p numVisibleAc number of elements in visibleAc
 * @return AM_SUCCESS if lock is successfully.
 * @return AM_ERROR_MISC if lock is unsuccessful.
 */
am_status_t am_memory_host_lock(hc::accelerator &ac, void *hostPtr, size_t size, hc::accelerator *visibleAc, size_t numVisibleAc);

/*
 * Unlock page locked host memory
 * 
 * @p ac current device accelerator
 * @p hostPtr host pointer 
 * @return AM_SUCCESS if unlocked successfully.
 * @return AM_ERROR_MISC if @p hostPtr unlock is un-successful.
 */
am_status_t am_memory_host_unlock(hc::accelerator &ac, void *hostPtr);


}; // namespace hc

