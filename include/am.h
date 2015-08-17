/* AM : Accelerator Memory
    @file am.h
    @brief Header for C-style API for managing accelerator memory ("AM").
*/

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace hc {

typedef int am_status_t;

/** 
Pointer types and address spaces. 
*/



typedef int am_accelerator_view_t ; /* TODO */


// Provide automatic type conversion for void*.
class auto_voidp {
    void *_ptr;
    public:
        auto_voidp (void *ptr) : _ptr (ptr) {}
        template<class T> operator T *() { return (T *) _ptr; }
};

#if 0
//#define AM_MAX_EVENT_DEP 4

// TODO - add real fields to this structure...
struct am_event_s {
    // hsa signal
    int dep_count;
    //struct am_event_s  dep[AM_MAX_EVENT_DEP];
} ;
typedef struct am_event_s am_event_t;

extern am_event_t am_null_event;
#endif

// TODO - alternative design would use an IFDEF to select 
// whether am_event_s contains HSA signal or CL-event.
// Saves an indirection but would require selecting back-end at compile-time?
typedef struct am_event_s {
    void * _handle; // TODO - need this?

    uint64_t _timestamp;
} am_event_t;


/**
Accelerator Identifer can be one of the following special codes, or 
a positive integer from 1..N, where N is the number of accelerators in the system.
*/

//! All accelerators in the system TODO - explain which APIs support this and what effect it has 
#define AM_ACCELERATOR_ALL		  -3 

//! Default accelerator, for this thread.  Set with am_set_default_accelerator
#define AM_ACCELERATOR_DEFAULT	  -2 

//! Host accelerator
#define AM_ACCELERATOR_HOST		  0 

//! No accelerator or host.  See API routines for interpretation of this field.
#define AM_ACCELERATOR_NONE       -3 


/**
Error Codes
*/
#define AM_SUCCESS                           0
#define AM_ERROR_MEMORY_ALLOCATION          -1
#define AM_ERROR_INVALID_ARG                -2
#define AM_ERROR_UNSUPPORTED_MODE           -3 /* Use a feature not supported by this AM Mode */
#define AM_ERROR_INVALID_ACCELERATOR_VIEW   -4
#define AM_ERROR_MISC                       -6 /** Misellaneous error */
#define AM_ERROR_TBD                        -42 /** Error logic to be created - these indicate TODO items in the implementation.*/


//---------------

// TODO - assign codes.
#define AM_DEFAULT_FLAGS        0x00

/** Allocate pinned memory on the host. */
#define AM_HOST_PINNED          0x01

/** Allocate memory on device which is mapped as write-combining in host VA space */
#define AM_HOST_WRITE_COMBINED  0x02

#define AM_HOST_READ_ONLY_ACCESS 0xF00D//TODO
#define AM_HOST_READ_WRITE_ACCESS 0xF00D//TODO
#define AM_HOST_WRITE_ONLY_ACCESS 0xF00D//TODO
#define AM_HOST_NO_ACCESS 0xF00D//TODO- delete me?

/** Request Read-Only access at the target accelerator.  Will allow other agents to maintain Shared Read-Only copies */
#define AM_ACCELERATOR_READ_ONLY_ACCESS  0xF00D

/** Request RW access at the target accelerator.   Will invalidate data from other accelerator caches. */
#define AM_ACCELERATOR_READ_WRITE_ACCESS 0xF00D  

/** Request Write-Only access at the target accelerator.  Will invalidate data from other accelerator caches. */
#define AM_ACCELERATOR_WRITE_ONLY_ACCESS 0xF00D


// Pointer is block aligned.
#define AM_BLOCK_ALIGN  0xF00D// TODO - add to alloc description.

/** Disable automatic sync when a block is filled from host memory to an accelerator cache. 
    This can be useful in cases where the memory is written by the accelerator before being read.
    Care should be taken with this flag as it effects a block of memory
*/
#define AM_DISABLE_AUTO_SYNC_IN      0xF00D

#define AM_ENABLE_AUTO_SYNC_IN  0xF00D     

/** Disable automatic synchronization when a block is evicted from the accelerator cache to the host.
    This is useful in cases where the data is not needed on the host. */
#define AM_DISABLE_AUTO_SYNC_OUT     0xF00D 

#define AM_ENABLE_AUTO_SYNC_OUT      0xF00D

/** Combine both DISABLE_SYNC flags. Application will explicitly manage synchronization for this
memory region using @ref am_copy or @ref am_update.
*/
#define AM_EXPLICIT_SYNC (AM_DISABLE_AUTO_SYNC_IN | AM_DISABLE_AUTO_SYNC_OUT)

/** Support fine-grain coherency for the memory region.
*/
#define AM_FINE_GRAIN_COHERENCY  0xF00D


/** 
Do not cache block in accelerator memory.  This flag is recommended for streaming data structures which
are expected to offer little cache re-use and thus should not be cached to avoid creating thrash with
other, more useful data structures.  
*/ 
#define AM_DISABLE_CACHING  0xF00D
#define AM_ENABLE_CACHING  0xF00D

/**
 These flags control the replacement priority for blocks cached in accelerator memories.  
 NORMAL is the default and will give the block one "life" in the replacement algorithm, ie it will be evicted
 from the cache when first selected as a victim.  "HIGH" gives the block two trips through the replacement
 algorithm, and blocks with very-high priority take three trips through the replacement algorithm.
 */
#define AM_HINT_REPLACEMENT_PRIORITY_VERY_HIGH  0xF00D
#define AM_HINT_REPLACEMENT_PRIORITY_HIGH  0xF00D
#define AM_HINT_REPLACEMENT_PRIORITY_NORMAL  0xF00D


// TODO - remove these?
#define AM_SYNC_RESERVE_ONLY    0xF00D
#define AM_EVICT_DISCARD        0xF00D


/** Holds info for a pointer managed by AM, returned by am_get_pointer_info */
typedef struct {
	size_t	size; /* size specified at allocation */
	int		flags;   /* flags specified at allocation */
	void   *acc_ptr;  /* For registered memory, the pointer valid for use on the accelerator. */
} am_pointer_info_s;
typedef am_pointer_info_s *am_pointer_info_t;

// Functions:
//am_event_t am_null_event; // TODO - initialize to empty event?  Removge me?

/**  
	@brief Allocate memory in the unified virtual address space, create a mapping on the host and all accelerators.


	@param [out] ptr     *ptr is written with the allocated pointer.
	@param [in]  size	 Requested allocation size in bytes
	@param [in]  flags  Enables control over special types of allocation. See below.

		
	Allocates @p size bytes, and returns in @p *ptr a pointer to the allocated memory.  
    The allocated memory is aligned on a 64-Kbyte boundary. // TODO - make this 16B with a flag?

	ptr is in the unified virtual address space and can be used on either the host or the specified accelerator,  
    provided appropriate synchronization is employed.  Several synchronization options are possible - see the flags 
    below for more information.

    For auto memory (without AM_EXPLICIT_SYNC), AM will use page protection bits on the CPU and accelerators
    to limit unnecessary copies between the host and accelerator.  Specifically, the initial state of
    host memory created by @ref am_alloc is marked as read-only.  Host writes will be detected by AM and 
    used to track which pages are dirty and need to be copied to the GPU.  See @ref Optimization for more information.

    The returned UVM pointer is guaranteed to never alias with other allocations from am_alloc or with other CPU-side
    addresses, such as pointers returned by malloc or addresses of stack variables.

    The @p flags provide considerable flexibility in how the memory is allocated:
		- AM_HOST_PINNED : Allocate pinned host virtual space.  This flag will attempt to pin the memory, even on systems which support accelerator access to unpinned memory.  
		- AM_HOST_WRITE_COMBINED : Allocate write-combined host virtual space.    
            It is recommended to combine @ref am_lock with write-combined memory so that the app can guarantee that the stores
            are streaming into the accelerator cache (rather than host memory).  An example is shown here: @ref write_combined.cpp.

	        @warning Write-combining allows fast writing from host memory to the target accelerator's memory space, but cannot be 
            read efficiently by the host on most CPUs. 

        - AM_HOST_NO_ACCESS : Do not allocate virtual space for CPU access. TBD.  How to handle evictions?

	*/
auto_voidp am_alloc(size_t size, unsigned flags, am_status_t *err) ;


/**  
	 @brief Free memory allocated through am* calls.  
 
	Frees the memory referenced by ptr, which must have been returned by a previous call to @ref am_alloc.  

    The virtual address range is deallocated on host and all accelerators, and may be re-assigned to another allocation.

    Additionally, any cached data associated with the specified pointer is also deallocated and may be used to cache other data.

	If ptr is NULL, no memory operation is performed and AM_SUCCESS is returned.  
    am_free(NULL) will initialize the AM system, and this can be useful for programs that want to avoid overhead of lazy initilization on first call.
	@ return : AM_SUCCESS : am_free always returns AM_SUCCESS.
 */
am_status_t am_free(void*  ptr);



/**
	@brief Copy memory, and optionally inject it into a target accelerator's cache.

	This function may be used to copy between host to host, host to device, device to host, or device to device.   
	The source and destination memory regions must not overlap.

    @param av  Specify target accelerator cache for copy data:
        - AM_ACCELERATOR_DEFAULT or specific accelerator id : Inject copied data into the specified accelerator's cache.
        - AM_ACCELERATOR_HOST : Invalidate data in destination range from all accelerator caches and copy data to host physical memory.
        - AM_ACCELERATOR_NONE : Do not perform cache injection.
        - AM_ACCELERATOR_ALL : Multicast and mark read-only?
	
	@return TODO
	*/
am_status_t am_copy(void*  dst, const void*  src, size_t size, am_accelerator_view_t dst_av);

// TODO
am_status_t am_copy_async(void*  dst, const void*  src, size_t size, am_accelerator_view_t dst_av, am_event_t wait_event, am_event_t completion_event);


/**
	@brief Provide advice on the intended use of memory

    Advice can apply to a specific accelerator or to all accelerators.

    TODO : Can advise be used to transition memory?
    TODO : Add debug mode to enable memory protection changes.
*/
am_status_t am_advise(void*  ptr, size_t size, am_accelerator_view_t av, int advice);


/**
	@brief Synchronize memory to the specified accelerator_view.  

	@param av Specify target accelerator to synchronize. 
	@param ptr  Pointer to memory to synchronize.  This must be same pointer returned by am_alloc.
	
	@return 

	Calling this functions synchronizes any modifications to the specified accelerator view.  
        - If @p av specifies the host accelerator (AM_ACCELERATOR_HOST), then the host memory storage is synchronized from 
          any dirty copies on accelerators.   
        - If @p av specifies an accelerator (a specific accelerator ID or AM_ACCELERATOR_DEFAULT), this function updates the cache of the specified accelerator.
        - If @p av is AM_ACCELERATOR_ALL, then the data is broadcast and synchronized read-only to all accelerators.
        - If @p av is AM_ACCELERATOR_NONE, then the function has no effect.


    The synchronized range is guaranteed to enclose the range specified by @p ptr + @p size at the cache block granularity:
        - The @p ptr is rounded down to the block granularity.
        - @p ptr+@p size is rounded up to the block granularity.

	For "auto" managed memory, this API can be used to control the synchronizing of data either from host to a target accelerator, 
	or from accelerator back to host.  Synchronizing is an optimization to control the timing of data transfer:
		- By default data transfer to the device typically occurs just before the kernel is launched.  This API allows applicaitons to initiate the
		 host-to-device synchronization earlier, perhaps hiding some of the latency of the transfer.
		- Data transfer from the accelerator to the host typically is deferred to when the data is first referenced on the host.  This API 
		allows applications to initiate the device-to-host transfer earlier.    

	For memory allocated with am_alloc with flag AM_EXPLICIT_SYNC, this API can be used to synchronize the host and accelerator caches. 
	For explicit memory, there is no automatic memory copy, and this API (or explicit am_copy or am_copy_async) MUST be called to ensure
	the data is appropriately synchronized before use.   This API provides the benefit that the virtual address on host and accelerator 
	is the same - this can be useful for structures which contain embedded pointers. 
		
	For HSA devices with Page Migration Cache, this API may prefetch some or all of the specified data to the target accelerator.

    Flags:
        - AM_SYNC_RESERVE_ONLY : Space is requested in the target cache but is not actually synchronized.  Useful with WC target.  
           Must specify an accelerator in @p av. It is an error if @p av is AM_ACCELERATOR_NONE or AM_ACCELERATOR_HOST.  TODO - needed?

        - AM_ACCELERATOR_READ_ONLY_ACCESS :
        - AM_ACCELERATOR_READ_WRITE_ACCESS :
        - AM_ACCELERATOR_WRITE_ONLY_ACCESS :

        - AM_HOST_READ_ONLY_ACCESS : 

    @param ptr      Pointer to start of memory range to synchronize.   Will be rounded down to containing cache block.
    @param size     Size of memory to synchronize.      
    @param av       Accelerator to synchronize the memory to.
    @param flags    Flags to control synchronization properties.

    @returns : TODO


*/
am_status_t am_update(void*  ptr, size_t size, am_accelerator_view_t av, unsigned flags);




/**
	@brief Synchronize memory.  Function returns immediately and executes asyncronously in background.

	For HSA devices with Page Migration Cache, this API is a hint to prefetch some or all 
    of the specified data to the target accelerator. 
	The kernel may begin executing before the synchronize operation completes.  For auto memory this
    will will work functionally and may be faster since it allows overlapping data transfer 
    with the kernel execution.  
*/
am_status_t am_update_async(void*  ptr, size_t size, am_accelerator_view_t av, unsigned flags, am_event_t wait_event, am_event_t completion_future);


/**
	@brief Evict memory from specified accelerator view.

    Data will be synchronized back to the host memory, unless AM_EVICT_DISCARD is specified.  If AM_EVICT_DISCARD is
    specified, the data will be discarded from the specified caches.  This flag should be used carefully and only in cases where
    it is known that the data is no longer needed.

    After eviction, the cache space may be re-assigned to other hot blocks.

    @param [in] ptr  Pointer to evict. 
    @param [in] flags  
    @param [in] av Data is evicted from the accelerator specifid in @p av.  If @p av is AM_ACCELERATOR_ALL, then the data is evicted from all cached.  
        - If @p av is AM_ACCELERATOR_NONE, this API has no effect.

*/
am_status_t am_evict(void*  ptr, size_t size, am_accelerator_view_t av, unsigned flags);


/**
	@brief: Register the specified host_ptr so that it can be accessed on the specified accelerator(s).  

	This call will register the specified @p ptr and return an address in accelerator memory space.  
	On some systems, this may page-lock the memory.  

    On Gen1 systems, the pointer returned by @p reg_ptr may differ from @p ptr, and may only be used on the 
	targeted accelerator(s), and not on the host.   This is one exception to the unified virtual address space model, since we have
    host pointer and device pointer which have different virtual addresses but refer to the same data.
    This API is designed to support legacy hardware without full 48-bit addressing capability.  Gen2 and later hardware
    remove this restriction, and therefore the returned *acc_ptr will be equivalent to ptr.  On gen2 systems, 
    am_register must still be called for raw pointers since it ensures that the desired mapping is available on the 
    target accelerator(s).  For gen3 systems, calling @ref am_register is no longer a requirement (these devices can
    access raw pointers in host memory directly), but still recommended when possible since this gives AM information about the 
    size of the memory-range which can be useful for optimization purposes.  

    AM will attempt to register the memory-range specified by ptr and size.  The memory-range may include pages which are
    not migrateable - ie because some pages in the range are not valid, or have inconsistent permissions.  If this occurs, 
    this API will return a valid virtual address but also returns a warning in the form of the AM_WARNING_REGISTER_RANGE_FAILED.  
    In this case, the registered memory cannot be migrated or cached in any accelerator.  To avoid this warning and associated
    performance cost, pointers passed to am_register are recommended (but not required) to follow these guidelines:

        * Align the size of pointer on a 64K page boundary.
        * Pass a size with is a multiple of 64K.

    TODO - how to handle registering memory with aliases (ie from shmget)

    AM created a mapping on all accelerators in the AM space, and thus the returned @p acc_ptr can be used on any accelerator. 
    Each accelerator uses the same mapping ; however as noted above the host pointer may be different on gen1 platforms.
    (TODO - describe behavior for systems with mixed gen1/gen2 accelerators)

  
	On systems that support shared page tables between accelerator and device (ie HSA APUs), this function will not page-lock the memory. 
	However, the function may still be useful because it informs the runtime that the memory range may be used on the accelerator - this
	can enable page migration optimization and prefetching.  

    @ref am_register can be called multiple times for the same memory-range.  Each call will increment a reference count for the memory-range.
    It is not legal to register or unregister "sub-ranges" of an existing memory-range. (TODO - is this too restrictive?) 
    
	The memory registered by this function must be unregistered with @ref am_unregister


	@param [out] acc_ptr	 Returns registered pointer in the specified accelerator address space.
	@param [in]  ptr		 Host pointer to register.  
	@param [in]  size		 Size of pointer
	@param [in]  flags       TODO - allow subset of MADVISE flags here?

	@return	AM_SUCCESS, AM_ERROR_INVALID_ARG, AM_ERROR_MEMORY_ALLOCATION
    @return AM_WARNING_REGISTER_RANGE_FAILED
*/
am_status_t am_register(void **acc_ptr, const void * host_ptr, size_t size, unsigned flags);


/**
	@brief: Remove reference to registered memory.

    Decrement the reference count for the memory-range whose base address is specified by @p ptr. 
    If the reference count reches 0, the memory-range is no longer tracked by AM and cannot be used by an accelerator.

	The base address must be the same address supplied in the @p ptr argument to am_register().
    It is not legal to unregister "sub-ranges" of an existing memory-range. (TODO - is this too restrictive?) 

    @warning: Note this function accepts a *host* pointer used in the original AM allocation - not the device pointer
    returned am_register.

	@param [in] ptr : Host pointer to unregister.
    @return 
        - AM_SUCCESS : Call succeeded.
        - AM_ERROR_INVALID_ARG : If ptr was not previously registered with am_register, or has already been unregistered.
*/
am_status_t am_unregister(void * host_ptr);

/**
    @brief: Locks the specified memory range into memory, and updates valid virtual mappings on host and accelerators.


    The locked range is guaranteed to enclose the range specified by @p ptr+@p size at the cache block granularity:
        - The @p ptr is rounded down to the block granularity.
        - @p ptr+@p size is rounded up to the block granularity.

    Locking memory is an expensive operation.

    P2P capabilities are only supported on 64-bit systems which contain BAR regions large enough to expose the entire
    cache.  Thus AM lock does not distinguish between "visible" and "invisible" regions of the cache; either the
    entire cache is visible or P2P is not supported.

    The memory will be synchronized to the @p av if specified.  This API performs synchronization and locking 
    performed atomically, meaning that the synchronized blocks cannot be evicted before the lock completes.

    Data which is not cached at the time that am_lock is called, but which is later cached, will not
    be locked.

   @param ptr   UVM pointer to the block to lock.
   @param size  size of the memory to lock in memory. Must not be 0.
   @param av  Accelerator to sync the memory to before locking.
    - AM_ACCELERATOR_DEFAULT or valid accelerator number : Memory is synchronized to the specified accelerator before locking.
    - AM_ACCELERATOR_ALL  : Return AM_ERROR_INVALID_ARG.  Memory can only be locked into accelerator at a time.
    - AM_ACCELERATOR_NONE : No implicit synchronization to a target accelerator is performed.  Data may be 
        cached in zero or more accelerators, and the locked blocks may be spread among different accelerators
        or uncached (ie on the host).
    - AM_ACCELERATOR_HOST : AM_ERROR_INVALID_ARG.  Memory can only be locked into an accelerator.  (TODO?)

   @return 
    - AM_SUCCESS : memory succcessfully locked.
    - AM_ERROR_INVALID_ARG : see above for illegal arguments.
    - AM_ERROR_MEMORY_ALLOCATION : if the lock fails (ie size too long, invalid virtual range)

*/
am_status_t am_lock(void*  ptr, size_t size, am_accelerator_view_t av);


/**
    @brief: Unlocks the specified ptr.
*/
am_status_t am_unlock(void*  ptr);


/*
    @brief: This function must be called before launching a kernel that uses automatic memory.

    am_launch_sync ensures that automatic memory is visible on the GPU.

    On most languages this function will be called by the code which launches the kernel, and thus will
    not need to be explicitly called by the programmer.
*/
am_status_t am_launch_sync(am_accelerator_view_t av);


/**
	@brief : Return info about the specified ptr info.

	@param[out] am_pointer_info_s : pointer to structure which is written with the info
	@param[in]  ptr : pointer to query for  

	@return :   
*/
am_status_t am_get_pointer_info(am_pointer_info_s *am_pointer_info, void*  ptr);


/**
	@brief Set the default accelerator for this thread.  

	The default accelerator is used when AM_ACCELERATOR_DEFAULT is passed to one of the am functions.

	This API uses thread-local-storage (TLS) - thus each thread maintains its own default accelerator.
*/
am_status_t am_set_default_accelerator(am_accelerator_view_t av);

am_status_t am_get_default_accelerator(am_accelerator_view_t *av);

am_status_t am_set_default_advice(am_accelerator_view_t av, int advice);

/**
    @brief Query attribute information for the target accelerator.
*/
am_status_t am_query_accelerator(am_accelerator_view_t av); // TODO


// TODO - add descriptions.
am_status_t am_event_create(am_event_t *e, unsigned flags);
am_status_t am_event_destroy(am_event_t e);
am_status_t am_event_record(am_event_t e, am_accelerator_view_t av);

// TODO - move this to HIP? - does it belong in AM?
am_status_t am_event_elapsed_time(float *ms, am_event_t start, am_event_t stop);

// Notes: Synchronization model is based on av (queues) and events.
// Copy and Sync command have both sync and async flavors.
// Sync waits for command to complete before returning to host.
// Async dispatches the command but does not wait for it to finish.
// Items sent to same AV are dispatched in order.  
// grid_launch uses the same event syntax.
// ? Can we handle returned events?  Mem management seems hard.  But could return a pointer to something else.
am_status_t am_event_wait(am_event_t) ;
am_status_t am_event_add_dep(am_event_t) ;
//int am_event_add_dep_or(am_event_t) ; // OR-style dependencies?

// Wait for all outstanding copy and sync commands targeting the specified accelerator.?
am_status_t am_accelerator_wait(am_accelerator_view_t av);




// TODO - Control runtime cache bits? ie mark-dirty through madvise?
// TODO - how to support multi-cast? 
// TODO - fine-grain coherency - add to intro.
// TODO - how do fallback RDMA cases work?
// TODO - add args to copy. Maybe make this copy_ext ?  Add _ext routines for everythi9ng ?
// TODO - write first version of API using HSA APIs, HSA extensions, and Thunk.

// TODO - link in the RDMA examples to intro.
// 

// TODO - design notes:
//  Unified memory by default.
//    One hole in registered memory on today's system - mark that API as weird??
//    Allocation does not specify an accelerator, the virtual address is valid on any accelerator.
//    Copies and synchronization can specify an accelerator - this allows these APIs to perform cache injection.
// TODO - Design does not contain 'contexts' - all accelerators in the system exist in the shared address space.
// Porting from CUDA.
// _ Lazy initialization.
//

} // namespace hc
