//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <hc/hc.hpp>
#include <hc/hc_runtime.hpp>

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <mutex>

// TODO: this shouldn't be squatting in the global namespace.
enum am_status_t { AM_ERROR_MISC = -1, AM_SUCCESS };
enum am_memory_t {
    am_device, am_host_pinned, am_host_noncoherent, am_host_coherent
};

namespace hc
{
    namespace detail
    {
        class auto_voidp {
            // Provide automatic type conversion for void*.
            // TODO: not very robust, replace.
            void* ptr_{};
        public:
            auto_voidp(void* ptr) : ptr_{ptr} {}

            template<typename T>
            operator T*() const { return static_cast<T*>(ptr_); }
        };
    } // Namespace detail.

    struct AmPointerInfo {
        // Info for each pointer in the memory tracker.
        // TODO: ROCr already tracks all of this, making it redundant.
        void* host_pointer{};             ///< Host pointer. If host access is
                                          ///  not allowed, NULL.
        void* device_pointer{};           ///< Device pointer.
        void* unaligned_device_pointer{}; ///< Unaligned device pointer
        std::size_t size_bytes{};         ///< Size of allocation.
        hc::accelerator* acc{};           ///< Accelerator where allocation is
                                          ///  physically located.
        bool is_in_device_mem{false};     ///< Memory is physically resident on
                                          ///  a device (if false, memory is
                                          /// located on host)
        bool is_am_managed{false};        ///< Memory was allocated by AM and
                                          ///  should be freed when am_reset is
                                          ///  called.
        std::uint64_t alloc_seq_num{};    ///< Sequence number of allocation.
        void* app_pointer{};              ///< App-specific pointer to
                                          ///  additional information.

        // creates a dummy copy of AmPointerInfo
        AmPointerInfo() = default;
        AmPointerInfo(
            void* host_ptr,
            void* device_ptr,
            void* unaligned_device_ptr,
            std::size_t size,
            hc::accelerator& acc,
            bool is_device_mem = false,
            bool is_am_mem = false)
            :
            host_pointer{host_ptr},
            device_pointer{device_ptr},
            unaligned_device_pointer{unaligned_device_ptr},
            size_bytes{size},
            acc{&acc},
            is_in_device_mem{is_device_mem},
            is_am_managed{is_am_mem},
            alloc_seq_num{0},
            app_pointer{nullptr}
        {}
        AmPointerInfo(const AmPointerInfo&) = default;
        AmPointerInfo(AmPointerInfo&&) = default;
        ~AmPointerInfo() = default;

        AmPointerInfo& operator=(const AmPointerInfo&) = default;
        AmPointerInfo& operator=(AmPointerInfo&&) = default;
    };

    /**
     * Allocate a block of @p size bytes of memory on the specified @p acc.
     *
     * The contents of the newly allocated block of memory are not initialized.
     *
     * If @p size == 0, 0 is returned.
     *
     * Flags:
     *  am_host_pinned : Allocated pinned host memory and map it into the
     *                   address space of the specified accelerator.
     *
     * @return : On success, pointer to the newly allocated memory is returned.
     * The pointer is typecast to the desired return type.
     *
     * If an error occurred trying to allocate the requested memory, 0 is
     * returned.
     *
     * @see am_free, am_copy
     */
    template<typename Accelerator>
    inline
    detail::auto_voidp am_aligned_alloc(
        std::size_t size,
        Accelerator& acc,
        std::uint32_t flags,
        std::size_t alignment = 0)
    {   // TODO: this logic should be reviewed, it is interesting.
        if (size == 0u) return nullptr;
        if (!acc.is_hsa_accelerator()) return nullptr;

        hsa_region_t* region{};
        switch (flags) {
        case am_host_pinned : case am_host_noncoherent :
            region = static_cast<hsa_region_t*>(acc.get_hsa_am_system_region());
            break;
        case am_host_coherent :
            region = static_cast<hsa_region_t*>(
                acc.get_hsa_am_finegrained_system_region());
            break;
        default :
            region = static_cast<hsa_region_t*>(acc.get_hsa_am_system_region());
        }

        if (!region || region->handle == 0) {
            region = static_cast<hsa_region_t*>(acc.get_hsa_am_system_region());
        }

        size = (alignment == 0) ? size : (size + alignment);
        void* r{nullptr};
        detail::throwing_hsa_result_check(
            hsa_memory_allocate(*region, size, &r),
            __FILE__, __func__, __LINE__);

        static const auto round_up_to_next_multiple =
            [](std::uintptr_t x, std::uintptr_t y) {
            x = x + y - 1;
            return x - x % y;
        };

        return reinterpret_cast<void*>(round_up_to_next_multiple(
            reinterpret_cast<std::uintptr_t>(r), alignment ? alignment : 1));
    }

    /**
     * Allocate a block of @p size bytes of memory on the specified @p acc.
     *
     * The contents of the newly allocated block of memory are not initialized.
     *
     * If @p size == 0, 0 is returned.
     *
     * Flags:
     *  amHostPinned : Allocated pinned host memory and map it into the address
     *                 space of the specified accelerator.
     *
     * @return : On success, pointer to the newly allocated memory is returned.
     * The pointer is typecast to the desired return type.
     *
     * If an error occurred trying to allocate the requested memory, 0 is
     * returned.
     *
     * @see am_free, am_copy
     */
    template<typename Accelerator>
    inline
    detail::auto_voidp am_alloc(
        std::size_t size, Accelerator& acc, std::uint32_t flags)
    {
        return am_aligned_alloc(size, acc, flags, 0u);
    }

    namespace detail
    {
        inline
        hsa_amd_pointer_info_t hsa_pointer_info(void* ptr)
        {
            hsa_amd_pointer_info_t r{};
            r.size = sizeof(hsa_amd_pointer_info_t);
            detail::throwing_hsa_result_check(
                hsa_amd_pointer_info(ptr, &r, nullptr, nullptr, nullptr),
                __FILE__, __func__, __LINE__);

            return r;
        }
    }
    /**
     * Free a block of memory previously allocated with am_alloc.
     *
     * @return AM_SUCCESS
     * @see am_alloc, am_copy
     */
    inline
    am_status_t am_free(void* ptr)
    {
        if (!ptr) return AM_SUCCESS;

        auto tmp = detail::hsa_pointer_info(ptr);

        if (tmp.type != HSA_EXT_POINTER_TYPE_HSA) return AM_ERROR_MISC;

        detail::throwing_hsa_result_check(
            hsa_memory_free(tmp.agentBaseAddress),
            __FILE__, __func__, __LINE__);

        return AM_SUCCESS;
    }

    /**
     * Copy @p size bytes of memory from @p src to @ dst. The memory areas
     * (src+size and dst+size) must not overlap.
     *
     * @return AM_SUCCESS on error or AM_ERROR_MISC if an error occurs.
     * @see am_alloc, am_free
     */
    __attribute__((deprecated(
        "use accelerator_view::copy instead (and note src/dst order"
        "reversal)")))
    am_status_t am_copy(void* dst, const void* src, std::size_t size);

    /**
     * Return information about tracked pointer.
     *
     * AM tracks pointers when they are allocated or added to tracker with
     * am_track_pointer.
     * The tracker tracks the base pointer as well as the size of the
     * allocation, and will find the information for a pointer anywhere in the
     * tracked range.
     *
     * @returns AM_ERROR_MISC if pointer is not currently being tracked. In this
     * case, @p info is not modified.

    * @returns AM_SUCCESS if pointer is tracked and writes info to @p info. If
    * @info is NULL, no info is written but the returned status indicates if the
    * pointer was tracked.
    *
    * @see AM_memtracker_add
    */
    inline
    am_status_t am_memtracker_get_info(hc::AmPointerInfo* info, const void* ptr)
    {
        if (!ptr) return AM_SUCCESS;

        auto tmp = detail::hsa_pointer_info(const_cast<void*>(ptr));

        if (tmp.type == HSA_EXT_POINTER_TYPE_UNKNOWN) return AM_ERROR_MISC;

        info->host_pointer = tmp.hostBaseAddress;
        info->device_pointer = tmp.agentBaseAddress;
        info->unaligned_device_pointer = tmp.agentBaseAddress;
        info->size_bytes = tmp.sizeInBytes;
        // hc::accelerator* acc{};           ///< Accelerator where allocation is
        //                                   ///  physically located.
        // bool is_in_device_mem{false};     ///< Memory is physically resident on
        //                                   ///  a device (if false, memory is
        //                                   /// located on host)
        // bool is_am_managed{false};        ///< Memory was allocated by AM and
        //                                   ///  should be freed when am_reset is
        //                                   ///  called.
        //std::uint64_t alloc_seq_num{};    ///< Sequence number of allocation.
        info->app_pointer = tmp.userData;

        return AM_SUCCESS;
    }

    /**
     * Add a pointer to the memory tracker.
     *
     * @return AM_ERROR_MISC : If @p ptr is NULL, or info._sizeBytes = 0, the
     *                         info is not added to the tracker and
     *                         AM_ERROR_MISC is returned.
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
    am_status_t am_memtracker_update(
        const void* ptr,
        std::int32_t appId,
        std::uint32_t allocationFlags,
        void* appPtr = nullptr);

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
     * Remove all memory allocations associated with specified accelerator from
     * the memory tracker.
     *
     * @returns Number of entries reset.
     * @see am_memtracker_getinfo
     */
    std::size_t am_memtracker_reset(const hc::accelerator& acc);

    /**
     * Print the entries in the memory tracker table.
     *
     * Intended primarily for debug purposes.
     * @see am_memtracker_getinfo
     **/
    inline
    void am_memtracker_print(void* targetAddress = nullptr)
    {
        if (!targetAddress) return;

        // const char* targetAddressP = static_cast<const char*>(targetAddress);
        // std::ostream &os = std::cerr;

        // uint64_t beforeD = std::numeric_limits<uint64_t>::max();
        // uint64_t afterD = std::numeric_limits<uint64_t>::max();
        // auto closestBefore = g_amPointerTracker.end();
        // auto closestAfter = g_amPointerTracker.end();
        // bool foundMatch = false;

        // for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
        //     const auto basePointer = static_cast<const char*> (iter->first._basePointer);
        //     const auto endPointer = static_cast<const char*> (iter->first._endPointer);
        //     if ((targetAddressP >= basePointer) && (targetAddressP < endPointer)) {
        //         ptrdiff_t offset = targetAddressP - basePointer;
        //         os << "db: memtracker found pointer:" << targetAddress << " offset:" << offset << " bytes inside this allocation:\n";
        //         os << "   " << iter->first._basePointer << "-" << iter->first._endPointer << "::  ";
        //         os << iter->second << std::endl;
        //         foundMatch = true;
        //         break;
        //     } else {
        //         if ((targetAddressP < basePointer) && (basePointer - targetAddressP < beforeD)) {
        //             beforeD = (basePointer - targetAddressP);
        //             closestBefore = iter;
        //         }
        //         if ((targetAddressP > endPointer) && (targetAddressP - endPointer < afterD)) {
        //             afterD = (targetAddressP - endPointer);
        //             closestAfter = iter;
        //         }
        //     };
        //     }

        //     if (!foundMatch) {
        //         os << "db: memtracker did not find pointer:" << targetAddress << ".  However, it is closest to the following allocations:\n";
        //         if (closestBefore != g_amPointerTracker.end()) {
        //             os << "db: closest before: " << beforeD << " bytes before base of: " << closestBefore->second << std::endl;
        //         }
        //         if (closestAfter != g_amPointerTracker.end()) {
        //             os << "db: closest after: " << afterD << " bytes after end of " << closestAfter->second << std::endl ;
        //         }
        //     }
        // } else {
        //     using namespace std;
        //     os <<  setw(PTRW) << "base" << "-" << setw(PTRW) << "end" << ": ";
        //     os  << setw(6+1) << "#SeqNum"
        //         << setw(PTRW+1) << "HostPtr"
        //         << setw(PTRW+1) << "DevPtr"
        //         << setw(12+1) << "SizeBytes"
        //         << setw(8+1) << "SizeMB"
        //         << setw(5) << "Dev?"
        //         << setw(6) << "Reg?"
        //         << setw(6) << " AppId"
        //         << setw(7) << " AppFlags"
        //         << setw(12) << left << " Peers" << right
        //         << "\n";

        //     for (auto iter = g_amPointerTracker.readerLockBegin() ; iter != g_amPointerTracker.end(); iter++) {
        //         os << setw(PTRW) << iter->first._basePointer << "-" << setw(PTRW) << iter->first._endPointer << ": ";
        //         printShortPointerInfo(os, iter->second);
        //         printRocrPointerInfo(os, iter->first._basePointer);
        //         os << "\n";
        //     }
        // }

        // g_amPointerTracker.readerUnlock();
    }

    /**
     * Return total sizes of device, host, and user memory allocated by the
     * application.
     *
     * User memory is registered with am_tracker_add.
     **/
    void am_memtracker_sizeinfo(
        const hc::accelerator& acc,
        std::size_t* deviceMemSize,
        std::size_t* hostMemSize,
        std::size_t* userMemSize);


    void am_memtracker_update_peers(
        const hc::accelerator& acc, int peerCnt, hsa_agent_t* agents);

    /*
    * Map device memory or hsa allocated host memory pointed to by @p ptr to the
    * peers.
    *
    * @p ptr pointer which points to device memory or host memory
    * @p num_peer number of peers to map
    * @p peers pointer to peer accelerator list.
    * @return AM_SUCCESS if mapped successfully.
    * @return AM_ERROR_MISC if @p ptr is nullptr or @p num_peer is 0 or @p peers
    *                       is nullptr.
    * @return AM_ERROR_MISC if @p ptr is not am managed.
    * @return AM_ERROR_MISC if @p ptr is not found in the pointer tracker.
    * @return AM_ERROR_MISC if @p peers includes a non peer accelerator.
    */
    template<typename Accelerator>
    inline
    am_status_t am_map_to_peers(
        void* ptr, std::size_t num_peer, const Accelerator* peers)
    {
        if (!ptr) return AM_ERROR_MISC;
        if (num_peer == 0u) return AM_ERROR_MISC;
        if (!peers) return AM_ERROR_MISC;

        auto tmp = detail::hsa_pointer_info(ptr);

        if (tmp.type != HSA_EXT_POINTER_TYPE_HSA) return AM_ERROR_MISC;

        std::vector<hsa_agent_t> as{num_peer};
        while (num_peer--) {
            as[num_peer] =
                *static_cast<hsa_agent_t*>(peers[num_peer].get_hsa_agent());
        }
        const auto s =
            hsa_amd_agents_allow_access(as.size(), as.data(), nullptr, ptr);

        if (s == HSA_STATUS_SUCCESS) return AM_SUCCESS;

        return AM_ERROR_MISC;
    }

    /*
    * Locks a host pointer to a vector of agents
    *
    * @p ac accelerator corresponding to current device
    * @p hostPtr pointer to host memory which should be page-locked
    * @p size size of hostPtr to be page-locked
    * @p visibleAc pointer to hcc accelerators to which the hostPtr should be
    *    visible
    * @p numVisibleAc number of elements in visibleAc
    * @return AM_SUCCESS if lock is successfully.
    * @return AM_ERROR_MISC if lock is unsuccessful.
    */
    template<typename Accelerator>
    inline
    am_status_t am_memory_host_lock(
        Accelerator& acc,
        void* hostPtr,
        std::size_t size,
        Accelerator* visibleAcc,
        std::size_t numVisibleAcc)
    {
        (void)acc;

        if (!hostPtr) return AM_SUCCESS;

        std::vector<hsa_agent_t> ag{numVisibleAcc};
        while (numVisibleAcc--) {
            ag[numVisibleAcc] = *static_cast<hsa_agent_t*>(
                visibleAcc[numVisibleAcc].get_hsa_agent());
        }

        void* p{};
        const auto s =
            hsa_amd_memory_lock(hostPtr, size, ag.data(), ag.size(), &p);

        (void)p;

        if (s == HSA_STATUS_SUCCESS) return AM_SUCCESS;

        return AM_ERROR_MISC;
    }

    /*
    * Unlock page locked host memory
    *
    * @p ac current device accelerator
    * @p hostPtr host pointer
    * @return AM_SUCCESS if unlocked successfully.
    * @return AM_ERROR_MISC if @p hostPtr unlock is un-successful.
    */
    template<typename Accelerator>
    inline
    am_status_t am_memory_host_unlock(Accelerator& acc, void* hostPtr)
    {
        (void)acc;

        if (!hostPtr) return AM_SUCCESS;

        if (hsa_amd_memory_unlock(hostPtr) == HSA_STATUS_SUCCESS) {
            return AM_SUCCESS;
        }

        return AM_ERROR_MISC;
    }
} // namespace hc

