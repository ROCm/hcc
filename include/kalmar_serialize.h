#pragma once

#include <set>
#include "kalmar_runtime.h"
#include "kalmar_exception.h"

/** \cond HIDDEN_SYMBOLS */
namespace Kalmar
{

/// traverse all the buffers that are going to be used in kernel
class FunctorBufferWalker {
public:
    virtual void Append(size_t sz, const void* s) {}
    virtual void AppendPtr(size_t sz, const void* s) {}
    virtual void visit_buffer(struct rw_info* rw, bool modify, bool isArray) = 0;
};

/// This is used to avoid incorrect compiler error
class Serialize {
    FunctorBufferWalker* vis;
public:
    Serialize(FunctorBufferWalker* vis) : vis(vis) {}
    void Append(size_t sz, const void* s) { vis->Append(sz, s); }
    void AppendPtr(size_t sz, const void* s) { vis->AppendPtr(sz, s); }
    void visit_buffer(struct rw_info* rw, bool modify, bool isArray) {
        vis->visit_buffer(rw, modify, isArray);
    }
};

/// Change the data pointer with device pointer
/// before/after kernel launches in cpu path
class CPUVisitor : public FunctorBufferWalker
{
    std::shared_ptr<KalmarQueue> pQueue;
    std::set<struct rw_info*> bufs;
public:
    CPUVisitor(std::shared_ptr<KalmarQueue> pQueue) : pQueue(pQueue) {}
    void visit_buffer(struct rw_info* rw, bool modify, bool isArray) override {
        if (isArray) {
            auto curr = pQueue->getDev()->get_path();
            auto path = rw->master->getDev()->get_path();
            if (path == L"cpu") {
                auto asoc = rw->stage->getDev()->get_path();
                if (asoc == L"cpu" || path != curr)
                    throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
            }
        }
        rw->sync(pQueue, modify, false);
        if (bufs.find(rw) == std::end(bufs)) {
            void*& device = rw->devs[pQueue->getDev()].data;
            void*& data = rw->data;
            bufs.insert(rw);
            std::swap(device, data);
        }
    }
};

/// Append kernel argument to kernel
class BufferArgumentsAppender : public FunctorBufferWalker
{
    std::shared_ptr<KalmarQueue> pQueue;
    void* k_;
    int current_idx_;
public:
    BufferArgumentsAppender(std::shared_ptr<KalmarQueue> pQueue, void* k)
        : pQueue(pQueue), k_(k), current_idx_(0) {}
    void Append(size_t sz, const void *s) override {
        CLAMP::PushArg(k_, current_idx_++, sz, s);
    }
    void AppendPtr(size_t sz, const void *s) override {
        CLAMP::PushArgPtr(k_, current_idx_++, sz, s);
    }
    void visit_buffer(struct rw_info* rw, bool modify, bool isArray) override {
        if (isArray) {
            auto curr = pQueue->getDev()->get_path();
            auto path = rw->master->getDev()->get_path();
            if (path == L"cpu") {
                auto asoc = rw->stage->getDev()->get_path();
                if (asoc == L"cpu" || path != curr)
                    throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
            }
        }
        rw->sync(pQueue, modify, false);
        pQueue->Push(k_, current_idx_++, rw->devs[pQueue->getDev()].data, modify);
    }
};

/// In C++AMP Standard V1.2 Line 3014
/// If pfe is launched without explicitly specified view, the target accelerator
/// and the view using which work is submitted to the accelerator, is chosen
/// from the objects of type array<T,N> that were captured in the kernel lambda.
///
/// Thise Searcher will visit all the array<T, N> and find a view to launch kernel
class QueueSearcher : public FunctorBufferWalker
{
    std::shared_ptr<KalmarQueue> pQueue;
public:
    QueueSearcher() = default;
    void visit_buffer(struct rw_info* rw, bool modify, bool isArray) override {
        if (isArray && !pQueue) {
            if (rw->master->getDev()->get_path() != L"cpu")
                pQueue = rw->master;
            else if (rw->stage->getDev()->get_path() != L"cpu")
                pQueue = rw->stage;
        }
    }
    std::shared_ptr<KalmarQueue> get_que() const { return pQueue; }
};

} // namespace Kalmar
/** \endcond */
