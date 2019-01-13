/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ACTIVITY_PROF_H
#define ACTIVITY_PROF_H

#include <atomic>
#include <mutex>
#include <thread>

#include "kalmar_runtime.h"

#if USE_PROF_API
#include "prof_protocol.h"
#include "hsa_rt_utils.hpp"

#define ACTIVITY_PROF_INSTANCES()                                                    \
    hsa_rt_utils::Timer* hsa_rt_utils::TimerFactory::instance_ = nullptr;            \
    hsa_rt_utils::TimerFactory::mutex_t hsa_rt_utils::TimerFactory::mutex_;          \
    namespace activity_prof {                                                        \
        CallbacksTable::table_t CallbacksTable::_table{};                            \
        CallbacksTable::mutex_t CallbacksTable::_mutex;                              \
        std::atomic<record_id_t> ActivityProf::_glob_record_id(0);                   \
    } // activity_prof

namespace activity_prof {
typedef activity_correlation_id_t record_id_t;
typedef activity_op_t op_id_t;
typedef Kalmar::hcCommandKind command_id_t;
typedef hsa_rt_utils::TimerFactory TimerFactory;

typedef activity_id_callback_t id_callback_fun_t;
typedef activity_async_callback_t callback_fun_t;
typedef void* callback_arg_t;

class CallbacksTable {
    public:
    typedef std::mutex mutex_t;
    typedef std::atomic<uint32_t> sem_t;
    struct table_t {
        id_callback_fun_t id_callback;
        callback_fun_t fun[hc::HSA_OP_ID_NUMBER];
        callback_arg_t arg[hc::HSA_OP_ID_NUMBER];
        bool enabled[hc::HSA_OP_ID_NUMBER];
        sem_t sem[hc::HSA_OP_ID_NUMBER];
    };

    enum {
      WRITER_BIT = 0x1,
      READER_BIT = 0x2
    };

    static void sem_wait(sem_t& sem, const uint32_t mask) {
        while ((sem.load(std::memory_order_acquire) & mask) != 0) std::this_thread::yield();
    }
    static void reader_wait(sem_t& sem) {
        sem.fetch_sub(READER_BIT, std::memory_order_relaxed);
        sem_wait(sem, WRITER_BIT);
        sem.fetch_add(READER_BIT, std::memory_order_relaxed);
    }

    static void set_id_callback(const id_callback_fun_t& fun) { _table.id_callback = fun; }
    static id_callback_fun_t get_id_callback() { return _table.id_callback; }
    
    static bool set_async_callback(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) {
        bool ret = true;
        std::lock_guard<mutex_t> lck(_mutex);
        std::atomic<uint32_t>& sem = _table.sem[op_id];
        if (sem.fetch_add(WRITER_BIT, std::memory_order_acquire) != 0) sem_wait(sem, ~WRITER_BIT);

        if (op_id < hc::HSA_OP_ID_NUMBER) {
            _table.fun[op_id] = fun;
            _table.arg[op_id] = arg;
            _table.enabled[op_id] = (fun != nullptr);
        } else {
            ret = false;
        }

        sem.fetch_sub(WRITER_BIT, std::memory_order_release);
        return ret;
    }
    
    static bool get_async_callback(const op_id_t& op_id, callback_fun_t* fun, callback_arg_t* arg) {
        std::atomic<uint32_t>& sem = _table.sem[op_id];
        if ((sem.fetch_add(READER_BIT, std::memory_order_acquire) & WRITER_BIT) != 0) reader_wait(sem);

        *fun = _table.fun[op_id];
        *arg = _table.arg[op_id];
        const bool enabled = _table.fun[op_id];

        sem.fetch_sub(READER_BIT, std::memory_order_release);
        return enabled;
    }

    private:
    static table_t _table;
    static mutex_t _mutex;
};

// Activity profile class
class ActivityProf {
public:
    // Domain ID
    static const int ACTIVITY_DOMAIN_ID = ACTIVITY_DOMAIN_HCC_OPS;
    typedef hsa_rt_utils::Timer timer_t;

    ActivityProf(const op_id_t& op_id, const uint64_t& queue_id, const int& device_id) :
        _op_id(op_id),
        _queue_id(queue_id),
        _device_id(device_id),
        _enabled(false),
        _callback_fun(nullptr),
        _callback_arg(nullptr),
        _record_id(0)
    {}

    // Initialization
    void initialize() {
        _enabled = CallbacksTable::get_async_callback(_op_id, &_callback_fun, &_callback_arg);
        if (_enabled == true) {
            TimerFactory::Create();
            _record_id = _glob_record_id.fetch_add(1, std::memory_order_relaxed);
            (CallbacksTable::get_id_callback())(_record_id);
        }
    }

    // Activity callback routine
    void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {
        if (_enabled == true) {
            activity_record_t record {
                ACTIVITY_DOMAIN_ID,                   // domain id
                (activity_kind_t)command_id,          // activity kind
                _op_id,                               // operation id
                _record_id,                           // activity correlation id
                TimerFactory::Instance().timestamp_to_ns(begin_ts),    // begin timestamp, ns
                TimerFactory::Instance().timestamp_to_ns(end_ts),      // end timestamp, ns
                _device_id,                           // device id
                _queue_id,                            // stream id
                bytes                                 // copied data size, for memcpy
            };
            _callback_fun(_op_id, &record, _callback_arg);
        }
    }

private:
    const op_id_t _op_id;
    const uint64_t& _queue_id;
    const int& _device_id;

    bool _enabled;
    activity_async_callback_t _callback_fun;
    callback_arg_t _callback_arg;
    record_id_t _record_id;

    static std::atomic<record_id_t> _glob_record_id;
};

} // namespace activity_prof

#else
#define ACTIVITY_PROF_INSTANCES()

namespace activity_prof {
typedef uint32_t op_id_t;
typedef Kalmar::hcCommandKind command_id_t;

typedef void* id_callback_fun_t;
typedef void* callback_fun_t;
typedef void* callback_arg_t;

struct CallbacksTable {
    static void set_id_callback(const id_callback_fun_t& fun) {}
    static bool set_async_callback(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) { return true; }
};

class ActivityProf {
public:
    ActivityProf(const op_id_t& op_id, const uint64_t& queue_id, const int& device_id) {}
    inline void initialize() {}
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {}
};

} // namespace activity_prof

#endif

ACTIVITY_PROF_INSTANCES();

#endif // ACTIVITY_PROF_H
