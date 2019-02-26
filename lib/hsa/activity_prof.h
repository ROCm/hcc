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

// Activity callbacks table
class CallbacksTable {
    public:
    struct table_t {
        id_callback_fun_t id_callback;
        callback_fun_t op_callback;
        callback_arg_t arg;
        std::atomic<bool> enabled[hc::HSA_OP_ID_NUMBER];
    };

    // Initialize record id callback and activity callback
    static void init(const id_callback_fun_t& id_callback, const callback_fun_t& op_callback, const callback_arg_t& arg) {
        _table.id_callback = id_callback;
        _table.op_callback = op_callback;
        _table.arg = arg;
    }

    static bool set_enabled(const op_id_t& op_id, const bool& enable) {
        bool ret = true;
        if (op_id < hc::HSA_OP_ID_NUMBER) {
            _table.enabled[op_id].store(enable, std::memory_order_release);
        } else {
            ret = false;
        }
        return ret;
    }

    static bool is_enabled(const op_id_t& op_id) {
        return _table.enabled[op_id].load(std::memory_order_acquire);
    }

    static id_callback_fun_t get_id_callback() { return _table.id_callback; }
    static callback_fun_t get_op_callback() { return _table.op_callback; }
    static callback_arg_t get_arg() { return _table.arg; }

    private:
    static table_t _table;
};

// Activity profile class
class ActivityProf {
public:
    // Domain ID
    static const int ACTIVITY_DOMAIN_ID = ACTIVITY_DOMAIN_HCC_OPS;
    // HSA timer
    typedef hsa_rt_utils::Timer timer_t;

    ActivityProf(const op_id_t& op_id, const uint64_t& queue_id, const int& device_id) :
        _op_id(op_id),
        _queue_id(queue_id),
        _device_id(device_id),
        _enabled(false),
        _record_id(0)
    {}

    // Initialization
    void initialize() {
        _enabled = CallbacksTable::is_enabled(_op_id);
        if (_enabled == true) {
            TimerFactory::Create();
            _record_id = _glob_record_id.fetch_add(1, std::memory_order_relaxed);
            (CallbacksTable::get_id_callback())(_record_id);
        }
    }

    template <class T>
    inline void report_gpu_timestamps(T* obj, const size_t& bytes = 0) {
        if (_enabled == true) {
            const command_id_t command_id = obj->getCommandKind();
            uint64_t start = obj->getBeginTimestamp();
            uint64_t end   = obj->getEndTimestamp();
            callback(command_id, start, end, bytes);
        }
    }

    template <class T>
    inline void report_system_ticks(T* obj, const size_t& bytes = 0) {
        if (_enabled == true) {
            const command_id_t command_id = obj->getCommandKind();
            uint64_t start = obj->getStartTick();
            uint64_t end   = obj->getSystemTicks();
            callback(command_id, start, end, bytes);
        }
    }

    bool is_enabled() { return _enabled; }

private:
    // Activity callback routine
    void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts, const size_t& bytes) {
        activity_record_t record {
            ACTIVITY_DOMAIN_ID,                   // domain id
            (activity_kind_t)command_id,          // activity kind
            _op_id,                               // operation id
            _record_id,                           // activity correlation id
            TimerFactory::Instance().timestamp_to_ns(begin_ts),    // begin timestamp, ns
            TimerFactory::Instance().timestamp_to_ns(end_ts),      // end timestamp, ns
            _device_id,                           // device id
            _queue_id,                            // queue id
            bytes                                 // copied data size, for memcpy
        };
        (CallbacksTable::get_op_callback())(_op_id, &record, CallbacksTable::get_arg());
    }

    const op_id_t _op_id;
    const uint64_t& _queue_id;
    const int& _device_id;

    bool _enabled;
    record_id_t _record_id;

    // Global record ID
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
    static void init(const id_callback_fun_t& id_callback, const callback_fun_t& op_callback, const callback_arg_t& arg) {}
    static bool set_enabled(const op_id_t& op_id, const bool& enable) { return false; }
};

class ActivityProf {
public:
    ActivityProf(const op_id_t& op_id, const uint64_t& queue_id, const int& device_id) {}
    inline void initialize() {}
    template <class T> inline void report_gpu_timestamps(T* obj, const size_t& bytes = 0) {}
    template <class T> inline void report_system_ticks(T* obj, const size_t& bytes = 0) {}
    inline bool is_enabled() { return false; }
};

} // namespace activity_prof

#endif

ACTIVITY_PROF_INSTANCES();

#endif // ACTIVITY_PROF_H
