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

#include <mutex>

#include "kalmar_runtime.h"

#if USE_PROF_API
#include "prof_protocol.h"
#include "hsa_rt_utils.hpp"

#define ACTIVITY_PROF_INSTANCES()                                                    \
    namespace activity_prof {                                                        \
        CallbacksTable::table_t CallbacksTable::_table{};                            \
        CallbacksTable::mutex_t CallbacksTable::_mutex;                              \
        template<> HSAOp::ActivityProf::timer_t* HSAOp::ActivityProf::_timer = NULL; \
        template<> std::atomic<record_id_t> HSAOp::ActivityProf::_glob_record_id(0); \
    } // activity_prof

namespace activity_prof {
typedef uint64_t record_id_t;
typedef uint32_t op_id_t;
typedef uint32_t activity_kind_t;
typedef Kalmar::hcCommandKind command_id_t;

typedef activity_id_callback_t id_callback_fun_t;
typedef activity_async_callback_t callback_fun_t;
typedef void* callback_arg_t;

class CallbacksTable {
    public:
    struct table_t {
        id_callback_fun_t id_callback;
        callback_fun_t fun[hc::HSA_OP_ID_NUM];
        callback_arg_t arg[hc::HSA_OP_ID_NUM];
    };
    typedef std::recursive_mutex mutex_t;

    static void set_id_callback(const id_callback_fun_t& fun) { _table.id_callback = fun; }
    static id_callback_fun_t get_id_callback() { return _table.id_callback; }
    
    static bool set_async_callback(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) {
        std::lock_guard<mutex_t> lck(_mutex);
        if (op_id == hc::HSA_OP_ID_ANY) {
            for (op_id_t i = 0; i < hc::HSA_OP_ID_NUM; ++i) {
                set_async_callback(i, fun, arg);
            }
        } else if (op_id < hc::HSA_OP_ID_NUM) {
            _table.fun[op_id] = fun;
            _table.arg[op_id] = arg;
        } else {
          return false;
        }
        return true;
    }
    
    static void get_async_callback(const op_id_t& op_id, callback_fun_t* fun, callback_arg_t* arg) {
        std::lock_guard<mutex_t> lck(_mutex);
        *fun = _table.fun[op_id];
        *arg = _table.arg[op_id];
    }

    private:
    static table_t _table;
    static mutex_t _mutex;
};

// Activity profile class
template <typename OpCoord>
class ActivityProf {
public:
    // Domain ID
    static const int ACTIVITY_DOMAIN_ID = ACTIVITY_DOMAIN_HCC_OPS;
    // Timeer type
    typedef hsa_rt_utils::Timer timer_t;

    ActivityProf(const op_id_t& op_id, const OpCoord& op_coord) :
        _op_id(op_id),
        _op_coord(op_coord),
        _callback_fun(NULL),
        _callback_arg(NULL),
        _record_id(0)
    {}

    // Initialization
    void initialize() {
        CallbacksTable::get_async_callback(_op_id, &_callback_fun, &_callback_arg);
        if (_callback_fun != NULL) {
            if (_timer == NULL) _timer = new timer_t;
            _record_id = _glob_record_id.fetch_add(1, std::memory_order_relaxed);
            (CallbacksTable::get_id_callback())(_record_id);
        }
    }

    // Activity callback routine
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {
        if (_callback_fun != NULL) {
            activity_record_t record {
                ACTIVITY_DOMAIN_ID,                   // domain id
                (activity_kind_t)command_id,          // activity kind
                _op_id,                               // operation id
                _record_id,                           // activity correlation id
                _timer->timestamp_to_ns(begin_ts),    // begin timestamp, ns
                _timer->timestamp_to_ns(end_ts),      // end timestamp, ns
                _op_coord._deviceId,                  // device id
                _op_coord._queueId,                   // stream id
                bytes                                 // copied data size, for memcpy
            };
            _callback_fun(_op_id, &record, _callback_arg);
        }
    }

private:
    const op_id_t _op_id;
    const OpCoord& _op_coord;
    activity_async_callback_t _callback_fun;
    callback_arg_t _callback_arg;
    record_id_t _record_id;

    static timer_t* _timer;
    static std::atomic<record_id_t> _glob_record_id;
};

} // namespace activity_prof

#else
#define ACTIVITY_PROF_INSTANCES() do {} while(0)

namespace activity_prof {
typedef void* id_callback_fun_t;
typedef void* callback_fun_t;

struct CallbacksTable {
    static void set_id_callback(const id_callback_fun_t& fun) {}
    static bool set_async_callback(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) { return true; }
};

template <typename OpCoord>
class ActivityProf {
public:
    ActivityProf(const op_id_t& op_id, const OpCoord& op_coord) {}
    inline void initialize() {}
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {}
};

} // namespace activity_prof

#endif

#endif // ACTIVITY_PROF_H
