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

#include "kalmar_runtime.h"

#define ACTIVITY_PROF_INSTANCES(current, callbacks)                      \
    namespace activity_prof {                                            \
        static thread_local current_t current{};                         \
        static ActivityCallbacksTable callbacks;                         \
        template<> HSAOp::ActivityProf::timer_t* HSAOp::ActivityProf::_timer = NULL; \
    } // activity_prof

namespace activity_prof {
using Kalmar::CLAMP::record_id_t;
typedef void* callback_arg_t;
typedef uint32_t op_id_t;
typedef uint32_t activity_kind_t;
typedef Kalmar::hcCommandKind command_id_t;

// Current thread local data
struct current_t {
    record_id_t record_id;
    int device_id;
    uint64_t queue_id;
};

// Activity callbacks
template <typename Fun>
class ActivityCallbacksTableTempl {
    public:
    typedef Fun callback_fun_t;
    typedef std::atomic<Fun> fun_ptr_t;

    ActivityCallbacksTableTempl() {
        for (op_id_t i = 0; i < hc::HSA_OP_ID_NUM; ++i) {
          _arg[i] = NULL;
          _fun[i].store(NULL);
        }
    }

    bool set(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) {
        if (op_id == hc::HSA_OP_ID_ANY) {
            for (op_id_t i = 0; i < hc::HSA_OP_ID_NUM; ++i) {
                set(i, fun, arg);
            }
        } else if (op_id < hc::HSA_OP_ID_NUM) {
            _arg[op_id] = arg;
            _fun[op_id].store(fun, std::memory_order_release);
        } else {
          return false;
        }
        return true;
    }

    void get(const op_id_t& op_id, callback_fun_t* fun, callback_arg_t* arg) const {
        *fun = _fun[op_id].load(std::memory_order_acquire);
        *arg = _arg[op_id];
    }

    private:
    fun_ptr_t _fun[hc::HSA_OP_ID_NUM];
    callback_arg_t _arg[hc::HSA_OP_ID_NUM];
};

// Activity profile class
template <int Domain, typename OpCoord, typename Fun, typename Record, typename Timer>
class ActivityProfTempl {
public:
    // Activity callback type
    typedef Fun callback_fun_t;
    // Activity record type
    typedef Record op_record_t;
    // Callbacks table type
    typedef ActivityCallbacksTableTempl<callback_fun_t> ActivityCallbacksTable;
    // Timeer type
    typedef Timer timer_t;

    ActivityProfTempl(const op_id_t& op_id, const OpCoord& op_coord, timer_t* &timer) :
        _op_id(op_id),
        _op_coord(op_coord),
        _callback_fun(NULL),
        _callback_arg(NULL),
        _record_id(0),
        _timer(timer)
    {}

    // Initialization
    void initialize(current_t& current, const ActivityCallbacksTable& callbacks) {
        callbacks.get(_op_id, &_callback_fun, &_callback_arg);
        if (_callback_fun != NULL) {
            _record_id = current.record_id;
            current.device_id = _op_coord._deviceId;
            current.queue_id = _op_coord._queueId;

            if (_timer == NULL) _timer = new timer_t;
        }
    }

    // Activity callback routine
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {
        if (_callback_fun != NULL) {
            op_record_t record {
                Domain,                               // domain id
                _op_id,                               // operation id
                (activity_kind_t)command_id,          // command id
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
    callback_fun_t _callback_fun;
    callback_arg_t _callback_arg;
    record_id_t _record_id;
    timer_t* &_timer;
};

} // namespace activity_prof

#if USE_PROF_API
#include "prof_protocol.h"
#include "hsa_rt_utils.hpp"

namespace activity_prof {

typedef ActivityCallbacksTableTempl<activity_async_callback_t> ActivityCallbacksTable;
template <typename OpCoord>
class ActivityProf : public ActivityProfTempl<ACTIVITY_DOMAIN_HCC_OPS,
                                              OpCoord,
                                              activity_async_callback_t,
                                              activity_record_t,
                                              hsa_rt_utils::Timer> {
public:
    typedef ActivityProfTempl<ACTIVITY_DOMAIN_HCC_OPS,
                             OpCoord,
                             activity_async_callback_t,
                             activity_record_t,
                             hsa_rt_utils::Timer> parent_t;
    typedef hsa_rt_utils::Timer timer_t;

    ActivityProf(const op_id_t& op_id, const OpCoord& op_coord) : parent_t(op_id, op_coord, _timer) {}

private:
    static timer_t* _timer;
};

typedef ActivityCallbacksTable::callback_fun_t callback_fun_t;

} // namespace activity_prof

#else

namespace activity_prof {
typedef void* callback_fun_t;

class ActivityCallbacksTable {
    public:
    bool set(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) { return true; }
    void check(const op_id_t& op_id, callback_fun_t* fun, callback_arg_t* arg) const {}
};

template <typename OpCoord>
class ActivityProf {
public:
    ActivityProf(const op_id_t& op_id, const OpCoord& op_coord) {}
    void initialize(current_t& current, const ActivityCallbacksTable& callbacks) {}
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {}
    typedef void* timer_t;
    static timer_t* _timer;
};

} // namespace activity_prof

#endif

#endif // ACTIVITY_PROF_H
