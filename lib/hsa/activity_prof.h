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

#define ACTIVITY_PROF_INSTANCES()                                        \
    namespace activity_prof {                                            \
        static activity_id_callback_t activity_id_callback;              \
        static ActivityCallbacksTable callbacks;                         \
        template<> HSAOp::ActivityProf::timer_t* HSAOp::ActivityProf::_timer = NULL; \
        template<> std::atomic<record_id_t> HSAOp::ActivityProf::_glob_record_id(0); \
    } // activity_prof

namespace activity_prof {
using Kalmar::CLAMP::record_id_t;
typedef void* callback_arg_t;
typedef uint32_t op_id_t;
typedef uint32_t activity_kind_t;
typedef Kalmar::hcCommandKind command_id_t;

// Activity callbacks
template <typename IdCb, typename Fun>
class ActivityCallbacksTableTempl {
    public:
    typedef IdCb id_callback_fun_t;
    typedef Fun callback_fun_t;
    typedef std::recursive_mutex mutex_t;

    ActivityCallbacksTableTempl() {
        std::lock_guard<mutex_t> lck(_mutex);
        for (op_id_t i = 0; i < hc::HSA_OP_ID_NUM; ++i) {
            _arg[i] = NULL;
            _fun[i] = NULL;
        }
    }

    void set_id_callback(const id_callback_fun_t& fun) { _id_callback = fun; }
    id_callback_fun_t get_id_callback() const { return _id_callback; }

    bool set(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) {
        std::lock_guard<mutex_t> lck(_mutex);
        if (op_id == hc::HSA_OP_ID_ANY) {
            for (op_id_t i = 0; i < hc::HSA_OP_ID_NUM; ++i) {
                set(i, fun, arg);
            }
        } else if (op_id < hc::HSA_OP_ID_NUM) {
            _arg[op_id] = arg;
            _fun[op_id] = fun;
        } else {
          return false;
        }
        return true;
    }

    void get(const op_id_t& op_id, callback_fun_t* fun, callback_arg_t* arg) const {
        std::lock_guard<mutex_t> lck(*const_cast<mutex_t*>(&_mutex));
        *fun = _fun[op_id];
        *arg = _arg[op_id];
    }

    private:
    id_callback_fun_t _id_callback;
    callback_fun_t _fun[hc::HSA_OP_ID_NUM];
    callback_arg_t _arg[hc::HSA_OP_ID_NUM];
    mutex_t _mutex;
};

// Activity profile class
template <int Domain, typename OpCoord, typename IdCb, typename Fun, typename Record, typename Timer>
class ActivityProfTempl {
public:
    // Activity ID callback type
    typedef IdCb id_callback_fun_t;
    // Activity callback type
    typedef Fun callback_fun_t;
    // Activity record type
    typedef Record op_record_t;
    // Callbacks table type
    typedef ActivityCallbacksTableTempl<id_callback_fun_t, callback_fun_t> ActivityCallbacksTable;
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
    void initialize(std::atomic<record_id_t>& record_id_ref, const ActivityCallbacksTable& callbacks) {
        callbacks.get(_op_id, &_callback_fun, &_callback_arg);
        if (_callback_fun != NULL) {
            if (_timer == NULL) _timer = new timer_t;
            _record_id = record_id_ref.fetch_add(1, std::memory_order_relaxed);
            (callbacks.get_id_callback())(_record_id);
        }
    }

    // Activity callback routine
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {
        if (_callback_fun != NULL) {
            op_record_t record {
                Domain,                               // domain id
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
    callback_fun_t _callback_fun;
    callback_arg_t _callback_arg;
    record_id_t _record_id;
    timer_t*& _timer;
};

} // namespace activity_prof

#if USE_PROF_API
#include "prof_protocol.h"
#include "hsa_rt_utils.hpp"

namespace activity_prof {

typedef ActivityCallbacksTableTempl<activity_id_callback_t, activity_async_callback_t> ActivityCallbacksTable;
template <typename OpCoord>
class ActivityProf : public ActivityProfTempl<ACTIVITY_DOMAIN_HCC_OPS,
                                              OpCoord,
                                              activity_id_callback_t,
                                              activity_async_callback_t,
                                              activity_record_t,
                                              hsa_rt_utils::Timer> {
public:
    typedef ActivityProfTempl<ACTIVITY_DOMAIN_HCC_OPS,
                             OpCoord,
                             activity_id_callback_t,
                             activity_async_callback_t,
                             activity_record_t,
                             hsa_rt_utils::Timer> parent_t;
    typedef hsa_rt_utils::Timer timer_t;

    ActivityProf(const op_id_t& op_id, const OpCoord& op_coord) : parent_t(op_id, op_coord, _timer) {}
    inline void initialize(const ActivityCallbacksTable& callbacks) {
      parent_t::initialize(_glob_record_id, callbacks);
    }

private:
    static timer_t* _timer;
    static std::atomic<record_id_t> _glob_record_id;
};

typedef ActivityCallbacksTable::id_callback_fun_t id_callback_fun_t;
typedef ActivityCallbacksTable::callback_fun_t callback_fun_t;

} // namespace activity_prof

#else

namespace activity_prof {
typedef void* id_callback_fun_t;
typedef void* callback_fun_t;

class ActivityCallbacksTable {
    public:
    void set_id_callback(const id_callback_fun_t& fun) {}
    bool set(const op_id_t& op_id, const callback_fun_t& fun, const callback_arg_t& arg) { return true; }
};

template <typename OpCoord>
class ActivityProf {
public:
    ActivityProf(const op_id_t& op_id, const OpCoord& op_coord) {}
    inline void initialize(const ActivityCallbacksTable& callbacks) {}
    inline void callback(const command_id_t& command_id, const uint64_t& begin_ts, const uint64_t& end_ts,
                         const size_t& bytes = 0) {}
    typedef void* timer_t;
    static timer_t* _timer;
};

} // namespace activity_prof

#endif

#endif // ACTIVITY_PROF_H
