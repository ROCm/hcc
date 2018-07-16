//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <exception>

namespace Kalmar {

#ifndef E_FAIL
#define E_FAIL 0x80004005
#endif

static const char *__errorMsg_UnsupportedAccelerator = "concurrency::parallel_for_each is not supported on the selected accelerator \"CPU accelerator\".";

typedef int HRESULT;
class runtime_exception : public std::exception
{
public:
  runtime_exception(const char * message, HRESULT hresult) throw() : _M_msg(message), err_code(hresult) {}
  explicit runtime_exception(HRESULT hresult) throw() : err_code(hresult) {}
  runtime_exception(const runtime_exception& other) throw() : _M_msg(other.what()), err_code(other.err_code) {}
  runtime_exception& operator=(const runtime_exception& other) throw() {
    _M_msg = *(other.what());
    err_code = other.err_code;
    return *this;
  }
  virtual ~runtime_exception() throw() {}
  virtual const char* what() const throw() {return _M_msg.c_str();}
  HRESULT get_error_code() const {return err_code;}

private:
  std::string _M_msg;
  HRESULT err_code;
};

class invalid_compute_domain : public runtime_exception
{
public:
  explicit invalid_compute_domain (const char * message) throw()
  : runtime_exception(message, E_FAIL) {}
  invalid_compute_domain() throw()
  : runtime_exception(E_FAIL) {}
};

class accelerator_view_removed : public runtime_exception
{
public:
  explicit accelerator_view_removed (const char * message, HRESULT view_removed_reason) throw()
  : runtime_exception(message, view_removed_reason) {}
  accelerator_view_removed(HRESULT view_removed_reason) throw()
  : runtime_exception(view_removed_reason) {}
  HRESULT get_view_removed_reason() const throw() { return get_error_code(); }
};

} // namespace Kalmar

