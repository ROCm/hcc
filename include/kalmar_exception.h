//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

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

} // namespace Kalmar

