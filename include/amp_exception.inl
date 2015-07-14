//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <kalmar_exception.h>

namespace Concurrency {

using runtime_exception = Kalmar::runtime_exception;
using HRESULT = Kalmar::HRESULT;

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

} // namespace Concurrency 

