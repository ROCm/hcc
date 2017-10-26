#pragma once

#include "hc_am.hpp"

namespace hc {
  namespace internal {

   /**
    * Allocate a block of size bytes of host coherent system memory.
    *
    * The contents of the newly allocated block of memory are not initialized.
    *
    * @return : On success, pointer to the newly allocated memory is returned.
    * The pointer is typecast to the desired return type.
    *
    * If an error occurred trying to allocate the requested memory, nullptr is returned.
    *
    * Use am_free to free the newly allocated memory.
    *
    * @see am_free, am_copy
    */
    auto_voidp am_alloc_host_coherent(size_t);

  } // namespace internal
} // namespace hc
