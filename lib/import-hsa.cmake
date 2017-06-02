
find_path(HSA_HEADER hsa/hsa.h
  PATHS
    /opt/rocm/include
)

find_library(HSA_LIBRARY hsa-runtime64
  PATHS
    /opt/rocm/lib
)

add_library(hsa-runtime64 SHARED IMPORTED GLOBAL)

set_target_properties(hsa-runtime64 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${HSA_HEADER}"
  IMPORTED_LOCATION "${HSA_LIBRARY}"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HSA_HEADER}"
)
