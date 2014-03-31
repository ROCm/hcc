macro(ensure_libcxxrt_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
MESSAGE("libcxxrt is present.")
else(EXISTS "${dest_dir}/${name}")

set(GIT_REPOSITORY git://github.com/pathscale/libcxxrt.git)

find_package(Git)
if(NOT GIT_EXECUTABLE)
  MESSAGE(FATAL_ERROR "Upstream libcxxrt is not present at ${dest_dir}/${name} and git could not be found")
else()
  execute_process( COMMAND ${GIT_EXECUTABLE} clone ${GIT_REPOSITORY} ${dest_dir}/${name} )
endif()

endif()
endmacro()
