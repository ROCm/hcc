macro(ensure_libcxx_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
MESSAGE("libcxx is present.")
else(EXISTS "${dest_dir}/${name}")
Find_Package(Git)

if(${GIT_FOUND})
  execute_process( COMMAND ${GIT_EXECUTABLE} clone --depth 1 http://llvm.org/git/libcxx ${dest_dir}/${name} )
else(${GIT_FOUND})
  MESSAGE(FATAL_ERROR "Upstream llvm is not present at ${dest_dir}/${name} and git could not be found")
endif()

endif()
endmacro()
