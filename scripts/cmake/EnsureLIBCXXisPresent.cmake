macro(ensure_libcxx_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
MESSAGE("libcxx is present.")
else(EXISTS "${dest_dir}/${name}")
Find_Package(Subversion)

if(${Subversion_FOUND})
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} co -q http://llvm.org/svn/llvm-project/libcxx/trunk ${dest_dir}/${name} )
else(${Subversion_FOUND})
  MESSAGE(FATAL_ERROR "Upstream llvm is not present at ${dest_dir}/${name} and svn could not be found")
endif()

endif()
endmacro()
