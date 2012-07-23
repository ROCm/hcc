macro(ensure_llvm_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
MESSAGE("llvm is present")
else(EXISTS "${dest_dir}/${name}")
Find_Package(Subversion)
Find_Package(Mercurial)

if(${Subversion_FOUND})
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} co http://llvm.org/svn/llvm-project/llvm/trunk ${dest_dir}/${name} )
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} propset svn:ignore ${name} ${dest_dir}  )
  Find_Program(MERCIRUAL_EXECUTABLE hg)
  execute_process( COMMAND ${MERCURIAL_EXECUTABLE} clone https://hskim@bitbucket.org/multicoreware/cppamp ${dest_dir}/${name}/tools/clang )
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} co http://llvm.org/svn/llvm-project/compiler-rt/trunk ${dest_dir}/${name}/projects/compiler-rt )
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} propset svn:ignore compiler-rt ${dest_dir}/projects  )
else(${Subversion_FOUND})
  MESSAGE(FATAL_ERROR "Upstream llvm is not present at ${dest_dir}/${name} and svn could not be found")
endif()

endif()
endmacro()

