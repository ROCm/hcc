macro(ensure_llvm_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
MESSAGE("LLVM seems to present. Force updating to version "${LLVM_REVISION})
execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} switch -r ${LLVM_REVISION} http://llvm.org/svn/llvm-project/llvm/trunk )

execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} switch -r ${LLVM_REVISION} -q http://llvm.org/svn/llvm-project/compiler-rt/trunk ${dest_dir}/${name}/projects/compiler-rt )
else(EXISTS "${dest_dir}/${name}")
Find_Package(Subversion)
if(${Subversion_FOUND})
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} co -r ${LLVM_REVISION} -q http://llvm.org/svn/llvm-project/llvm/trunk ${dest_dir}/${name} )
  execute_process( COMMAND ${Subversion_SVN_EXECUTABLE} co -r ${LLVM_REVISION} -q http://llvm.org/svn/llvm-project/compiler-rt/trunk ${dest_dir}/${name}/projects/compiler-rt )
else(${Subversion_FOUND})
  MESSAGE(FATAL_ERROR "Upstream llvm is not present at ${dest_dir}/${name} and svn could not be found")
endif()

endif()
endmacro()

