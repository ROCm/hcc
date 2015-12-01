macro(patch_LLVM_for_ClangOMP name)

MESSAGE("Sync LLVM with that used by clang-omp.")
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/LLVM.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

endmacro()
