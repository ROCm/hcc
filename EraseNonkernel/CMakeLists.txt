add_llvm_loadable_module( LLVMEraseNonkernel
  EraseNonkernel.cpp
  )

add_dependencies(LLVMEraseNonkernel intrinsics_gen)
