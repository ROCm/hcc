add_llvm_loadable_module(LLVMDirectFuncCall
  DirectFuncCall.cpp
  )

add_dependencies(LLVMDirectFuncCall intrinsics_gen)
