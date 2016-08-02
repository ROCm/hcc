add_llvm_loadable_module( LLVMPromote
  Promote.cpp
  )

add_dependencies(LLVMPromote intrinsics_gen)
