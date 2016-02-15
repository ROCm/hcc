add_llvm_loadable_module( LLVMCpuRename
    CpuRename.cpp
  )

add_dependencies(LLVMCpuRename intrinsics_gen)
