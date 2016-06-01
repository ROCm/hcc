add_llvm_loadable_module( LLVMDivisionPrecision
  DivisionPrecision.cpp
  )

add_dependencies(LLVMDivisionPrecision intrinsics_gen)
