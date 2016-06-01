add_llvm_loadable_module(LLVMWrapperGen
  WrapperGen.cpp
  )

add_dependencies(LLVMWrapperGen intrinsics_gen)
