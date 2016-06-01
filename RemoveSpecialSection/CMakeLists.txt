add_llvm_loadable_module( LLVMRemoveSpecialSection
  RemoveSpecialSection.cpp
  )

add_dependencies(LLVMRemoveSpecialSection intrinsics_gen)
