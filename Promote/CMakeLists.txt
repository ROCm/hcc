add_llvm_loadable_module( LLVMPromote
  Promote.cpp
  PromotePrivate.cpp
  MallocSelect.cpp
  )

add_dependencies(LLVMPromote intrinsics_gen)
