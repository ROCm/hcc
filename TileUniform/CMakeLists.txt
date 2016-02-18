add_llvm_loadable_module( LLVMTileUniform
  TileUniform.cpp
  )

add_dependencies(LLVMTileUniform intrinsics_gen)
