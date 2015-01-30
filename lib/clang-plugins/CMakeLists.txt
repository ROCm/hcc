# in clamp build mode
set (LLVM_INC_DIR "${PROJECT_SOURCE_DIR}/compiler/include")
set (CLANG_SRC_DIR "${PROJECT_SOURCE_DIR}/compiler/tools/clang")
set (COMPILER_BUILD_DIR ${PROJECT_BINARY_DIR}/compiler/lib)
set (CLANG_BUILD_DIR ${PROJECT_BINARY_DIR}/compiler/tools/clang)
# clamp headers path
set (CLAMP_INC_DIR ${PROJECT_SOURCE_DIR}/include)

set (INC_DIRS ${CLAMP_INC_DIR}
    ${LIBCXX_INC_DIR}
    "${CLANG_SRC_DIR}/include"
    "${LLVM_INC_DIR}"
    "${CLANG_BUILD_DIR}/include"
    ${BOLT_INC_DIRS} )

set (LIB_DIRS ${LIBCXX_LIB_DIR} 
    ${LIBCXXRT_LIB_DIR}
    "${COMPILER_BUILD_DIR}/lib" )
if (CXXAMP_ENABLE_BOLT)
  list(APPEND LIB_DIRS ${BOLT_LIB_DIRS})
endif()
  
add_definitions (-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)
add_definitions (-D_GNU_SOURCE -DHAVE_CLANG_CONFIG_H)
add_definitions (-D__STDC_FORMAT_MACROS)
 
set (CMAKE_CXX_FLAGS "-fPIC -Wall -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wno-unused-parameter -Wwrite-strings  -fno-rtti ")
set (CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
 
set (LLVM_LIBS
  LLVMJIT
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Disassembler
  LLVMExecutionEngine
  LLVMAsmPrinter
  LLVMSelectionDAG
  LLVMX86AsmPrinter
  LLVMX86Info
  LLVMMCParser
  LLVMCodeGen
  LLVMX86Utils
  LLVMScalarOpts
  LLVMInstCombine
  LLVMTransformUtils
  LLVMipa
  LLVMAnalysis
  LLVMTarget
  LLVMCore
  LLVMMC
  LLVMSupport
  LLVMBitReader
  LLVMOption
)

macro(add_clang_plugin name)
  set (srcs ${ARGN})
 
  include_directories (${INC_DIRS})
  link_directories (${LIB_DIRS})
  
  # The plugin needs to be shared object
  add_library( ${name} SHARED ${srcs} )
  
  if (SYMBOL_FILE)
    set_target_properties( ${name} PROPERTIES LINK_FlAGS
      "-exported_symbols_list ${SYMBOL_FILE}" )
  endif()
 
  foreach (clang_lib ${CLANG_LIBS})
    target_link_libraries( ${name} ${clang_lib} )  
  endforeach()
  
  foreach (llvm_lib ${LLVM_LIBS})
    target_link_libraries( ${name} ${llvm_lib} )
  endforeach()
  
  foreach (user_lib ${USER_LIBS})
    target_link_libraries( ${name} ${user_lib} )
  endforeach()

  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_CXX_FLAGS "-I${LIBCXX_INC_DIR} -I${CLAMP_INC_DIR} -stdlib=libc++ -std=c++11" )
 
endmacro(add_clang_plugin)

set(SYMBOL_FILE StmtRewriter.exports)
 
set ( CLANG_LIBS
  clang
  clangFrontend
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontendTool
  clangLex
  clangParse
  clangSema
  clangEdit
  clangSerialization
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
)
set ( USER_LIBS
  mcwamp
  cxxrt
  c++ 
)

add_clang_plugin(StmtRewriter StmtRewriter.cpp ParallelRewriter.cpp)
  
install(TARGETS StmtRewriter
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
