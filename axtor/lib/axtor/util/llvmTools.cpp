/*
 * llvmTools.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/util/llvmTools.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/IRReader.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/system_error.h>
#include <llvm/Bitcode/ReaderWriter.h>

#include <iostream>

namespace axtor {

	llvm::Module* createModuleFromFile(std::string fileName)
	{
		llvm::LLVMContext & context = SharedContext::get();
		return createModuleFromFile(fileName, context);
	}


	llvm::Module* createModuleFromFile(std::string fileName, llvm::LLVMContext & context)
	{
		llvm::SMDiagnostic diag;
		llvm::Module * mod = llvm::getLazyIRFileModule(fileName, diag, context);
		if (!mod) {
      std::cout << diag.getMessage() << "\n"; 
			return 0;
    }

#ifdef DEBUG
    std::cout << "module created correctly\n";
#endif

		std::string errInfo;
		mod->MaterializeAll(&errInfo);
		if (errInfo == "")
			return mod;
		else
			return 0;
	}


	void writeModuleToFile(llvm::Module * M, const std::string & fileName)
	{
		assert (M);
		std::string errorMessage = "";
		llvm::raw_fd_ostream file(fileName.c_str(), errorMessage);
		llvm::WriteBitcodeToFile(M, file);
		file.close();
	}

}

