/*
 * llvmBuiltins.h
 *
 *  Created on: 31 Jan 2012
 *      Author: v1smoll
 */

#ifndef LLVMBUILTINS_H_
#define LLVMBUILTINS_H_

#include <llvm/Module.h>

namespace axtor {
	llvm::Function * create_memcpy(llvm::Module & M, std::string funcName, uint destSpace, uint srcSpace);

	llvm::Function * create_memset(llvm::Module & M, std::string funcName, uint space);
}


#endif /* LLVMBUILTINS_H_ */
