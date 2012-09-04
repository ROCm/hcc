/*
 * SharedContext.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/util/SharedContext.h>

namespace axtor {

llvm::LLVMContext * SharedContext::context = NULL;

void SharedContext::init()
{
	if (!context)
		context = &llvm::getGlobalContext();
}

llvm::LLVMContext & SharedContext::get()
{
	init();
	return *context;
}

}
