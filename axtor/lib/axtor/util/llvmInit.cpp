/*
 * llvmInit.cpp
 *
 *  Created on: 09.04.2011
 *      Author: gnarf
 */
#include <axtor/util/llvmInit.h>

void axtor::initLLVM()
{
	llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
	llvm::initializeCore(Registry);
	llvm::initializeScalarOpts(Registry);
	llvm::initializeIPO(Registry);
	llvm::initializeAnalysis(Registry);
	llvm::initializeIPA(Registry);
	llvm::initializeTransformUtils(Registry);
	llvm::initializeInstCombine(Registry);
	llvm::initializeInstrumentation(Registry);
	llvm::initializeTarget(Registry);
}
