/*
 * TrivialReturnSplitter.cpp
 *
 *  Created on: 08.02.2011
 *      Author: gnarf
 */

#include <axtor/pass/TrivialReturnSplitter.h>
#include <axtor/util/llvmDuplication.h>

namespace axtor {
	llvm::RegisterPass<axtor::TrivialReturnSplitter> __regReturnSplitter("returnsplitter", "axtor - trivial return splitter",
		true,
		false);

	char TrivialReturnSplitter::ID = 0;

	TrivialReturnSplitter::TrivialReturnSplitter() :
		llvm::FunctionPass(ID)
	{}

	bool TrivialReturnSplitter::runOnFunction(llvm::Function & F)
	{
		for(llvm::Function::iterator block = F.begin(); block != F.end(); ++block)
		{
			if (llvm::isa<llvm::ReturnInst>(block->begin()))
			{
				BlockCopyTracker tracker(*F.getParent());
				splitNode(tracker, block);
				return true;
			}
		}
		return false;
	}

	void TrivialReturnSplitter::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{
	}

	const char * TrivialReturnSplitter::getPassName() const
	{
		return "axtor - trivial returning block splitter";
	}

	TrivialReturnSplitter::~TrivialReturnSplitter()
	{
	}
}
