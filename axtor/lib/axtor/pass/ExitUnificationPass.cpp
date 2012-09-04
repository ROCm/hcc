/*
 * ExitUnificationPass.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/pass/ExitUnificationPass.h>
#include <axtor/util/llvmDebug.h>

namespace axtor {

llvm::RegisterPass<ExitUnificationPass> __regExitUnificationPass("loopexitenum", "axtor - loop exit enumeration pass",
					false,
                    false);

char ExitUnificationPass::ID = 0;

	/*
	 * generate a switch-like construct branching to elements of dest on their index
	 */
	void ExitUnificationPass::appendEnumerationSwitch(llvm::Value * val, std::vector<llvm::BasicBlock*> dests, llvm::BasicBlock * block)
	{
		llvm::SwitchInst * switchInst = llvm::SwitchInst::Create(val, NULL, dests.size(), block);

		uint exitID = 0;
		for(BlockVector::iterator it = dests.begin(); it != dests.end(); ++it, ++exitID)
		{
			llvm::BasicBlock * dest = *it;
			switchInst->addCase(get_uint(exitID), dest);
		}
	}

	/*
	 * if this loop has multiple exits to the parent loop enumerate them and move the branches to a dedicated exit block
	 *
	 * @return true, if the loop has changed
	 */
	bool ExitUnificationPass::unifyLoopExits(llvm::Function & func, llvm::Loop * loop)
	{
		llvm::Loop * parent = loop->getParentLoop();

		if (loop->getExitBlock())
			return false; // exactly one exit

		if (! parent) {
			std::cerr << "LEE: outer-most multi exit loop\n";
			return false; // the exits will not reach this loop again, so we do not care
		} else {
			std::cerr << "LEE: inner multi exit..\n";
		}

		BlockPairVector edges;
		getExitEdges(*loop, edges);

		llvm::Type * intType = llvm::IntegerType::get(SharedContext::get(), 32);

		llvm::BasicBlock * uniqueExitBlock = llvm::BasicBlock::Create(SharedContext::get(), "exitswitch", &func, edges.begin()->second);
		parent->addBlockEntry(uniqueExitBlock);
    // FIXME
		llvm::PHINode * phi = llvm::PHINode::Create(intType, 0, "exitID", uniqueExitBlock);
 		std::vector<llvm::BasicBlock*> enumeratedExits;

		for (BlockPairVector::iterator edge = edges.begin(); edge != edges.end(); ++edge)
		{
			llvm::BasicBlock * from = edge->first;
			llvm::BasicBlock * to = edge->second;

			//only enumerate exits to the parent loop
			if (parent->contains(to))
			{
				llvm::TerminatorInst * term = from->getTerminator();

				for(uint succIdx = 0; succIdx < term->getNumSuccessors(); ++succIdx)
				{
					llvm::BasicBlock * target = term->getSuccessor(succIdx);

					if (target == to)
					{
						uint exitID = enumeratedExits.size();

						llvm::Constant * exitConstant = get_uint(exitID);
						phi->addIncoming(exitConstant, from);

						term->setSuccessor(succIdx, uniqueExitBlock);
						enumeratedExits.push_back(to);
					}
				}
			}
		}

		appendEnumerationSwitch(phi, enumeratedExits, uniqueExitBlock);

		return true;
	}

	ExitUnificationPass::ExitUnificationPass() : llvm::ModulePass(ID) {}


	void ExitUnificationPass::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{
		usage.addRequired<llvm::LoopInfo>();
	}


	bool ExitUnificationPass::runOnFunction(llvm::Function & func)
	{
		llvm::LoopInfo & loopInfo = getAnalysis<llvm::LoopInfo>(func);

		for(llvm::LoopInfo::iterator loop = loopInfo.begin(); loop != loopInfo.end(); ++loop)
		{
			unifyLoopExits(func, *loop);
		}

		return false;
	}

	bool ExitUnificationPass::runOnModule(llvm::Module & M)
	{
#ifdef DEBUG_PASSRUN
		std::cerr << "\n\n##### PASS: Loop Exit Enumeration #####\n\n";
#endif

		bool changed = false;
		for (llvm::Module::iterator func = M.begin(); func != M.end(); ++func)
		{
			if (! func->isDeclaration())
				changed |= runOnFunction(*func);
		}

#ifdef DEBUG
		std::cerr << "\n\n### Module after LEE #####\n\n";
		M.dump();
		verifyModule(M);
#endif

		return changed;
	}

	 const char * ExitUnificationPass::getPassName() const	{
		return "axtor - loop exit enumeration";
	}
}
