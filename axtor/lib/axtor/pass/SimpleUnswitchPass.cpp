/*
 * SimpleUnswitchPass.cpp
 *
 *  Created on: 16.03.2011
 *      Author: gnarf
 */

#include <axtor/pass/SimpleUnswitchPass.h>

#include <llvm/Instruction.h>

#include <axtor/util/stringutil.h>

namespace axtor {


	char SimpleUnswitchPass::ID = 0;

	namespace {
		llvm::RegisterPass<SimpleUnswitchPass> __regUnswitchPass("unswitch", "axtor - simple unswitch pass",
							true /* mutates the CFG */,
							false /* transformation */);
	}

	SimpleUnswitchPass::SimpleUnswitchPass() :
		llvm::ModulePass(ID)
	{}

	SimpleUnswitchPass::~SimpleUnswitchPass() {}

	const char * SimpleUnswitchPass::getPassName() const
	{
		return "unswitch";
	}

	void SimpleUnswitchPass::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{}

	void SimpleUnswitchPass::processSwitch(llvm::Function * func, llvm::BasicBlock * switchBlock)
	{
		llvm::Module * mod = func->getParent();
		llvm::LLVMContext & context = mod->getContext();

		llvm::SwitchInst * switchInst = llvm::cast<llvm::SwitchInst>(switchBlock->getTerminator());
		llvm::Value * switchValue = switchInst->getCondition();

		llvm::BasicBlock * defaultDest = switchInst->getDefaultDest();
		llvm::BasicBlock * exitBlock = switchInst->getDefaultDest();

		//create IF-cascade (skip the default case)
		for(uint i = 1; i < switchInst->getNumCases(); ++i)
		{
			//llvm::Value * succVal =(llvm::Value*)switchInst->getSuccessorValue(i);
			llvm::Value * succVal = (llvm::Value*)switchInst->getOperand(i*2);
			llvm::BasicBlock * succBlock = switchInst->getSuccessor(i);

			{
				llvm::BasicBlock * caseBlock = llvm::BasicBlock::Create(context, "cascade" + str<uint>(i), func,  exitBlock);
				llvm::CmpInst * testInst = llvm::CmpInst::Create(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchValue, succVal, "caseCompare", caseBlock);
				llvm::BranchInst::Create(succBlock, exitBlock, testInst, caseBlock);

				//the last IF takes over the default branch of the switchBlock (replace in PHI-nodes)
				if (exitBlock == defaultDest) {
					for (llvm::BasicBlock::iterator itPhi = exitBlock->begin(); llvm::isa<llvm::PHINode>(itPhi); ++itPhi)
					{
						llvm::PHINode * phi = llvm::cast<llvm::PHINode>(itPhi);
						for(uint i = 0; i < phi->getNumIncomingValues(); i++)
						{
							if (phi->getIncomingBlock(i) == switchBlock) {
								phi->setIncomingBlock(i, caseBlock);
							}
						}
					}
				}

				exitBlock = caseBlock;
			}
		}

		//replace Switch-Instruction
		switchInst->eraseFromParent();
		llvm::BranchInst::Create(exitBlock, switchBlock);
	}

	bool SimpleUnswitchPass::runOnFunction(llvm::Function * func)
	{
		bool foundSwitch = false;
		for (llvm::Function::iterator block = func->begin(); block != func->end(); ++block)
		{
			llvm::TerminatorInst * termInst = block->getTerminator();
			if (llvm::isa<llvm::SwitchInst>(termInst)) {
				processSwitch(func, &*block);
				foundSwitch = true;
			}
		}
		return foundSwitch;
	}

	bool SimpleUnswitchPass::runOnModule(llvm::Module & M)
	{
		bool changed = false;

		for (llvm::Module::iterator func = M.begin(); func != M.end(); ++func)
		{
			changed |= runOnFunction(&*func);
		}

		return changed;
	}
}
