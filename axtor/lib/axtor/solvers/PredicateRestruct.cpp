/*
 * PredicateRestruct.cpp
 *
 *  Created on: 27 Feb 2012
 *      Author: v1smoll
 */

#include <axtor/solvers/PredicateRestruct.h>

#include <axtor/util/llvmConstant.h>
#include <axtor/util/llvmDebug.h>
#include <llvm/Support/CFG.h>
#include <axtor/util/llvmDomination.h>

namespace axtor {

PredicateRestruct PredicateRestruct::instance;

PredicateRestruct::PredicateRestruct() {}
PredicateRestruct::~PredicateRestruct() {}

bool PredicateRestruct::resolve(RegionVector & regions, llvm::BasicBlock * requiredExit, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock *& oExitBlock)
{
#ifdef DEBUG
		std::cerr << "&&&&& PredicateRestruct::Resolve(..):context ;\n";
		context.dump();
#endif

	llvm::LLVMContext & LLVMContext = SharedContext::get();
	llvm::DominatorTree & domTree = analysis.getDomTree();

	//split all exits until the regions satisfy the single exit node property
	BlockSet regularExits = context.getRegularExits();
	BlockSetPair exitInfo = computeExitSet(regions, analysis.getDomTree(), context.getAnticipatedExits());
	BlockSet exitSet = exitInfo.first;
	BlockSet usedAnticipated = exitInfo.second;

	if (context.exitBlock && !set_contains(regularExits, context.exitBlock)
		&& set_contains(usedAnticipated, context.exitBlock)) { //split up until the exit block, if it is not also covered by BREAK/CONTINUE
		exitSet.insert(context.exitBlock);
	}

	// early escape
	bool modifiedCFG = false;
	if (exitSet.size() == 1) {
		oExitBlock = *exitSet.begin();
	} else if (exitSet.empty()) {
		oExitBlock = context.exitBlock;
	} else {
		modifiedCFG = true;

		// we only modify branches from blocks in the target region
		BlockSet consideredPredBlocks;
		for (RegionVector::iterator itRegion = regions.begin(); itRegion != regions.end(); ++itRegion)
		{
			BlockSet blocksInRegion = computeDominatedRegion(domTree, itRegion->getHeader(), exitSet);
#ifdef DEBUG
			itRegion->dump(     );
			dumpBlockSet(blocksInRegion);
#endif

			consideredPredBlocks.insert(blocksInRegion.begin(), blocksInRegion.end());
		}

#ifdef DEBUG
		llvm::errs() << "&&&&& exitSet:";
		dumpBlockSet(exitSet);

		llvm::errs() << "&&&&& consideredPredBlocks:";
		dumpBlockSet(consideredPredBlocks);
#endif

		// compute the least common dominator of all common exit blocks
		llvm::DomTreeNode * criticalEntryNode = findImmediateDominator(domTree, exitSet);
		llvm::BasicBlock * criticalEntryBlock = criticalEntryNode->getBlock();

		// create a new fused block
		llvm::BasicBlock * fusedBlock = llvm::BasicBlock::Create(LLVMContext, "fused", analysis.getFunction());
		llvm::PHINode * indexPHI = llvm::PHINode::Create(llvm::Type::getInt32Ty(LLVMContext), 6, "index", fusedBlock);
		domTree.addNewBlock(fusedBlock, criticalEntryBlock);

		// index exiting blocks
		BlockVector indexedExits;
		indexedExits.reserve(exitSet.size());
		uint index = 0;

		BlockSet predecessorSet;

		for (BlockSet::iterator itExitBlock = exitSet.begin(); itExitBlock != exitSet.end(); ++itExitBlock, ++index) {

			llvm::BasicBlock * const exitBlock = *itExitBlock;
			indexedExits.push_back(exitBlock);

			// patch-up all predecessors
			llvm::pred_iterator itPredBlock, S, E;
			S = llvm::pred_begin(exitBlock);
			E = llvm::pred_end(exitBlock);

			ValueMap fixPHIMap;

			// modify branchInsts to exitNodes
			BlockSet localPredecessorSet;
			for (itPredBlock = S; itPredBlock != E;) {

				 //ignore branches from other exit blocks (if pred reaches this exit w/o usign other exit blocks)
				llvm::BasicBlock * predBlock = *itPredBlock++;
				if (
						set_contains(consideredPredBlocks, predBlock)
						/* && !set_contains(exitSet, predBlock) &&
						reaches(predBlock, exitBlock, getWithout(exitSet, exitBlock)) */
				) {
					localPredecessorSet.insert(predBlock);
					predecessorSet.insert(predBlock);

					llvm::TerminatorInst * termInst = predBlock->getTerminator();
					int succIdx = getSuccessorIndex(termInst, exitBlock);

					assert(succIdx > -1 && "block unused by predecessor?!");

					int blockIdx = indexPHI->getBasicBlockIndex(predBlock);

					// insert an intermediate block
					if (blockIdx > -1) {
						llvm::BasicBlock * branchBlock = llvm::BasicBlock::Create(LLVMContext, "detachedBranch", analysis.getFunction(),fusedBlock);
						llvm::BranchInst::Create(fusedBlock, branchBlock);
						termInst->setSuccessor(succIdx, branchBlock);
						// Maintain that all PHI-nodes after the fuse block
						// refer to direct predecessors of the fused block
						fixPHIMap[exitBlock] = branchBlock;
						indexPHI->addIncoming(get_int(index), branchBlock);

					} else {
						termInst->setSuccessor(succIdx, fusedBlock);
						indexPHI->addIncoming(get_int(index), predBlock);
					}
				}
			}

			// TODO : just move the PHIs if no branches remain
			// Create new receiving PHIs Move all PHIs to the fuseBlock
			for (llvm::BasicBlock::iterator itPHI = exitBlock->begin(); llvm::isa<llvm::PHINode>(itPHI); itPHI = exitBlock->begin())
			{
				llvm::PHINode * oldPHI = llvm::cast<llvm::PHINode>(itPHI);
				llvm::PHINode * newPHI = llvm::PHINode::Create(oldPHI->getType(), oldPHI->getNumIncomingValues(), "newPHI", indexPHI);

				// attach the new PHI
				oldPHI->addIncoming(newPHI, fusedBlock);

				// reconnect all incoming values from this exitBlock's predecessors to new PHI
				// and remove them from oldPHI
				for (BlockSet::iterator itMovedPred = localPredecessorSet.begin(); itMovedPred != localPredecessorSet.end(); ++itMovedPred)
				{
					llvm::BasicBlock * movedPred = *itMovedPred;
					int blockIdx = oldPHI->getBasicBlockIndex(movedPred);

					assert(blockIdx > -1 && "analysing former predecessors. Must be included in PHI");

					// move branch to new PHI
					llvm::Value * inVal = oldPHI->getIncomingValue(blockIdx);
					newPHI->addIncoming(inVal, movedPred);
					oldPHI->removeIncomingValue(blockIdx, false);
				}

				// rename the blocks as necessary (if a branch was detached to enable different PHI-evaluations for a predecessor block)
				LazyRemapInstruction(newPHI, fixPHIMap);

#ifdef DEBUG
				llvm::errs() << "&&&&& mapped PHIs:\n";
				oldPHI->dump();
				newPHI->dump();
#endif

			}
		}

		// Attach UndefValues for all additional predecessors of the PHIs
		for (llvm::BasicBlock::iterator itPHI = fusedBlock->begin(); llvm::isa<llvm::PHINode>(itPHI); ++itPHI)
		{
			llvm::PHINode * phi = llvm::cast<llvm::PHINode>(itPHI);
			llvm::UndefValue * undefVal = llvm::UndefValue::get(phi->getType());

			for (BlockSet::const_iterator itPred = predecessorSet.begin(); itPred != predecessorSet.end(); ++itPred) {
				llvm::BasicBlock * const pred = *itPred;
				int predIdx = phi->getBasicBlockIndex(pred);
				if (predIdx < 0) {
					phi->addIncoming(undefVal, pred);
				}
			}
		}

		// Re-attach the exitNodes to the fused Block (think switch)
		llvm::BasicBlock * lowestExitBlock = *indexedExits.begin();
		llvm::BasicBlock * altBlock = lowestExitBlock;
		BlockVector blockCascade(indexedExits.size() - 1);

		for (uint i = 1; i < indexedExits.size(); ++i)
		{
			llvm::BasicBlock * exitBlock = indexedExits[i];

			llvm::BasicBlock * caseBlock = 0;

			// put the fusedBlock at the top of the cascade
			if (i < indexedExits.size() - 1) {
				caseBlock = llvm::BasicBlock::Create(LLVMContext, "fuseCascade" + str<uint>(i), analysis.getFunction(),  exitBlock);
			} else {
				caseBlock = fusedBlock;
			}
			blockCascade[indexedExits.size() - i - 1] = caseBlock;

			llvm::CmpInst * testInst = llvm::CmpInst::Create(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, indexPHI, get_int(i), "caseCompare", caseBlock);
			llvm::BranchInst::Create(exitBlock, altBlock, testInst, caseBlock);

			// cascade
			altBlock = caseBlock;

			domTree.changeImmediateDominator(indexedExits[i], caseBlock);
#ifdef DEBUG
			llvm::errs() << "DOMTREE: " << caseBlock->getName() << " idoms " << indexedExits[i]->getName() << "\n";
#endif
		}

#ifdef DEBUG_VIEW_CFGS
		analysis.getFunction()->viewCFGOnly();
#endif

		// update the dom-tree for the cascade
		domTree.changeImmediateDominator(lowestExitBlock, blockCascade[0]);
#ifdef DEBUG
			llvm::errs() << "DOMTREE: " << blockCascade[0]->getName() << " idoms " << lowestExitBlock->getName() << "\n";
#endif

		for (
				BlockVector::iterator itBlock = blockCascade.begin();
				(itBlock != blockCascade.end()) && (itBlock + 1 != blockCascade.end());
				++itBlock)
		{
			llvm::BasicBlock * dominatingBlock = *itBlock;
			llvm::BasicBlock * dominatedBlock = *(itBlock + 1);

#ifdef DEBUG
			llvm::errs() << "DOMTREE: " << dominatingBlock->getName() << " idoms (new) " << dominatedBlock->getName() << "\n";
#endif
			domTree.addNewBlock(dominatedBlock, dominatingBlock);
		}

		// the fused block is the mandatory exit for all cases
		oExitBlock = fusedBlock;
	}


	//make external exit the anticipated exit for all child regions
	for (RegionVector::iterator itRegion = regions.begin(); itRegion != regions.end(); ++itRegion)
	{
		ExtractorRegion & region = *itRegion;
		region.context.exitBlock = oExitBlock;
	}

#ifdef DEBUG
	analysis.getDomTree().dump();
#endif

EXPENSIVE_TEST verifyModule(*analysis.getFunction()->getParent());

	return modifiedCFG;
}

PredicateRestruct * PredicateRestruct::getInstance()
{
	return &instance;
}

}


