/*
 * Normalizer.cpp
 *
 *  Created on: Jul 26, 2010
 *      Author: simoll
 */

#include <axtor/pass/Normalizer.h>

#include <iostream>

#include <llvm/Pass.h>
#include <llvm/PassManager.h>
#include <llvm/PassManagers.h>

#include <axtor/util/llvmDebug.h>
#include <axtor/util/llvmDuplication.h>
#include <axtor/pass/CNSPass.h>
#include <axtor/util/llvmLoop.h>
#include <axtor/util/llvmShortCuts.h>

#include <utility>
#include <vector>

namespace axtor
{
	char Normalizer::ID = 0;

	Normalizer::Normalizer() :
		llvm::ModulePass(ID)
	{}

	Normalizer::~Normalizer()
	{}

	/*
	 * normalize all loops in th CFG
	 */
	bool Normalizer::normalizeSubLoopGraphs(BlockCopyTracker & tracker, llvm::Function & func, llvm::Loop * loop)
	{
		bool changed = false;

		typedef std::vector<llvm::Loop*> LoopVector;

		const LoopVector & childLoops = loop->getSubLoops();

		for(LoopVector::const_iterator itChildLoop = childLoops.begin(); itChildLoop != childLoops.end(); ++itChildLoop)
		{
			changed |= normalizeSubLoopGraphs(tracker, func, *itChildLoop);
		}

		changed |= normalizePostDomSubgraphs(tracker, func, loop->getHeader(), NULL, loop);

		return changed;
	}

	/*
	 * subdivide the graph at its post dominators
	 */
	bool Normalizer::normalizePostDomSubgraphs(BlockCopyTracker & tracker, llvm::Function & func, llvm::BasicBlock * entry, llvm::BasicBlock * barrierBlock, llvm::Loop * loopScope)
	{
		bool changed = false;

		if (entry == barrierBlock)
			return false;

		llvm::PostDominatorTree & postDomTree = getAnalysis<llvm::PostDominatorTree>(func);
		//llvm::LoopInfo & loopInfo = getAnalysis<llvm::LoopInfo>(func);

		{ //linear subdivision

	#ifdef DEBUG
			std::cerr << "\tnormalizePostDomSubgraph (from=" << entry->getName().str() << " to=" << (barrierBlock ? barrierBlock->getName().str() : "NULL") << ")\n";
	#endif
			assert(entry && "must not be NULL");
			bool changed = false;


			llvm::BasicBlock * graphHeader = entry;
			llvm::DomTreeNode * pdNode = NULL;
			llvm::BasicBlock * splitBlock = NULL;

			do
			{
				pdNode = postDomTree.getNode(graphHeader)->getIDom();
				splitBlock = pdNode ? pdNode->getBlock() : NULL;

#ifdef DEBUG
				std::cerr << "{\n";
#endif
				changed |= normalizeNode(tracker, func, graphHeader, splitBlock, loopScope);
#ifdef DEBUG
				std::cerr << "}\n";
#endif

				graphHeader = splitBlock;
			}
			while (graphHeader != NULL && graphHeader != barrierBlock && //postdom- barrier condition
					(!loopScope || loopScope->contains(graphHeader))); //loop scope condition
		}

		return changed;
	}

	/*
	 * considers nested loops as atomic nodes in the graph and thus returns their exits as successors
	 */
	EdgeVector Normalizer::getAbstractSuccessors(llvm::LoopInfo & loopInfo, llvm::BasicBlock * entry, llvm::Loop * loopScope)
	{
		EdgeVector successors;

		llvm::Loop * loop = getNestedLoop(loopInfo, loopScope, entry);
		/*llvm::BasicBlock * loopExit = loop ? loop->getExitBlock() : NULL;
		llvm::BasicBlock * loopHeader = loop ? loop->getHeader() : NULL;*/

		//loop case : treat loop exits as successors blocks
		if (loop != NULL) {

			BlockPairVector edges;
			getExitEdges(*loop, edges);

			for(BlockPairVector::iterator itExit = edges.begin(); itExit != edges.end(); ++itExit)
			{
				successors.push_back(*itExit);
			}
		} else {
			typedef llvm::GraphTraits<llvm::BasicBlock*> CFG;

			for (CFG::ChildIteratorType itChild = CFG::child_begin(entry); itChild != CFG::child_end(entry); ++itChild)
			{
				successors.push_back(std::make_pair(entry, *itChild));
			}
		}

		return successors;
	}

	/*
	 * normalize a node (basic block OR loop)
	 */
	//typedef llvm::SmallVector<llvm::BasicBlock*> SmallBlockVector;

	bool Normalizer::normalizeNode(BlockCopyTracker & tracker, llvm::Function & func, llvm::BasicBlock * entry, llvm::BasicBlock * entryPostDom, llvm::Loop * loopScope)
	{
		bool changed = false;
#ifdef DEBUG
		std::cerr << "\tnormalizeNode (from=" << entry->getName().str() << " to=" << (entryPostDom ? entryPostDom->getName().str() : "NULL") << ")\n";
#endif

		typedef llvm::GraphTraits<llvm::BasicBlock*> CFG;

		assert(entry && "must not be NULL");

		llvm::LoopInfo & loopInfo = getAnalysis<llvm::LoopInfo>(func);
		llvm::DominatorTree & domTree = getAnalysis<llvm::DominatorTree>(func);

		llvm::BasicBlock * loopHeader = loopScope ? loopScope->getHeader() : NULL;

		assert(! llvm::isa<llvm::SwitchInst>(entry->getTerminator()) && "kill switches first");

		EdgeVector successors = getAbstractSuccessors(loopInfo, entry, loopScope);

		//linear case : treat branch targets as successor blocks
		for (EdgeVector::iterator itEdge = successors.begin(); itEdge != successors.end(); ++itEdge)
		{
			//std::pair<const llvm::BasicBlock, const llvm::BasicBlock*> edge = *itChild;
      //FIXME
			//llvm::BasicBlock *pred = itEdge->first;
			llvm::BasicBlock *child = itEdge->second;

			llvm::Loop * childLoop = getNestedLoop(loopInfo, loopScope, child);

			if (
					child != entryPostDom &&
					child != loopHeader && //continue block
					!(loopScope && loopScope->contains(child)) //already anticipated loop exit
			) {
				if (
					!domTree.dominates(entry, child)
				) {
					changed = true;

					if (childLoop) {
						llvm::LPPassManager lpm;
            //FIXME
						//cloneLoopForBranch(tracker, lpm, this, loopInfo, childLoop, pred);
						normalizeNode(tracker, func, child, entryPostDom, loopScope);

					} else {
						llvm::BasicBlock * clone = cloneBlockForBranch(tracker, child, entry);

						normalizePostDomSubgraphs(tracker, func, clone, entryPostDom, loopScope);
					}
				} else {
					normalizePostDomSubgraphs(tracker, func, child, entryPostDom, loopScope);
				}
			}
		}

		return changed;
	}

	bool Normalizer::runOnModule(llvm::Module & M)
	{
#ifdef DEBUG_PASSRUN
		std::cerr << "\n\n##### PASS: Normalizer #####\n\n";
#endif
		TargetProvider & target = getAnalysis<TargetProvider>();
		BlockCopyTracker & tracker = target.getTracker();

		bool changed = false;
		for (llvm::Module::iterator func = M.begin();func != M.end(); ++func)
		{
			if (! func->isDeclaration())
			{
#ifdef DEBUG
				std::cerr << "\n\nnormalizing function=" << func->getName().str() << "\n";
#endif
				llvm::BasicBlock & entry = func->getEntryBlock();
				changed |= normalizePostDomSubgraphs(tracker, *func, &entry, NULL, NULL);
			}
		}

#ifdef DEBUG
		tracker.dump();
		verifyModule(M);
#endif

		return changed;
	}

	void Normalizer::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{
		usage.addRequired<llvm::LoopInfo>();
		usage.addRequired<llvm::PostDominatorTree>();
		usage.addRequired<llvm::DominatorTree>();
		usage.addRequired<TargetProvider>();
		//usage.addRequired<Regularizer>();

		//usage.addPreserved<Regularizer>();
		//usage.addPreserved<OpaqueTypeRenamer>();
	}
}
