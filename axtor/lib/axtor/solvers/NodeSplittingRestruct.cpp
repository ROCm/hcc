/*
 * NodeSplittingRestruct.cpp
 *
 *  Created on: 20.02.2011
 *      Author: gnarf
 */

#include <axtor/solvers/NodeSplittingRestruct.h>

#include <axtor/util/llvmDebug.h>
#include <llvm/Support/CFG.h>

namespace axtor{

NodeSplittingRestruct NodeSplittingRestruct::instance;

	BlockVector NodeSplittingRestruct::getAbstractSuccessors(llvm::BasicBlock * block, llvm::Loop * parentLoop, llvm::LoopInfo & loopInfo)
	{
		BlockVector result;
		assert(block);
		llvm::Loop * loop = loopInfo.getLoopFor(block);

		if (loop && loop != parentLoop && block == loop->getHeader()) {

			typedef llvm::SmallVector<llvm::BasicBlock*, 4> SmallBlockVector;
			SmallBlockVector exits;
			loop->getUniqueExitBlocks(exits);
			result.reserve(exits.size());
      
			for(SmallBlockVector::iterator it = exits.begin(); it != exits.end(); ++it)
			{
				result.push_back(*it);
			}

		} else {
			typedef llvm::GraphTraits<llvm::BasicBlock*> CFG;

			for (CFG::ChildIteratorType itChild = CFG::child_begin(block); itChild != CFG::child_end(block); ++itChild)
			{
				result.push_back(*itChild);
			}
		}

		return result;
	}

	void NodeSplittingRestruct::sortByReachability(BlockVector & vector, int start, int end, BlockSet regularExits)
	{
		// base cases
		if (end == start) {
			return;
		} else if (end - start == 1) {
			if (reaches(vector[end], vector[start], regularExits)) {
				std::swap(vector[end], vector[start]);
				return;
			}
		}

		llvm::BasicBlock * pivotBlock = vector[start];
#ifdef DEBUG
		std::cerr << "\t sortReach from=" << start << "; to=" << end << "\nbefore:";
		dumpBlockVector(vector);
#endif

		int upper = end;
		int idx = start + 1;
		while (idx < upper)
		{
			llvm::BasicBlock * block = vector[idx]; //overwrite the start block

			if (reaches(pivotBlock, block, regularExits)) { //push right
				std::swap(vector[idx], vector[upper]);
				upper--;
			} else { //swap left
				vector[idx - 1] = block;
				idx++;
			}
		}

		assert(upper == idx && "wrong stack indices after partitioning");
		int middle = idx - 1;
		vector[middle] = pivotBlock;

#ifdef DEBUG
		std::cerr << "after (lower=" << start << "; upper=" << end << "; middle=" << middle << ")";
		dumpBlockVector(vector);
#endif

		//recurse
		if (middle - start > 1) {
			sortByReachability(vector, start, middle, regularExits);
		}
		if (end - middle > 1) {
			sortByReachability(vector, middle + 1, end, regularExits);
		}
	}

	void NodeSplittingRestruct::sortByReachability(BlockVector & vector, BlockSet regularExits)
	{
		sortByReachability(vector, 0, vector.size() - 1, regularExits);
	}

	llvm::BasicBlock * NodeSplittingRestruct::getUnreachableBlock(BlockSet blocks, BlockSet anticipated)
	{
		for(BlockSet::iterator itCand = blocks.begin(); itCand != blocks.end(); ++itCand)
		{
			bool unreachable = true;

			for (BlockSet::iterator itOther = blocks.begin(); itOther != blocks.end(); ++itOther)
			{
				if (itOther != itCand)
				{
					if (reaches(*itOther, *itCand, anticipated)) {
						unreachable = false;
						break;
					}
				}
			}

			if (unreachable)
				return *itCand;
		}

		return 0;
	}

	NodeSplittingRestruct::NodeSplittingRestruct() {}

	NodeSplittingRestruct::~NodeSplittingRestruct() {}

	int NodeSplittingRestruct::findUniqueRegion(llvm::DominatorTree & domTree, RegionVector & regions, llvm::BasicBlock * block)
	{
		int domIdx = -1;
		int i = 0;

		for(RegionVector::iterator itRegion = regions.begin(); itRegion != regions.end(); ++itRegion, ++i)
		{
			if (itRegion->contains(domTree, block)) {
				if (domIdx == -1) {
					domIdx = i;
				} else {
					return -1;
				}
			}
		}

		return domIdx;
	}

	llvm::BasicBlock * NodeSplittingRestruct::splitAndRinseOnStack_perRegion(RegionVector & regions, BlockVector & stack, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock * mandatoryExit)
	{
#ifdef DEBUG
		std::cerr << "##### before splitting #####\n";
		dumpBlockVector(stack);
#endif

		BlockSet regularExits = context.getRegularExits();

		llvm::BasicBlock * top = *stack.begin();
		stack.erase(stack.begin());
		llvm::TerminatorInst * termInst = top->getTerminator();

		stack.reserve(stack.size() + termInst->getNumSuccessors());
#ifdef DEBUG
		std::cerr << "##### without top #####\n";
		dumpBlockVector(stack);
#endif

		BlockVector nextBlocks = getAbstractSuccessors(top, context.parentLoop, analysis.getLoopInfo());

		for(BlockVector::iterator itNext = nextBlocks.begin(); itNext != nextBlocks.end(); ++itNext)
		{
			llvm::BasicBlock * succ = *itNext;

			if (succ == mandatoryExit ||
				regularExits.find(succ) != regularExits.end()) //don't split the mandated exit block or any anticipated exit
			continue;

			bool inStack = (top == succ);
			for(uint i = 0; i < stack.size(); ++i) {
				inStack |= stack[i] == succ;
			}

			if (!inStack) {
				BlockVector::iterator itBeforeReached = stack.begin();
				for(BlockVector::iterator elem = stack.begin(); elem != stack.end(); elem++)
				{
					if (reaches(*elem, succ, regularExits)) {
						itBeforeReached = elem;
						itBeforeReached++;
					}
				}

				stack.insert(itBeforeReached, succ);
			}
		}
#ifdef DEBUG
	std::cerr << "##### after splitting #####\n";
	dumpBlockVector(stack);
#endif

		//build the predecessor set
		BlockSetVector blockSetVec (regions.size() + 1, BlockSet());

		llvm::pred_iterator itPred, S, E;
		S = llvm::pred_begin(top); E = llvm::pred_end(top);

		for (itPred = S; itPred != E; ++itPred) {
			int regIdx = findUniqueRegion(analysis.getDomTree(), regions, *itPred);

			//if regIdx == -1 then store it in the preserved set (index 0)
			blockSetVec[regIdx + 1].insert(*itPred);
		}

#ifdef DEBUG
		std::cerr << "branch chunks { \n";
		for (BlockSetVector::iterator itChunk = blockSetVec.begin(); itChunk != blockSetVec.end(); ++itChunk)
		{
			dumpBlockSet(*itChunk);
		}
		std::cerr << "}\n";
#endif


		// perform the actual split on the CFG
		llvm::Loop * loop = analysis.getLoopFor(top);
		if (!loop || loop == context.parentLoop) {
			splitNodeExt(top, blockSetVec, &analysis.getDomTree());
#ifdef DEBUG_VIEW_CFGS
			analysis.getFunction()->viewCFGOnly();
#endif
		} else {
			//splitLoop(analysis.getLoopInfo(), loop, analysis.pass, &analysis.domTree);
			assert(false && "not implemented");
		}

		return top;
	}


	llvm::BasicBlock * NodeSplittingRestruct::splitAndRinseOnStack(BlockVector & stack, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock * mandatoryExit)
	{

#ifdef DEBUG
		std::cerr << "##### before splitting #####\n";
		dumpBlockVector(stack);
#endif

		BlockSet regularExits = context.getRegularExits();

		llvm::BasicBlock * top = *stack.begin();
		stack.erase(stack.begin());
		llvm::TerminatorInst * termInst = top->getTerminator();

		stack.reserve(stack.size() + termInst->getNumSuccessors());
#ifdef DEBUG
		std::cerr << "##### without top #####\n";
		dumpBlockVector(stack);
#endif

		BlockVector nextBlocks = getAbstractSuccessors(top, context.parentLoop, analysis.getLoopInfo());

		for(BlockVector::iterator itNext = nextBlocks.begin(); itNext != nextBlocks.end(); ++itNext)
		{
			llvm::BasicBlock * succ = *itNext;

			if (succ == mandatoryExit ||
				regularExits.find(succ) != regularExits.end()) //don't split the mandated exit block or any anticipated exit
			continue;

			bool inStack = (top == succ);
			for(uint i = 0; i < stack.size(); ++i) {
				inStack |= stack[i] == succ;
			}

			if (!inStack) {
				BlockVector::iterator itBeforeReached = stack.begin();
				for(BlockVector::iterator elem = stack.begin(); elem != stack.end(); elem++)
				{
					if (reaches(*elem, succ, regularExits)) {
						itBeforeReached = elem;
						itBeforeReached++;
					}
				}

				stack.insert(itBeforeReached, succ);
			}
		}
#ifdef DEBUG
	std::cerr << "##### after splitting #####\n";
	dumpBlockVector(stack);
#endif


		// perform the actual split on the CFG
		llvm::Loop * loop = analysis.getLoopFor(top);
		if (!loop || loop == context.parentLoop) {
			splitNode(top, &analysis.getDomTree());
		} else {
			assert(false && "not implemented");
			//splitLoop(analysis.loopInfo, loop, analysis.pass, &analysis.domTree);
		}

		return top;
	}

	/*
	 * ignore
	 * 		@requiredExit
	 */
	bool NodeSplittingRestruct::resolve(RegionVector & regions, llvm::BasicBlock * requiredExit, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock* & oExitBlock)
	{
#ifdef DEBUG
		std::cerr << "&&&&& NodeSplittingRestruct::Resolve(..):context ;\n";
		context.dump();
#endif
		//split all exits until the regions satisfy the single exit node property
		BlockSet regularExits = context.getRegularExits();
		BlockSetPair exitInfo = computeExitSet(regions, analysis.getDomTree(), context.getAnticipatedExits());
		BlockSet exitSet = exitInfo.first;
		BlockSet usedAnticipated = exitInfo.second;

		if (context.exitBlock && !set_contains(regularExits, context.exitBlock)
			&& set_contains(usedAnticipated, context.exitBlock)) { //split up until the exit block, if it is not also covered by BREAK/CONTINUE
			exitSet.insert(context.exitBlock);
		}

#ifdef DEBUG
		std::cerr << "# NodeSplittingRestruct {\n";
		std::cerr << "# regular exits: \n";
		dumpBlockSet(regularExits);
		std::cerr << "# exit set: \n";
		dumpBlockSet(exitSet);
#endif

#ifdef NS_NAIVE
		while (exitSet.size() > 1) {
			llvm::BasicBlock * splitBlock = getUnreachableBlock(exitSet, regularExits);
			assert(splitBlock && "graph must be acyclic");

			llvm::Loop * loop = analysis.getLoopFor(splitBlock);

			if (loop == context.parentLoop) {
				splitNode(splitBlock);
			} else {
				//splitLoop(analysis.getLoopInfo(), loop, analysis.pass);
				assert(false && "not implemented");
			}

			exitSet = computeExitSet(regions, analysis.getDomTree(), regularExits);
		}

		//return the detected exit
		llvm::BasicBlock * exitBlock = NULL;
		if (exitSet.size() == 1) {
			exitBlock = *exitSet.begin();
		} else {
			exitBlock = context.exitBlock;
		}

#else
		bool modifiedCFG = false;
		if (exitSet.size() > 1) {

			if (requiredExit) { // enforce the required exit block
				if (context.exitBlock)
					assert(analysis.postDominates(context.exitBlock, requiredExit) && "required exit is out of range");

				exitSet.erase(requiredExit);
				BlockVector stack = toVector<llvm::BasicBlock*>(exitSet);
				sortByReachability(stack, regularExits);

				//split all blocks except the mandatory exit block
				while (stack.size() > 0) {
					splitAndRinseOnStack_perRegion(regions, stack, context, analysis, requiredExit);
				}

				modifiedCFG = true;
				oExitBlock = requiredExit;

			} else { //find an arbitrary exit block
				BlockVector stack = toVector<llvm::BasicBlock*>(exitSet);
				sortByReachability(stack, regularExits);

				while (stack.size() > 1) {
					splitAndRinseOnStack_perRegion(regions, stack, context, analysis, NULL);
				}
				modifiedCFG = true;
				oExitBlock = *stack.begin();
			}

		} else {
			modifiedCFG = false;
			if (exitSet.size() == 1){
				oExitBlock = *exitSet.begin();
			} else {
				oExitBlock = context.exitBlock;
			}
		}
#endif


#ifdef DEBUG
			std::cerr << "# NodeSplittingRestruct detectedExit = " << (oExitBlock ? oExitBlock->getName().str() :"null") << "\n";
#endif

		//make external exit the anticipated exit for all child regions
		for (RegionVector::iterator itRegion = regions.begin(); itRegion != regions.end(); ++itRegion)
		{
			ExtractorRegion & region = *itRegion;
			region.context.exitBlock = oExitBlock;
		}

		return modifiedCFG;
	}

	NodeSplittingRestruct * NodeSplittingRestruct::getInstance()
	{
		return &instance;
	}
}
