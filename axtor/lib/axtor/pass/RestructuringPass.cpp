/*
 * RestructuringPass.cpp
 *
 *  Created on: 20.02.2011
 *      Author: gnarf
 */


#include <axtor/pass/RestructuringPass.h>

#include <axtor/solvers/RestructuringProcedure.h>
#include <axtor/console/CompilerLog.h>

#include <axtor/util/llvmDebug.h>

namespace axtor {

	llvm::RegisterPass<RestructuringPass> __regRestructPass(
			"restruct", "axtor - the new modular restructuring ast-^extraction pass",
			false, /* mutates the CFG */
			false); /* transformation */

	char RestructuringPass::ID = 0;

	RestructuringPass::RestructuringPass() :
		llvm::ModulePass(ID)
	{}

	ast::ControlNode * RestructuringPass::processRegion(bool enteredLoop, const ExtractorRegion & region, AnalysisStruct & analysis, BlockSet & visited)
	{
		ast::NodeVector nodes;
		const ExtractorContext & context = region.context;

	#ifdef DEBUG
		llvm::errs() << "processRegion : " << region.getHeader()->getName().str() << " until " << (context.exitBlock ? context.exitBlock->getName().str()  : "NULL") << "\n";
		llvm::errs() << "{\n";
	#endif
		//llvm::BasicBlock * exitBlock = region.context.exitBlock;


		llvm::BasicBlock * exitBlock = context.exitBlock;
		llvm::BasicBlock * block = region.getHeader();
		ast::ControlNode * childNode;

		while (block != NULL)
		{
			llvm::BasicBlock * next = processBasicBlock(enteredLoop, block, context, analysis, visited, childNode);
			enteredLoop = false; //following nodes are listed behind in a sequence
			nodes.push_back(childNode);
			block = next;
/*#ifdef DEBUG
			llvm::errs() << "returned " << (block ? block->getName().str() : "NULL") << "\n";
#endif*/
			if (block == exitBlock)
				break;
		}

	#ifdef DEBUG
		llvm::errs() << "}\n";
	#endif

		if (nodes.size() == 1) {
			return childNode;
		} else {
			return ast::ASTFactory::createList(nodes);
		}
	}

	void RestructuringPass::rebuildAnalysisStruct(llvm::Function & func, AnalysisStruct & analysis)
	{
		llvm::LoopInfo & funcLoopInfo = getAnalysis<llvm::LoopInfo>(func);
		llvm::DominatorTree & funcDomTree = getAnalysis<llvm::DominatorTree>(func);
		llvm::PostDominatorTree & funcPostDomTree = getAnalysis<llvm::PostDominatorTree>(func);
		analysis = AnalysisStruct(*this, func, funcLoopInfo, funcDomTree, funcPostDomTree);
	}


	llvm::BasicBlock * RestructuringPass::processBasicBlock(bool enteredLoop, llvm::BasicBlock * bb, const ExtractorContext & context, AnalysisStruct & analysis, BlockSet & visited, ast::ControlNode *& oNode)
	{

		assert(bb);
#ifdef DEBUG
		llvm::errs() << "\tprocessBlock : " << bb->getName() << "\n";
#endif

		/*
		 * check special branch cases (zero payload nodes)
		 */

		if (!enteredLoop) {
			if (bb == context.continueBlock) {
	#ifdef DEBUG
				llvm::errs() << "\t\t(continue)" << "\n";
	#endif
				oNode = new ast::ContinueNode(NULL); //preserve AST semantics
				return NULL;
			} else if (bb == context.breakBlock) {
	#ifdef DEBUG
				llvm::errs() << "\t\t(break)" << "\n";
	#endif
				oNode = new ast::BreakNode(NULL); //preserve AST-semantics
				return NULL;
			}
		}


		/*
		 * check special blocks (with payload)
		 */
		llvm::TerminatorInst * termInst = bb->getTerminator();
		if (termInst->getNumSuccessors() == 0) {
			if (llvm::isa<llvm::ReturnInst>(termInst)) {
#ifdef DEBUG
				llvm::errs() << "\t\t(return)" << "\n";
#endif
				oNode = new ast::ReturnNode(bb);
				return NULL;

			} else if (llvm::isa<llvm::UnreachableInst>(termInst)) {
#ifdef DEBUG
				llvm::errs() << "\t\t(unreachable)" << "\n";
#endif
				oNode = new ast::UnreachableNode(bb);
				return NULL;
			}
		}



		/*
		 * check loops (this will abort the program if it encounters unstructured loops)
		 */
		PrimitiveParser::BuilderSession * loopBuilder = parsers.loopParser->tryParse(bb, context, analysis);
		if (loopBuilder)
		{
#ifdef DEBUG
			llvm::errs() << "\t\t(loop)" << "\n";
#endif
			ast::NodeMap children;
			const ExtractorRegion & region = loopBuilder->getRegion(0);
			assert(region.verify(analysis.getDomTree()) && "loop body uses non-anticipated exits");

#ifdef DEBUG
			llvm::errs() << "\tloop body";
			region.dump("\t\t");
#endif

			children[region.getHeader()] = processRegion(true, region, analysis, visited);
			oNode = loopBuilder->build(children, NULL);

			return loopBuilder->getRequiredExit();
		}


		/*
		 * check acyclic control primitives
		 */

		/*
		 * 1-way primitives
		 */
		if (termInst->getNumSuccessors() == 1)
		{
			llvm::BasicBlock * next = termInst->getSuccessor(0);

#ifdef DEBUG
			llvm::errs() << "\t\t(sequence)" << "\n";
#endif
			oNode = new ast::BlockNode(bb);
			return next;

		}


		/*
		 * 2-way primitive
		 */
		PrimitiveParser::BuilderSession * ifBuilder = parsers.ifParser->tryParse(bb, context, analysis);
		if (ifBuilder)
		{
	#ifdef DEBUG
		llvm::errs() << "\t\tis acyclic primitive (builder = " << ifBuilder->getName() << ")\n";
	#endif
			ast::NodeMap children;
			RestructuringProcedure * solver = ifBuilder->getSolver();
			assert(solver && "solvers mandatory for acyclic primitives");

			// resolve control-flow issues (single node exit property)
			llvm::BasicBlock * exitBlock = 0;
			while (solver->resolve(ifBuilder->getRegions(), ifBuilder->getRequiredExit(), context, analysis, exitBlock)) {
				ifBuilder = parsers.ifParser->tryParse(bb, context, analysis);
			}

			RegionVector & regions = ifBuilder->getRegions();

			// recurse over all child regions
			for (RegionVector::iterator itRegion = regions.begin(); itRegion != regions.end(); ++itRegion)
			{
				ExtractorRegion & region = *itRegion;
#ifdef DEBUG
				region.dump("\t");
#endif

				EXPENSIVE_TEST if (! region.verify(analysis.getDomTree())) {
					analysis.getLoopInfo().dump();
					analysis.getDomTree().dump();
#ifdef DEBUG_VIEW_CFGS
					llvm::errs() << "VIEWCFG: with invalid regions at header " << (region.getHeader() ? region.getHeader()->getName() : "0") << "\n";
					bb->getParent()->viewCFGOnly();
#endif
					Log::fail(bb->getParent(), "region invalid after solving");
					assert(false && "region invalid after solving");
				}

				ast::ControlNode * childNode = processRegion(false, region, analysis, visited);

				children[region.getHeader()] = childNode;
			}

#ifdef DEBUG
			llvm::errs() << "checking on the session . . \n";
			ifBuilder->dump();
#endif

			// build node
			oNode = ifBuilder->build(children, exitBlock);

			if (exitBlock == context.exitBlock)
				return 0;
			else
				return exitBlock;
		}

		Log::fail(termInst, "unsupported control-flow detected");
	}

	ast::FunctionNode * RestructuringPass::runOnFunction(llvm::Function & func)
	{
		llvm::LoopInfo & funcLoopInfo = getAnalysis<llvm::LoopInfo>(func);
		llvm::DominatorTree & funcDomTree = getAnalysis<llvm::DominatorTree>(func);
		llvm::PostDominatorTree & funcPostDomTree = getAnalysis<llvm::PostDominatorTree>(func);
		AnalysisStruct analysis(*this, func, funcLoopInfo, funcDomTree, funcPostDomTree);

		BlockSet visited;

#ifdef DEBUG_VIEW_CFGS
		llvm::errs() << "VIEW_CFG: original CFG\n";
		func.viewCFGOnly();
#endif

#ifdef DEBUG
		llvm::errs() << "Restruct: begin dump \n";
		func.dump();
		llvm::errs() << "Restruct: end dump\n";
		llvm::errs() << "### LoopInfo ###\n";
		funcLoopInfo.dump();
		llvm::errs() << "### DomTree ###\n";
		funcDomTree.dump();
		llvm::errs() << "### PostDomTree ###\n";
		funcPostDomTree.dump();
		// func.viewCFGOnly();
#endif


		llvm::BasicBlock * bb = &( func.getEntryBlock() );
		ExtractorContext context;

		ExtractorRegion bodyRegion(bb, context);
		ast::ControlNode * body = processRegion(false, bodyRegion, analysis, visited);
		return ast::ASTFactory::createFunction(&func, body);
	}

	void  RestructuringPass::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{
		usage.addRequired<llvm::PostDominatorTree>();
		usage.addRequired<llvm::DominatorTree>();
		usage.addRequired<llvm::LoopInfo>();
	}

	bool RestructuringPass::runOnModule(llvm::Module & M)
	{
#ifdef DEBUG_PASSRUN
		llvm::errs() << "\n\n##### PASS: Restructuring Pass #####\n\n";
		verifyModule(M);
#endif
		for (llvm::Module::iterator func = M.begin(); func != M.end(); ++func)
		{
			if (!func->isDeclaration()) {
				astMap[(llvm::Function*) func] = runOnFunction(*func);
			}
		}

		return true;
	}

	const char *  RestructuringPass::getPassName() const
	{
		return "axtor - modular AST extraction pass";
	}


	void RestructuringPass::releaseMemory()
	{
		for (ast::ASTMap::iterator pair = astMap.begin(); pair != astMap.end(); ++pair)
		{
			delete pair->second;
		}

		astMap.clear();
	}

	const ast::ASTMap & RestructuringPass::getASTs() const
	{
		return astMap;
	}

}

