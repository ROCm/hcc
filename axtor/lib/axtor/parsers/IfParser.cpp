#include <axtor/parsers/IfParser.h>

#include <axtor/ast/BasicNodes.h>
#include <axtor/util/llvmShortCuts.h>

#include <axtor/solvers/NodeSplittingRestruct.h>
#include <axtor/solvers/PredicateRestruct.h>

namespace axtor {
	IfParser IfParser::instance;

	IfParser::IfBuilderSession::IfBuilderSession(RegionVector regions, llvm::BasicBlock * exitBlock, llvm::BasicBlock * entryBlock) :
		BuilderSession(regions, entryBlock, exitBlock)
	{}

	std::string IfParser::IfBuilderSession::getName() const
	{
		return "IfBuilderSession";
	}


	ast::ControlNode * IfParser::IfBuilderSession::build(ast::NodeMap children, llvm::BasicBlock * exitBlock)
	{
		//get successors (these might have changed after solving)
		llvm::BasicBlock * entry = getEntryBlock();
		llvm::TerminatorInst * termInst = entry->getTerminator();
		llvm::BasicBlock * onTrueBlock = termInst->getSuccessor(0);
		llvm::BasicBlock * onFalseBlock = termInst->getSuccessor(1);

		assert ((exitBlock == onTrueBlock || exitBlock == onFalseBlock) && "IF must have exit node as successor");
		assert( (children.size() == 1) && "IF #children == 1" );

		//fetch nested child node
		ast::ControlNode * onTrueNode = onTrueBlock != exitBlock ? children[onTrueBlock] : NULL;
		ast::ControlNode * onFalseNode = onFalseBlock != exitBlock ? children[onFalseBlock] : NULL;

		return new ast::ConditionalNode(getEntryBlock(), onTrueNode, onFalseNode);
	}

	RestructuringProcedure * IfParser::IfBuilderSession::getSolver() const
	{
		return NodeSplittingRestruct::getInstance();
	}

	void IfParser::IfBuilderSession::dump()
	{
		llvm::BasicBlock * entry = getEntryBlock();
		llvm::TerminatorInst * termInst = entry->getTerminator();
		llvm::BasicBlock * onTrueBlock = termInst->getSuccessor(0);
		llvm::BasicBlock * onFalseBlock = termInst->getSuccessor(1);

		std::cerr << "IfBuilderSession{"
				<< "\n\t" << (onTrueBlock ? onTrueBlock->getName().str() : "NULL")
				<< "\n\t" << (onFalseBlock ? onFalseBlock->getName().str() : "NULL")
		<<"}\n";
	}

	/*
	 * IF..ELSE builder
	 */
	IfParser::IfElseBuilderSession::IfElseBuilderSession(RegionVector regions, llvm::BasicBlock * exitBlock, llvm::BasicBlock * entryBlock) :
		BuilderSession(regions, entryBlock, exitBlock)
	{}

	std::string IfParser::IfElseBuilderSession::getName() const
	{
		return "IfElseBuilderSession";
	}

	ast::ControlNode * IfParser::IfElseBuilderSession::build(ast::NodeMap children, llvm::BasicBlock * exitBlock)
	{
		//get successors (these might have changed after solving)

		llvm::BasicBlock * entry = getEntryBlock();
		llvm::TerminatorInst * termInst = entry->getTerminator();
		llvm::BasicBlock * onTrueBlock = termInst->getSuccessor(0);
		llvm::BasicBlock * onFalseBlock = termInst->getSuccessor(1);

#ifdef DEBUG
		llvm::errs() << "IfElseBuilderSession::build(..) { \n"
				<< "entry       : " << entry->getName() << "\n"
				<< "onTrueBlock : " << onTrueBlock->getName() << "\n"
				<< "onFalseBlock : " << onFalseBlock->getName() << "\n";
#endif


#ifdef DEBUG
		if (exitBlock == onTrueBlock || exitBlock == onFalseBlock) {
			 assert(false && "IF..ELSE never exits on immediate successor");
		}
#else
		assert((exitBlock != onTrueBlock && exitBlock != onFalseBlock) && "IF..ELSE never exits on immediate successor");
#endif
		assert(children.size() == 2 && "IF..ELSE #children == 2");

		//fetch nested child node
		ast::ControlNode * onTrueNode = children[onTrueBlock];
		ast::ControlNode * onFalseNode = children[onFalseBlock];

		return new ast::ConditionalNode(getEntryBlock(), onTrueNode, onFalseNode);
	}

	RestructuringProcedure * IfParser::IfElseBuilderSession::getSolver() const
	{
		// TODO: find a heuristic to decide between the two
		//return NodeSplittingRestruct::getInstance();
		return PredicateRestruct::getInstance();
	}

	void IfParser::IfElseBuilderSession::dump()
	{}

	/*
	 * tries to set up a builder session for IF primitives and falls back to IF..ELSE
	 */
	PrimitiveParser::BuilderSession * IfParser::tryParse(llvm::BasicBlock * entry, ExtractorContext context, AnalysisStruct & analysis)
	{
		llvm::Loop * loop = analysis.getLoopFor(entry);

		//can't handle loops here (only if they are pulled in)
		if (loop && loop != context.parentLoop) {
#ifdef DEBUG
			std::cerr << "is leaving the loop! ignore..\n parent context:";
			context.dump();
			std::cerr << "this loop: " << (loop? "" : "none"); if (loop) loop->dump();
#endif
			return NULL;
		}

		BlockSet regularExits = context.getRegularExits();

		llvm::TerminatorInst * termInst = entry->getTerminator();
		llvm::BranchInst * branchInst = llvm::cast<llvm::BranchInst>(termInst);
		llvm::BasicBlock * consBlock  = branchInst->getSuccessor(0);
		llvm::BasicBlock * altBlock  = branchInst->getSuccessor(1);

#ifdef DEBUG
		std::cerr << "\tconsBlock=" << (consBlock ? consBlock->getName().str() : "NULL") << "\n";
		std::cerr << "\t altBlock=" << (altBlock ? altBlock->getName().str() : "NULL") << "\n";
#endif

		BlockSetPair consExitInfo = computeDominanceFrontierExt(consBlock, analysis.getDomTree(), regularExits);
		BlockSet consExits = consExitInfo.first;
		BlockSetPair altExitInfo = computeDominanceFrontierExt(altBlock, analysis.getDomTree(), regularExits);
		BlockSet altExits = altExitInfo.first;

#ifdef DEBUG
		std::cerr << "\t\tnum cons exits :" << consExits.size() <<"\n";
		std::cerr << "\t\tnum alt exits :" << altExits.size() <<"\n";
#endif

		//a branch rejoins control-flow with the other, if the other is in its exit set, (if the branches exits at all)

		bool trivialCons = consBlock == context.exitBlock;
		bool trivialAlt = altBlock == context.exitBlock;

		// cons block properties
		bool consRegionJoinsAlt = consExits.find(altBlock) != consExits.end();
		bool altReachesConsExit = false;

		for (BlockSet::const_iterator itExit = consExits.begin(); itExit != consExits.end(); ++itExit)
		{
			if (*itExit != altBlock)
				altReachesConsExit |= reaches(altBlock, *itExit, regularExits);
		}

		// alt block properties
		bool altRegionJoinsCons = altExits.find(consBlock) != altExits.end();
		bool consReachesAltExit = false;

		for (BlockSet::const_iterator itExit = altExits.begin(); itExit != altExits.end(); ++itExit)
		{
			if (*itExit != consBlock)
				consReachesAltExit |= reaches(consBlock, *itExit, regularExits);
		}

		//FIX ME: what if cons reaches alt, but alt reaches another cons exit (!)

		//this IF is partial, if cons/alt don't exit the primitive and join control-flow
		bool partialIf =
				trivialCons || trivialAlt // terminating branch
				|| (consRegionJoinsAlt && !altReachesConsExit) // cons finishes on alt
				|| (altRegionJoinsCons && !consReachesAltExit); // alt finished on cons

		bool completeIf = !partialIf;

#ifdef DEBUG
		if (consRegionJoinsAlt)
			std::cerr << "\t\tcons joins alt. cons= " << consBlock->getName().str() << "\n";

		if (altRegionJoinsCons)
			std::cerr << "\t\talt joins cons. alt = " << altBlock->getName().str() << "\n";
#endif

		//### IF ###
		if (! completeIf)
		{
		/*	bool hasCons =
				(trivialAlt && !trivialCons) || //trivial case
				(!trivialAlt && !trivialCons) && consRegionJoinsAlt; //non-trivial, cons exits on alt branch
		 */

			llvm::BasicBlock * exitBlock = NULL;
			RegionVector regions;

			//in trivial cases, just recurse into that region
			if (trivialCons) {
				regions.push_back(ExtractorRegion(altBlock, context));

			} else if (trivialAlt) {
				regions.push_back(ExtractorRegion(consBlock, context));

			} else if (altRegionJoinsCons) {
				assert(!consRegionJoinsAlt && "CFG is acyclic in non-trivial cases");
				exitBlock = consBlock;

				ExtractorContext altContext(context);
				altContext.exitBlock = consBlock;
				regions.push_back(ExtractorRegion(altBlock, altContext));

			} else if (consRegionJoinsAlt) { // (freeAlt)
				assert(!altRegionJoinsCons && "CFG is acyclic in non-trivial cases");
				exitBlock = altBlock;

				ExtractorContext consContext(context);
				consContext.exitBlock = altBlock;
				regions.push_back(ExtractorRegion(consBlock, consContext));
			}

			return new IfBuilderSession(regions, exitBlock, entry);

		//### IF..ELSE ###
		} else {
			RegionVector regions;
			ExtractorContext consContext(context);
			consContext.exitBlock = NULL;

			ExtractorContext altContext(context);
			altContext.exitBlock = NULL;

			regions.push_back(ExtractorRegion(consBlock, consContext));
			regions.push_back(ExtractorRegion(altBlock, altContext));

			return new IfElseBuilderSession(regions, NULL, entry);
		}
	}

	IfParser * IfParser::getInstance()
	{
		return &instance;
	}
}
