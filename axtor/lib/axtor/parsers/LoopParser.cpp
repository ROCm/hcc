#include <axtor/parsers/LoopParser.h>

#include <axtor/console/CompilerLog.h>
#include <axtor/ast/BasicNodes.h>

#include <axtor/util/stringutil.h>

namespace axtor {
	LoopParser LoopParser::instance;

	LoopParser::LoopBuilderSession::LoopBuilderSession(RegionVector regions, llvm::BasicBlock * entry, llvm::BasicBlock * requiredExit) :
		BuilderSession(regions, entry, requiredExit)
	{}

	std::string LoopParser::LoopBuilderSession::getName() const
	{
		return "LoopBuilderSession";
	}

	RestructuringProcedure * LoopParser::LoopBuilderSession::getSolver() const
	{
		return NULL; //loop restructuring is currently done with the loop exit enumeration pass
	}

	ast::ControlNode * LoopParser::LoopBuilderSession::build(ast::NodeMap children, llvm::BasicBlock * exitBlock)
	{
		assert(children.size() == 1 && "expected a single loop body");

		return new ast::LoopNode(children[getEntryBlock()]);
	}

	void LoopParser::LoopBuilderSession::dump() {}



	PrimitiveParser::BuilderSession * LoopParser::tryParse(llvm::BasicBlock * entry, ExtractorContext context, AnalysisStruct & analysis)
	{
		llvm::Loop * loop = analysis.getLoopFor(entry);

		if (!loop || loop == context.parentLoop)
			return 0;

		llvm::SmallVector<llvm::BasicBlock*, 16> exits;
		loop->getUniqueExitBlocks(exits);

		if (exits.size() > 1 && loop->getParentLoop())
		{
/*			for(int i = 0 ; i < exits.size() ; ++i)
			{
				std::cerr << exits[i]->getNameStr() << ",\n";
			}*/

			Log::fail(entry, "(loop exits == " + str<int>(exits.size()) + " > 1) Use the loop exit enumeration pass to normalize inner loops");
		}

		// move all destinations of a outer-most loop into its body
		llvm::BasicBlock * breakTarget = exits.size() == 1 ? *exits.begin() : 0;

		ExtractorContext bodyContext(context);
		bodyContext.parentLoop = loop;
		bodyContext.exitBlock = entry;
		bodyContext.continueBlock = entry;
		bodyContext.breakBlock = breakTarget;

		RegionVector regions;
		regions.push_back(ExtractorRegion(entry, bodyContext));

		return new LoopBuilderSession(regions, entry, breakTarget);
	}

	LoopParser * LoopParser::getInstance()
	{
		return &instance;
	}
}
