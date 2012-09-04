/*
 * Serializer.cpp
 *
 *  Created on: 12.06.2010
 *      Author: gnarf
 */

#include <axtor/pass/Serializer.h>

#include <axtor/pass/TargetProvider.h>

#include <axtor/util/llvmDebug.h>
#include <axtor/util/ResourceGuard.h>

namespace axtor
{
	llvm::RegisterPass<Serializer> __regSerializer("serialize", "AST serialization pass", true, true);

	char Serializer::ID = 0;

	Serializer::Serializer() :
		llvm::ModulePass(ID)
	{}


	/*
	 * translates the instruction list contained in @bb
	 */
	void Serializer::writeBlockInstructions(SyntaxWriter * writer, llvm::BasicBlock * bb, IdentifierScope & identifiers)
	{
#ifdef DEBUG
		std::cerr << "### TRANSLATING BB : " << bb->getName().str() << std::endl;
#endif
		assert(bb && " must not be 0");
		llvm::BasicBlock::iterator it = bb->begin();

		assert(it != bb->end() && "encountered empty bb");

		for(;! llvm::isa<llvm::TerminatorInst>(it);++it)
		{
			llvm::Instruction * inst = it;

			if (llvm::isa<llvm::GetElementPtrInst>(inst))
			{
				continue;
			}

			const VariableDesc * desc = identifiers.lookUp(inst);

			if (! desc) {
				if (! isType(inst, llvm::Type::VoidTyID)) {
					inst->dump();
					inst->getType()->dump();
					assert (false && "found undeclared variable");
				}
				writer->writeInstruction(0, inst, identifiers);
			} else {
				writer->writeInstruction(desc, inst, identifiers);
			}
		};
	}

	void Serializer::processBranch(SyntaxWriter * writer, llvm::BasicBlock * source, llvm::BasicBlock * target, IdentifierScope & locals)
	{
#ifdef DEBUG
		std::cerr << "### processing branches from " << source->getName().str() << " to " << target->getName().str() << "\n";
#endif
		for(llvm::BasicBlock::iterator inst = target->begin(); llvm::isa<llvm::PHINode>(inst); ++inst)
		{
			llvm::PHINode * phi = llvm::cast<llvm::PHINode>(inst);
			llvm::Value * val = phi->getIncomingValueForBlock(source);

			const VariableDesc * destVal = locals.lookUp(phi);

			assert(destVal && "does not have a designator");
			std::string destVar = destVal->name + "_in";
			writer->writeAssignRaw(destVar, val, locals);
		}
	}


	/*
	 * create variable bindings for function arguments
	 */
	void Serializer::createArgumentDeclarations(llvm::Function * func, ConstVariableMap & declares, std::set<llvm::Value*> & parameters)
	{
		const llvm::FunctionType * type = getFunctionType(func);
		ArgList & argList = func->getArgumentList();

		ArgList::iterator arg;
		uint i;

		for(i = 0, arg = argList.begin();
			i < type->getNumParams() && arg != argList.end();
			++i, ++arg)
		{
			//const llvm::Type * argType = type->getParamType(i);

			std::string argName = arg->getName();

			assert(llvm::cast<llvm::Argument>(arg) != 0 && "mapped zero value argument");

			declares[arg] = VariableDesc(arg, argName);
			parameters.insert(arg);
		}
	}

	llvm::BasicBlock * Serializer::writeNode(AxtorBackend & backend, SyntaxWriter * writer, llvm::BasicBlock * previousBlock, llvm::BasicBlock * exitBlock, ast::ControlNode * node, IdentifierScope & locals, llvm::BasicBlock * loopHeader, llvm::BasicBlock * loopExit)
	{
		typedef ast::ControlNode::NodeVector NodeVector;

		llvm::BasicBlock * block = node->getBlock();
#ifdef DEBUG
		llvm::errs() << "processing " << (block ? block->getName() : "none") << " exit : " << (exitBlock ? exitBlock->getName() : "null") << "\n";
#endif



		{
			switch (node->getType())
			{
				case ast::ControlNode::IF: {
					writeBlockInstructions(writer, block, locals);

					ast::ConditionalNode * condNode = reinterpret_cast<ast::ConditionalNode*>(node);
					llvm::Value * condition = condNode->getCondition();

					//determine properties
					bool hasElse = condNode->getOnTrue() && condNode->getOnFalse();
					bool negateCondition = !hasElse && !condNode->getOnTrue() && condNode->getOnFalse();

					ast::ControlNode * consNode;
					ast::ControlNode * altNode;
					llvm::BasicBlock * consBlock;
					llvm::BasicBlock * altBlock;
					if (negateCondition)
					{
						consNode = condNode->getOnFalse();
						altNode = condNode->getOnTrue();
						consBlock = condNode->getOnFalseBlock();
						altBlock = condNode->getOnTrueBlock();
					} else {
						consNode = condNode->getOnTrue();
						altNode = condNode->getOnFalse();
						consBlock = condNode->getOnTrueBlock();
						altBlock = condNode->getOnFalseBlock();
					}

					//write down
					//## consequence ##
					writer->writeIf(condition, negateCondition, locals);

					SyntaxWriter * consWriter = backend.createBlockWriter(writer);
						processBranch(consWriter, block, consBlock, locals);
						writeNode(backend, consWriter, block, exitBlock, consNode, locals, loopHeader, loopExit);
					delete consWriter;

					//## alternative ##
					if (hasElse) { //real alternative case
						writer->writeElse();
						SyntaxWriter * altWriter = backend.createBlockWriter(writer);

						processBranch(altWriter, block, altBlock, locals);

						llvm::BasicBlock * unprocessed = writeNode(backend, altWriter, block, exitBlock, altNode, locals, loopHeader, loopExit);
						if (unprocessed)
						{
							processBranch(altWriter, unprocessed, exitBlock, locals);
						}

						delete altWriter;

					} else if (containsPHINodes(exitBlock)) { // alternative only contains PHI-assignments
						writer->writeElse();
						SyntaxWriter * altWriter = backend.createBlockWriter(writer);

						processBranch(altWriter, block, exitBlock, locals);

						delete altWriter;
					}

					return 0;
				}

				case ast::ControlNode::LOOP: {
					writer->writeInfiniteLoopBegin();
						SyntaxWriter * bodyWriter = backend.createBlockWriter(writer);
						writeNode(backend, bodyWriter, previousBlock, node->getEntryBlock(), node->getNode(ast::LoopNode::BODY), locals, node->getEntryBlock(), exitBlock);
						delete bodyWriter;
					writer->writeInfiniteLoopEnd();
					return 0;
				}

				case ast::ControlNode::BREAK: {
					if (block) {
						writeBlockInstructions(writer, block, locals);
						processBranch(writer, block, loopExit, locals);
					} else {
						processBranch(writer, previousBlock, loopExit, locals);
					}
					writer->writeLoopBreak();

					return 0;
				}

				case ast::ControlNode::CONTINUE: {
					assert(loopHeader && "continue outside of loop");

					if (block) {
						writeBlockInstructions(writer, block, locals);
						processBranch(writer, block, loopHeader, locals);
					} else {
						processBranch(writer, previousBlock, loopHeader, locals);
					}
					writer->writeLoopContinue();

					return 0;
				}

				case ast::ControlNode::LIST: {
					llvm::BasicBlock * childPrevBlock = previousBlock;

					for(NodeVector::const_iterator itNode = node->begin(); itNode != node->end() ; ++itNode)
					{
						ast::ControlNode * child = *itNode;

						llvm::BasicBlock * nextExitBlock;

						// last block case
						if (((itNode + 1) == node->end())) {
							nextExitBlock = exitBlock;

						// last block before empty block case (CONTINUE, BREAK)
						} else if (
								(itNode + 2) == node->end() &&
								(*(itNode + 1))->getEntryBlock() == 0
						) {
							nextExitBlock = exitBlock;

						// inner node case
						} else {
							nextExitBlock =  (*(itNode + 1))->getEntryBlock();
						}

						childPrevBlock = writeNode(backend, writer, childPrevBlock, nextExitBlock, child, locals, loopHeader, loopExit);
					}

					return childPrevBlock;
				}

				case ast::ControlNode::BLOCK: {
					writeBlockInstructions(writer, block, locals);
					processBranch(writer, block, exitBlock, locals);

					return block;
				}

				case ast::ControlNode::RETURN: {
					ast::ReturnNode * returnNode = reinterpret_cast<ast::ReturnNode*>(node);

					writeBlockInstructions(writer, block, locals);
					writer->writeReturnInst(returnNode->getReturn(), locals);
					return 0;
				}

				case ast::ControlNode::UNREACHABLE: {
					writeBlockInstructions(writer, block, locals);
					return 0;
				}

				default:
					Log::fail(block, "unsupported node type");
			}
		}
	}

	void Serializer::runOnFunction(AxtorBackend & backend, SyntaxWriter * modWriter, IdentifierScope & globals, ast::FunctionNode * funcNode)
	{
		ValueSet parameters;

		IdentifierScope locals(&globals);
		llvm::Function * func = funcNode->getFunction();

		//### bind function parameters ###
		createArgumentDeclarations(func, locals.identifiers, parameters);

		//### create local identifiers ###
		for(llvm::Function::iterator itBlock = func->begin(); itBlock != func->end(); ++itBlock) {
			for(llvm::BasicBlock::iterator itInst = itBlock->begin(); itInst != itBlock->end(); ++itInst)
			{
				llvm::Instruction * inst = itInst;
				VariableDesc desc(inst, inst->getName());

				if (! isType(inst, llvm::Type::VoidTyID) &&
						! llvm::isa<llvm::GetElementPtrInst>(inst)) //no pointer support: only indirectly required by value useds
				{

				assert(inst != 0 && "mapped zero value");

				//dont overwrite preceeding mappings by PHI-Nodes
				if (locals.lookUp(inst) == 0)
					locals.bind(inst, desc);
				}
			}
		}


		//print out a list of declarations (single occurrence,not a parameter)
		//### write function body ###
		{
			SyntaxWriter * funcWriter = backend.createFunctionWriter(modWriter, func);
			ResourceGuard<SyntaxWriter> __guardFuncWriter(funcWriter);

			funcWriter->writeFunctionHeader(func, &locals);
			{
				SyntaxWriter * bodyWriter = backend.createBlockWriter(funcWriter);
				ResourceGuard<SyntaxWriter> __guardBodyWriter(bodyWriter);

				bodyWriter->writeFunctionPrologue(func, locals);

				//### translate instructions ###
#ifdef DEBUG
				std::cerr <<  "##### translating instructions of func " << func->getName().str() << "\n";
#endif

				ast::ControlNode * body = funcNode->getEntry();
				writeNode(backend, bodyWriter,0, 0, body, locals, 0, 0);
			}
		}
	}

	bool Serializer::runOnModule(llvm::Module & M)
	{
#ifdef DEBUG_PASSRUN
		std::cerr << "\n\n##### PASS: Serializer #####\n\n";
#endif

		TargetProvider & target = getAnalysis<TargetProvider>();
		AxtorBackend & backend = target.getBackend();
		ModuleInfo & modInfo = target.getModuleInfo();

		//not our module -> return
		if (! modInfo.isTargetModule(&M)) {
			return false;
		}

#ifdef DEBUG
		modInfo.dump();
		modInfo.verifyModule();
#endif

		IdentifierScope globalScope = modInfo.createGlobalBindings();

		SyntaxWriter * modWriter = backend.createModuleWriter(modInfo, globalScope);

		const ast::ASTMap & ASTs = getAnalysis<RestructuringPass>().getASTs();

		for(llvm::Module::iterator func = M.begin(); func != M.end(); ++func)
		{
			if (! func->isDeclaration())
			{
				ast::FunctionNode * funcNode = ASTs.at(func);
#ifdef DEBUG
				funcNode->dump();
#endif
				runOnFunction(backend, modWriter, globalScope, funcNode);
			}
		}

		delete modWriter;

		return false;
	}

	void Serializer::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{
		//usage.addRequired<OpaqueTypeRenamer>();
		usage.addRequired<RestructuringPass>();
		usage.addRequired<TargetProvider>();

		usage.setPreservesAll();
	}

	const char * Serializer::getPassName() const
	{
		return "axtor - serializer";
	}
}
