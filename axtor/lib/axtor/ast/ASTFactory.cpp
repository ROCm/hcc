/*
 * ASTFactory.cpp
 *
 *  Created on: 12.06.2010
 *      Author: gnarf
 */

#include <axtor/ast/ASTFactory.h>


namespace axtor {
namespace ast {

	FunctionNode * ASTFactory::createFunction(llvm::Function * _func, ControlNode * _body)
	{
		return new FunctionNode(_func, _body);
	}

	ControlNode * ASTFactory::createConditional(llvm::BasicBlock * block, ControlNode * _onTrue, ControlNode * _onFalse)
	{
		return new ConditionalNode(block, _onTrue, _onFalse);
	}

	ControlNode * ASTFactory::createBlock(llvm::BasicBlock * block)
	{
		return new BlockNode(block);
	}

	ControlNode * ASTFactory::createList(const ast::NodeVector & nodes)
	{
		return new ListNode(nodes);
	}

	ControlNode * ASTFactory::createInfiniteLoop(ControlNode * body)
	{
		return new LoopNode(body);
	}

	ControlNode * ASTFactory::createBreak(llvm::BasicBlock * block)
	{
		return new BreakNode(block);
	}

	ControlNode * ASTFactory::createBreak()
	{
		return new BreakNode(NULL);
	}

	ControlNode * ASTFactory::createContinue(llvm::BasicBlock * block)
	{
		return new ContinueNode(block);
	}

	ControlNode * ASTFactory::createContinue()
	{
		return new ContinueNode(NULL);
	}

	ControlNode * ASTFactory::createReturn(llvm::BasicBlock * block)
	{
		return new ReturnNode(block);
	}

	ControlNode * ASTFactory::createUnreachable(llvm::BasicBlock * block)
	{
		return new UnreachableNode(block);
	}
	}
}
