/*
 * BasicNodes.cpp
 *
 *  Created on: 12.06.2010
 *      Author: gnarf
 */

#include <axtor/ast/BasicNodes.h>

/*
 * the extractor exclusively uses nodes described in this file to rebuild the AST. More sophisticated control flow elements are created by enhancing a pre-existing AST
 */
namespace axtor
{
	namespace ast
	{

		llvm::Function * FunctionNode::getFunction() const
		{
			return func;
		}

		ControlNode * FunctionNode::getEntry() const
		{
			return entryNode;
		}

		FunctionNode::FunctionNode(llvm::Function * _func, ControlNode * _entryNode) :
			func(_func),
			entryNode(_entryNode)
		{}

		void FunctionNode::dump() const
		{
			std::cerr << "Function " << func->getName().str() << " :\n";
			std::cerr << "{\n";
				getEntry()->dump();
			std::cerr << "}\n";
		}


		ConditionalNode::ConditionalNode(llvm::BasicBlock * _block, ControlNode * onTrueNode, ControlNode * onFalseNode) :
			ControlNode(IF, _block, 2)
		{
			setNode(ON_TRUE, onTrueNode);
			setNode(ON_FALSE, onFalseNode);
		}

		llvm::Value * ConditionalNode::getCondition() const
		{
			llvm::TerminatorInst * term = getBlock()->getTerminator();
			return term->getOperand(0);
		}

		ControlNode * ConditionalNode::getOnTrue() const
		{
			return getNode(ON_TRUE);
		}

		ControlNode * ConditionalNode::getOnFalse() const
		{
			return getNode(ON_FALSE);
		}

		llvm::BasicBlock * ConditionalNode::getOnTrueBlock() const
		{
			return getTerminator()->getSuccessor(ON_TRUE);
		}

		llvm::BasicBlock * ConditionalNode::getOnFalseBlock() const
		{
			return getTerminator()->getSuccessor(ON_FALSE);
		}

		std::string ConditionalNode::getTypeStr() const
		{
			return "IF";
		}

		/*
		 * Infinitely looping node
		 */
		LoopNode::LoopNode(ControlNode * _body) :
				ControlNode(LOOP, NULL, 1)
		{
			setNode(BODY, _body);
		}

		std::string LoopNode::getTypeStr() const
		{
			return "LOOP";
		}

		BreakNode::BreakNode(llvm::BasicBlock * _block) :
			ControlNode(BREAK, _block, 0)
		{}

		std::string BreakNode::getTypeStr() const
		{
			return "BREAK";
		}

		ContinueNode::ContinueNode(llvm::BasicBlock * _block) :
				ControlNode(CONTINUE, _block, 0)
		{}

		std::string ContinueNode::getTypeStr() const
		{
			return "CONTINUE";
		}

		ReturnNode::ReturnNode(llvm::BasicBlock * _block) :
				ControlNode(RETURN, _block, 0)
		{}

		llvm::ReturnInst * ReturnNode::getReturn() const
		{
			return llvm::cast<llvm::ReturnInst>(getTerminator());
		}

		std::string ReturnNode::getTypeStr() const
		{
			return "RETURN";
		}

		UnreachableNode::UnreachableNode(llvm::BasicBlock * _block) :
				ControlNode(UNREACHABLE, _block, 0)
		{}

		std::string UnreachableNode::getTypeStr() const
		{
			return "UNREACHABLE";
		}

		/*
		 * sequence of ast nodes
		 */
		ListNode::ListNode(const NodeVector & nodeVector) :
				ControlNode(LIST, NULL, nodeVector)
		{}

		std::string ListNode::getTypeStr() const
		{
			return "LIST";
		}

		BlockNode::BlockNode(llvm::BasicBlock * _block) :
				ControlNode(BLOCK, _block, 0)
		{}

		std::string BlockNode::getTypeStr() const
		{
			return "BLOCK";
		}
	}
}
