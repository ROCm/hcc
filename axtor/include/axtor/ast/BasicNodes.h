/*  Axtor - AST-Extractor for LLVM
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
/*
 * BasicNodes.h
 *
 *  Created on: 12.06.2010
 *      Author: Simon Moll
 */

#ifndef BASICNODES_HPP_
#define BASICNODES_HPP_

#include <map>

#include <llvm/Function.h>

#include "ASTNode.h"
#include <axtor/CommonTypes.h>

/*
 * the extractor exclusively uses nodes described in this file to rebuild the AST. More sophisticated control flow elements are created by enhancing a pre-existing AST
 */
namespace axtor
{
	namespace ast
	{

		/*
		 * represents functions
		 */
		class FunctionNode
		{
			llvm::Function * func;
			ControlNode * entryNode;

		public:
			llvm::Function * getFunction() const;
			ControlNode * getEntry() const;

			FunctionNode(llvm::Function * _func, ControlNode * _entryNode);
			void dump() const;
		};

		typedef std::map<llvm::Function*,FunctionNode*> ASTMap;

		/*
		 * conditional node (if / if..else)
		 *
		 * The concrete structure of the described conditional entity is determined by the values of @onTrueNode and @onFalseNode
		 *
		 * Note that in a textual representation it might be necessary to negate the controlling expression in cases like onTrueNode == NULL, onFalseNode != NULL
		 */
		class ConditionalNode : public ControlNode
		{
		public:
			enum ChildIndex
			{
				ON_TRUE = 0,
				ON_FALSE = 1
			};

			ConditionalNode(llvm::BasicBlock * _block, ControlNode * onTrueNode, ControlNode * onFalseNode);

			llvm::Value * getCondition() const;
			ControlNode * getOnTrue() const;
			ControlNode * getOnFalse() const;
			llvm::BasicBlock * getOnTrueBlock() const;
			llvm::BasicBlock * getOnFalseBlock() const;

			virtual std::string getTypeStr() const;
		};

		/*
		 * Infinitely looping node
		 */
		class LoopNode : public ControlNode
		{
		public:
			enum ChildIndex
			{
				BODY = 0
			};

			LoopNode(ControlNode * _body);
			virtual std::string getTypeStr() const;
		};

		/*
		 * break
		 */
		class BreakNode : public ControlNode
		{
		public:
			BreakNode(llvm::BasicBlock * _block);
			virtual std::string getTypeStr() const;
		};

		/*
		 * continue
		 */
		class ContinueNode : public ControlNode
		{
		public:
			ContinueNode(llvm::BasicBlock * _block);
			virtual std::string getTypeStr() const;
		};

		/*
		 * return
		 */
		class ReturnNode : public ControlNode
		{
		public:
			ReturnNode(llvm::BasicBlock * _block);
			llvm::ReturnInst * getReturn() const;
			virtual std::string getTypeStr() const;
		};

		/*
		 * unreachable
		 */
		class UnreachableNode : public ControlNode
		{
		public:
			virtual std::string getTypeStr() const;
			UnreachableNode(llvm::BasicBlock * _block);
		};


		/*
		 * sequence of ast nodes
		 */
		class ListNode : public ControlNode
		{
		public:
			/*
			 * constructor for single unconditionally branching basic blocks
			 */
			ListNode(const NodeVector & nodeVector);
			virtual std::string getTypeStr() const;
		};

		/*
		 * wrapper for a unconditionally branching basic block
		 */
		class BlockNode : public ControlNode
		{
		public:
			BlockNode(llvm::BasicBlock * _block);
			virtual std::string getTypeStr() const;
		};
	}
}

#endif /* BASICNODES_HPP_ */
