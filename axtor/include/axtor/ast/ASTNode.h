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
 * ASTNode.h
 *
 *  Created on: 12.06.2010
 *      Author: Simon Moll
 */

#ifndef ASTNODE_HPP_
#define ASTNODE_HPP_

#include <llvm/BasicBlock.h>

#include <vector>
#include <map>

namespace axtor
{
	namespace ast
	{
		class ControlNode;
		typedef std::vector<ControlNode*> NodeVector;
		typedef std::map<llvm::BasicBlock*, ControlNode*> NodeMap;

		/*
		 * nodes representing control flow structure
		 */
		class ControlNode
		{
		public:
			typedef std::vector<ControlNode*> NodeVector;
			enum NodeType
			{
				IF,          //if or if..else
				LOOP,        //infinite loop
				BREAK,       //break
				CONTINUE,    //continue
				RETURN,      //return
				UNREACHABLE, //unreachable terminator
				BLOCK,       //unconditionally branching basic block (wrapper)
				LIST         //list of nodes
			};

		private:
			NodeType type;
			llvm::BasicBlock * block;
			NodeVector children;

		public:

			ControlNode(NodeType _type, llvm::BasicBlock * _block, uint numChildren);
			ControlNode(NodeType _type, llvm::BasicBlock * _block, const NodeVector & destChildren);
			virtual ~ControlNode();

			// returns the block this node represents
			llvm::BasicBlock * getBlock() const;
			ControlNode * getNode(int idx) const;

			void setNode(int idx, ControlNode * node);
			NodeType getType() const;

			llvm::TerminatorInst * getTerminator() const;

			NodeVector::const_iterator begin() const;
			NodeVector::const_iterator end() const;

			/*
			 * determines the first executed block on evaluation
			 */
			virtual llvm::BasicBlock * getEntryBlock() const;

			virtual void dump(std::string prefix) const;
			void dump() const;

			virtual std::string getTypeStr() const=0;
		};
	}
}

#endif /* ASTNODE_HPP_ */
