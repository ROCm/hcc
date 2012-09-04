/*
 * ASTNode.cpp
 *
 *  Created on: 12.06.2010
 *      Author: gnarf
 */

#include <axtor/ast/ASTNode.h>

#include <iostream>

namespace axtor {
namespace ast {

	ControlNode::ControlNode(NodeType _type, llvm::BasicBlock * _block, uint numChildren) :
			type(_type),
			block(_block),
			children(numChildren, NULL)
	{}


	ControlNode::ControlNode(NodeType _type, llvm::BasicBlock * _block, const NodeVector & destChildren) :
			type(_type),
			block(_block),
			children(destChildren)
	{}

	llvm::BasicBlock * ControlNode::getBlock() const
	{
		return block;
	}

	ControlNode * ControlNode::getNode(int idx) const
	{
		return children[idx];
	}

	void ControlNode::setNode(int idx, ControlNode * node)
	{
		children[idx] = node;
	}

	ControlNode::NodeType ControlNode::getType() const
	{
		return type;
	}

	llvm::TerminatorInst * ControlNode::getTerminator() const
	{
		return block->getTerminator();
	}

	ControlNode::NodeVector::const_iterator ControlNode::begin() const
	{
		return children.begin();
	}

	ControlNode::NodeVector::const_iterator ControlNode::end() const
	{
		return children.end();
	}

	llvm::BasicBlock * ControlNode::getEntryBlock() const
	{
		if (block) {
			return block;
		} else {
			for(NodeVector::const_iterator itNode = children.begin(); itNode != children.end(); ++itNode)
			{
				ControlNode * node = *itNode;
				llvm::BasicBlock * childEntry = node->getEntryBlock();
				if (childEntry)
					return childEntry;
			}
			return NULL;
		}
	}

	void ControlNode::dump() const
	{
		dump("");
	}

	void ControlNode::dump(std::string prefix) const
	{
#define PREFIXED std::cerr << prefix

		std::string name = block ? block->getName() : "none";

		std::cerr << getTypeStr() << " (" << name << ")\n";

		if (begin() != end())
		{
			PREFIXED << "{\n";
				uint i = 0;
				for(NodeVector::const_iterator itChild = begin(); itChild != end(); ++itChild, ++i)
				{
					ControlNode * child = *itChild;
					PREFIXED << "\t[" << i << "]:";
					if (child)
						child->dump(prefix + "\t");
					else
						std::cerr << "null\n";
				}
			PREFIXED << "}\n";
		}

#undef PREFIXED
	}

	ControlNode::~ControlNode()
	{
		for(NodeVector::const_iterator itNode = begin(); itNode != end(); ++itNode)
		{
			ast::ControlNode * node = *itNode;
			if (node)
				delete node;
		}
	}

}
}
