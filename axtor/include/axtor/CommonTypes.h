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
 * CommonTypes.h
 *
 *  Created on: ?.2010
 *      Author: Simon Moll
 */

#ifndef COMMONTYPES_HPP
#define COMMONTYPES_HPP

#include <map>
#include <vector>
#include <set>
#include <stack>
#include <stdio.h>
#include <iostream>

#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/APFloat.h>

#include <llvm/Pass.h>
#include <llvm/PassManager.h>
#include <llvm/PassManagers.h>
#include <llvm/BasicBlock.h>
#include <llvm/Module.h>
#include <llvm/Instructions.h>
#include <llvm/Instruction.h>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/Dominators.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Transforms/Utils/ValueMapper.h>


//#include <util/llvmShortCuts.h>
#include <axtor/util/stringutil.h>

namespace axtor {

	typedef llvm::ValueToValueMapTy ValueMap;
			//llvm::DenseMap<const llvm::Value*, llvm::Value*> ValueMap;

	typedef std::set<const llvm::StructType*> StructTypeSet;
	typedef std::set<const llvm::Type*> TypeSet;
	typedef std::map<const llvm::Type*, std::string> TypeNames;
	typedef std::vector<llvm::StructType*> StructTypeVector;
	typedef std::vector<llvm::Function*> FunctionVector;

	typedef std::vector<llvm::Value*> ValueVector;
	typedef std::set<llvm::Value*> ValueSet;
	typedef std::set<const llvm::Value*> ConstValueSet;

	typedef std::set<llvm::PHINode*> PHISet;

	typedef std::set<llvm::Instruction*> InstructionSet;
	typedef llvm::Function::ArgumentListType ArgList;
	typedef std::vector<llvm::BasicBlock*> BlockVector;
	typedef std::set<llvm::BasicBlock*> BlockSet;
	typedef std::vector<BlockSet> BlockSetVector;
	typedef std::pair<BlockSet, BlockSet> BlockSetPair;

	typedef std::vector<std::string> StringVector;
	typedef std::set<std::string> StringSet;

	typedef std::set<llvm::BasicBlock*> BlockSet;
	typedef std::set<const llvm::BasicBlock*> ConstBlockSet;
    typedef std::pair<llvm::BasicBlock*,llvm::BasicBlock*> BlockPair;
    typedef std::vector<std::pair<llvm::BasicBlock*,llvm::BasicBlock*> > BlockPairVector;
    typedef std::stack<llvm::BasicBlock*> BlockStack;

    typedef std::vector<llvm::Loop*> LoopVector;
    typedef std::set<llvm::Loop*> LoopSet;


	/*
	 * generic variable descriptor
	 */
	struct VariableDesc
	{
		const llvm::Type * type;
		std::string name;
		bool isAlloca;

		VariableDesc(const llvm::Value * val, std::string _name);
		VariableDesc();

		void print(llvm::raw_ostream & out);
	};

	typedef std::map<llvm::Value*, VariableDesc> VariableMap;
	typedef std::map<const llvm::Value*, VariableDesc> ConstVariableMap;

	/*
	 * region context information - used by extraction passes
	 */
	struct ExtractorContext {
		llvm::Loop * parentLoop;
		llvm::BasicBlock * continueBlock;
		llvm::BasicBlock * breakBlock;
		llvm::BasicBlock * exitBlock;

		InstructionSet expressionInsts; //distinguishes between insts used in a closed expressions and normal insts (prechecked-while)
		bool isPrecheckedLoop;

		ExtractorContext();

		/*
		* {B_break, B_continue}
		*
		* (used by ASTExtractor)
		*/
		BlockSet getRegularExits() const;

		/*
		* {B_break, B_continue, B_exit}
		*
		* (used by RestructuringPass)
		*/
		BlockSet getAnticipatedExits() const;

		void dump() const;
		void dump(std::string prefix) const;

	};

	/*
	 * stacked identifier bindings
	 */
	class IdentifierScope
	{
		IdentifierScope * parent;

	public:
		ConstVariableMap identifiers;

		IdentifierScope(const ConstVariableMap & _identifiers);
		IdentifierScope(IdentifierScope * _parent, const ConstVariableMap & _identifiers);
		IdentifierScope(IdentifierScope * _parent);

		IdentifierScope * getParent() const;
		const VariableDesc * lookUp(const llvm::Value * value) const;
		void bind(const llvm::Value * val, VariableDesc desc);

		ConstVariableMap::const_iterator begin() const;
		ConstVariableMap::const_iterator end() const;
	};
}


#endif
