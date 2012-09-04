/*
 * llvmutil.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/config.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/console/CompilerLog.h>

#include <llvm/Constants.h>
#include <llvm/Instructions.h>

namespace axtor {


FunctionVector findFunctionsPrefixed(llvm::Module & M, std::string prefix)
{
	FunctionVector funcs;

	for (llvm::Module::iterator itFunc = M.begin(); itFunc != M.end(); ++itFunc)
	{
		if (itFunc->getName().substr(0, prefix.length()) == prefix)
			funcs.push_back(itFunc);
	}

	return funcs;
}


std::string cleanDesignatorName(std::string name)
{
	std::stringstream alias;
	for(std::string::iterator itChar = name.begin(); itChar != name.end(); ++itChar)
		if (
			*itChar == '_' ||
			('0' <= *itChar && *itChar <= '9') ||
			('A' <= *itChar && *itChar <= 'Z') ||
			('a' <= *itChar && *itChar <= 'z'))
				alias << *itChar;

	return alias.str();
}

uint64_t generateTruncMask(uint width)
{
	uint64_t mask = 0;

	for(uint i = 0; i < width; ++i)
		mask |= (1 << i);

	return mask;
}

bool isGEP(llvm::Value * val)
{
	if (llvm::isa<llvm::GetElementPtrInst>(val)) {
		return true;
	} else if (llvm::isa<llvm::ConstantExpr>(val)) {
		llvm::ConstantExpr * constExpr = llvm::cast<llvm::ConstantExpr>(val);
		return constExpr->getOpcode() == llvm::Instruction::GetElementPtr;
	}

	return false;
}

bool isUnsigned(llvm::Value * val)
{
	if (llvm::isa<llvm::Instruction>(val)) {
		llvm::Instruction * inst = llvm::cast<llvm::Instruction>(val);
		switch (inst->getOpcode())
		{
			case llvm::Instruction::UDiv:
			case llvm::Instruction::URem:
				return true;
		}
	}

	return false;
}

bool isType(llvm::Value * value, const llvm::Type::TypeID type)
{
	return value->getType()->getTypeID() == type;
}

bool isVoid(llvm::Value * value)
{
	return isType(value, llvm::Type::VoidTyID);
}

bool isType(const llvm::Type * type, const llvm::Type::TypeID id)
{
	return type->getTypeID() == id;
}

bool isVoid(const llvm::Type * type)
{
	return isType(type, llvm::Type::VoidTyID);
}

const llvm::FunctionType * getFunctionType(llvm::Function * func)
{
	return llvm::cast<const llvm::FunctionType>(func->getType()->getElementType());
}

const llvm::Type * getReturnType(llvm::Function * func)
{
	return getFunctionType(func)->getReturnType();
}

BlockSetPair computeExitSet(RegionVector regions, llvm::DominatorTree & domTree, BlockSet regularExits)
{
	BlockSet exitSet;
	BlockSet usedRegularExits;

	for(RegionVector::iterator itRegion = regions.begin(); itRegion != regions.end(); ++itRegion)
	{
		llvm::BasicBlock * header = itRegion->getHeader();

		BlockSetPair ret = computeDominanceFrontierExt(header, domTree, regularExits);
		mergeInto<llvm::BasicBlock*>(exitSet, ret.first);
		mergeInto<llvm::BasicBlock*>(usedRegularExits, ret.second);
	}

	return std::make_pair<BlockSet,BlockSet>(exitSet, usedRegularExits);
}

bool reaches (llvm::BasicBlock * A, llvm::BasicBlock * B, BlockSet regularExits)
{
	assert(A && B && "was NULL");

	if (A == B)
		return true;

	BlockSet explored;
	explored.insert(A);

	std::stack<llvm::TerminatorInst*> termStack;
	termStack.push(A->getTerminator());

	while (! termStack.empty())
	{
		llvm::TerminatorInst * termInst = termStack.top();
		termStack.pop();

		for (uint i = 0; i < termInst->getNumSuccessors() ; ++i)
		{
			llvm::BasicBlock * succ = termInst->getSuccessor(i);

			if (succ == B)
				return true;

			if (explored.insert(succ).second &&
					regularExits.find(succ) == regularExits.end())
			{
				termStack.push(succ->getTerminator());
			}
		}
	}

	return false;
}

template<class T>
std::vector<T> toVector(std::set<T> S)
{
	std::vector<T> vector;
	vector.insert(vector.begin(), S.begin(), S.end());
	return vector;
}

//template<llvm::BasicBlock*>
template std::vector<llvm::BasicBlock*> toVector(std::set<llvm::BasicBlock*>);


/*
 * checks if the CFG dominated by @start is only left using @expectedExits
 * returns a set of unexpected exits
 */
BlockSetPair computeDominanceFrontierExt(llvm::BasicBlock * start, llvm::DominatorTree & domTree, BlockSet expectedExits)
{
	assert (start && "was NULL");


	BlockSet nonDomExits;
	BlockSet usedRegularExits;

	if (expectedExits.find(start) != expectedExits.end())
	{
#ifdef DEBUG_DOMFRONT
		std::cerr << "start Block was anticipated exit\n";
#endif
		usedRegularExits.insert(start);
		return std::make_pair<BlockSet, BlockSet>(nonDomExits, usedRegularExits);
	}


	std::stack<llvm::BasicBlock*> stack;
	stack.push(start);
    BlockSet processed; processed.insert(start);

#ifdef DEBUG_DOMFRONT
    std::cerr << "computeDFExt . .\n";
#endif

    do
    {
    	llvm::BasicBlock * current = stack.top();
    	stack.pop();

#ifdef DEBUG_DOMFRONT
    	std::cerr << "checking block : " << current->getNameStr() << "\n";
#endif

    	llvm::TerminatorInst * termInst = current->getTerminator();
#ifdef DEBUG_DOMFRONT
    	std::cerr << "Terminator:\n";
    	termInst->dump();
#endif

    	for(uint i = 0; i < termInst->getNumSuccessors(); ++i)
    	{
    		llvm::BasicBlock *  succ = termInst->getSuccessor(i);

    		if ((processed.find(succ) == processed.end())) {
    			if (expectedExits.find(succ) == expectedExits.end()) {
					processed.insert(succ);
					//just continue processing dominated blocks
					if (domTree.dominates(start, succ)) {
	#ifdef DEBUG_DOMFRONT
						std::cerr << "dominated : " << succ->getNameStr() << "\n";
	#endif
						stack.push(succ);
					} else {
	#ifdef DEBUG_DOMFRONT
						std::cerr << "not dominated : " << succ->getNameStr() << "\n";
	#endif
						nonDomExits.insert(succ);
					}
    			} else {
    				usedRegularExits.insert(succ);
    			}
    		}
    	}
    } while (stack.size() > 0);

    /*std::ostream_iterator< llvm::BasicBlock* > output( std::cout, " " );

    std::copy( nonDomExits.begin(), nonDomExits.end(), output );*/

    return std::make_pair<BlockSet, BlockSet>(nonDomExits, usedRegularExits);
}

/* llvm::BasicBlock * getLatch(llvm::Loop & loop, llvm::BasicBlock * entry)
{
	for(llvm::Value::use_iterator itUse = entry->use_begin(); itUse != entry->use_end(); ++itUse)
	{
		if (llvm::isa<llvm::TerminatorInst>(itUse))
		{
			llvm::TerminatorInst * termInst = llvm::cast<llvm::TerminatorInst>(itUse);
			if (loop.contains(termInst->getParent())) {
				return termInst->getParent();
			}
		}
	}

	return NULL;
} */

bool containsType(const llvm::Type * type, const TypeSet & typeSet)
{
	for(uint i = 0; i < type->getNumContainedTypes(); ++i) {
		const llvm::Type * containedType = type->getContainedType(i);

		if (typeSet.find(containedType) != typeSet.end() ||
			containsType(containedType, typeSet))
				return true;
	}

	return false;
}

bool doesContainType(const llvm::Type * type, llvm::Type::TypeID id)
{
	for(uint i = 0; i < type->getNumContainedTypes(); ++i) {
		const llvm::Type * containedType = type->getContainedType(i);

		if (containedType->getTypeID() == id)
			return true;
		else
			return doesContainType(containedType, id);
	}

	return false;
}

//FIXME
/*bool getTypeSymbol(llvm::TypeSymbolTable & typeSymbolTable, const llvm::Type * type, std::string & out)
{
	for(llvm::TypeSymbolTable::iterator it = typeSymbolTable.begin(); it != typeSymbolTable.end(); ++it)
	{
		if (it->second == type) {
			std::string tmp = it->first;
			out = tmp;//tmp.substr(0, tmp.length() - 1); // truncate the \0 at the end
			return true;
		}
	}

	return false;
}*/


int getSuccessorIndex(llvm::TerminatorInst * termInst, const llvm::BasicBlock * target)
{
	for(uint i = 0; i < termInst->getNumSuccessors(); ++i)
	{
		if (termInst->getSuccessor(i) == target)
		{
			return (int) i;
		}
	}

	return -1;
}

/*
 * returns the Opcode of Instructions and ConstantExprs
 */
uint getOpcode(const llvm::Value * value)
{
	if (llvm::isa<const llvm::ConstantExpr>(value)) {
		const llvm::ConstantExpr * expr = llvm::cast<const llvm::ConstantExpr>(value);
		return expr->getOpcode();

	} else if (llvm::isa<llvm::Instruction>(value))  {
		const llvm::Instruction * inst = llvm::cast<llvm::Instruction>(value);
		return inst->getOpcode();
	}

	Log::fail(value, "value does not encapsulate an instruction");
}

/*
 *  Taken from LLVM
 */

/// getExitEdges - Return all pairs of (_inside_block_,_outside_block_).
  void getExitEdges(llvm::Loop & loop, BlockPairVector & ExitEdges) {
    // Sort the blocks vector so that we can use binary search to do quick
    // lookups.
    llvm::SmallVector<llvm::BasicBlock*, 128> LoopBBs(loop.block_begin(), loop.block_end());
    std::sort(LoopBBs.begin(), LoopBBs.end());

    typedef llvm::GraphTraits<llvm::BasicBlock*> BlockTraits;
    for (llvm::Loop::block_iterator BI = loop.block_begin(), BE = loop.block_end(); BI != BE; ++BI)
      for (BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
           I != E; ++I)
        if (!std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
          // Not in current loop? It must be an exit block.
          ExitEdges.push_back(std::make_pair(*BI, *I));
  }



  void LazyRemapBlock(llvm::BasicBlock *BB,
                                        ValueMap &ValueMap) {
	  for (llvm::BasicBlock::iterator itInst = BB->begin(); itInst != BB->end(); ++itInst)
		  LazyRemapInstruction(itInst, ValueMap);

  }
  void LazyRemapInstruction(llvm::Instruction *I,
                                      ValueMap &ValueMap) {
    for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
      llvm::Value *Op = I->getOperand(op);
      ValueMap::iterator It = ValueMap.find(Op);
      if (It != ValueMap.end()) Op = It->second;
      I->setOperand(op, Op);
    }
  }

llvm::CmpInst * createNegation(llvm::Value * value, llvm::Instruction * before)
{
	llvm::LLVMContext &context = SharedContext::get();
	llvm::Type *boolType = llvm::Type::getInt1Ty(context);
	return llvm::CmpInst::Create(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, 
                               llvm::Constant::getAllOnesValue(boolType), 
                               value, "", before);
}

bool containsPHINodes(const llvm::BasicBlock * block)
{
	return llvm::isa<llvm::PHINode>(block->front());
}

BlockVector getAllPredecessors(llvm::BasicBlock * block)
{
	BlockVector predecessors;

	typedef llvm::GraphTraits<llvm::Inverse<llvm::BasicBlock*> > InverseCFG;
	for(InverseCFG::ChildIteratorType pred = InverseCFG::child_begin(block); pred != InverseCFG::child_end(block); ++pred)
		predecessors.push_back(*pred);

	return predecessors;
}

template<class T>
bool contains(const std::vector<T> & vector, T element)
{
	return std::find(vector.begin(), vector.end(), element) != vector.end();
}

template<class T>
bool set_contains(const std::set<T> & set, T element)
{
	return std::find(set.begin(), set.end(), element) != set.end();
}


template bool contains<llvm::BasicBlock*>(const BlockVector & vector, llvm::BasicBlock * element);
template bool set_contains<llvm::BasicBlock*>(const BlockSet & vector, llvm::BasicBlock * element);
template bool set_contains<llvm::Value*>(const ValueSet & vector, llvm::Value * element);

/*
 * "should" terminate . . . (no cycles in constant expressions)
 */
bool usedInFunction(llvm::Function * func, llvm::Value * val)
{
	for (llvm::Value::use_iterator itUse = val->use_begin(); itUse != val->use_end(); ++itUse)
	{
		llvm::User * user = *itUse;

		if (llvm::isa<llvm::Instruction>(user)) {
			llvm::Instruction * inst = llvm::cast<llvm::Instruction>(user);
			return inst->getParent()->getParent() == func;

		} else if (llvm::isa<llvm::Value>(user)) {
#ifdef DEBUG
			user->dump();
#endif
			return usedInFunction(func, llvm::cast<llvm::Value>(user));
		}
	}

	return false;
}

template<class T>
void mergeInto(std::set<T> & A, std::set<T> & B)
{
	typedef std::set<T> TSet;
	typedef typename TSet::const_iterator ConstIter;

	for(ConstIter it = B.begin(); it != B.end(); ++it)
	{
		A.insert(*it);
	}
}

template<class T>
std::set<T> getWithout(const std::set<T> & A, T object)
{
	typedef std::set<T> TSet;
	TSet result;
	result.insert(A.begin(), A.end());
	result.erase(object);
	return result;
}


//template<llvm::BasicBlock*>
template void mergeInto(std::set<llvm::BasicBlock*> & A, std::set<llvm::BasicBlock*> &B);
template BlockSet getWithout(const BlockSet & A, llvm::BasicBlock * object);

}
