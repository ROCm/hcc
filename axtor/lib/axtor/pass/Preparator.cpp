/*
 * Preparator.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/pass/Preparator.h>
#include <axtor/util/llvmDebug.h>
#include <axtor/backend/AddressSpaces.h>

#include "llvm/TypeFinder.h"

namespace axtor {

llvm::RegisterPass<Preparator> __regPreparator("preparator", "axtor - preparator pass",
						false /* mutates the CFG */,
						false /* transformation */);

	char Preparator::ID = 0;

	/*
	 * names all unnamed values in a function
	 */
	void Preparator::nameAllInstructions(llvm::Function * func)
	{
		uint blockIdx = 0;

		typedef llvm::Function::ArgumentListType ArgList;

		ArgList & args = func->getArgumentList();

		for(ArgList::iterator itArg = args.begin(); itArg != args.end(); ++itArg)
		{
			if (itArg->getName().str() == "") {
				itArg->setName("somearg");
			}
		}

		for(llvm::Function::iterator BB = func->begin(); BB != func->end(); ++BB, ++blockIdx)
		{
			uint instIdx = 0;

			for(llvm::BasicBlock::iterator inst = BB->begin(); inst != BB->end(); ++inst, ++instIdx)
			{
				if (/* ! inst->hasName() && */ !isType(inst, llvm::Type::VoidTyID) )
				{
					std::string bogusName = func->getName().str() + '_' + str<uint>(blockIdx) + '_' + str<uint>(instIdx);
					inst->setName(bogusName);
				}
			}
		}
	}

	void Preparator::nameAllInstructions(llvm::Module * mod)
	{
		for(llvm::Module::iterator func = mod->begin(); func != mod->end(); ++func)
			if (! func->empty())
			{
				nameAllInstructions(func);
			}
	}


	/*
	 * convert all constant arguments of PHI-Nodes to instructions in the entry block
	 */

	llvm::Instruction * Preparator::createAssignment(llvm::Value * val, llvm::Instruction * beforeInst)
	{
		llvm::Type * type = val->getType();
		return new llvm::BitCastInst(val, type,"", beforeInst);
	}

	llvm::Instruction * Preparator::createAssignment(llvm::Value * val, llvm::BasicBlock * block)
	{
		return createAssignment(val, block->getTerminator());
	}

	void Preparator::transformStoreConstants(llvm::StoreInst * store)
	{
		llvm::Value * storeVal = store->getOperand(0);

		if (llvm::isa<llvm::Constant>(storeVal))
		{
			llvm::Instruction * inst = createAssignment(storeVal, store);
			store->setOperand(0, inst);
		}
	}

	Preparator::AssignmentBridge::AssignmentBridge(llvm::BasicBlock * _bridge, llvm::Instruction * _value) :
		bridge(_bridge),
		value(_value)
	{}

	Preparator::AssignmentBridge Preparator::insertBridgeForBlock(llvm::Instruction * inst, llvm::BasicBlock * branchTarget)
	{
		std::string name = branchTarget->getName().str() + "_bridge";
		llvm::Function * parent = branchTarget->getParent();
		llvm::BasicBlock * branchBlock = inst->getParent();

		//create a bridge block
		llvm::BasicBlock * bridgeBlock = llvm::BasicBlock::Create(SharedContext::get(), name, parent, branchTarget);
		llvm::BranchInst::Create(branchTarget, bridgeBlock);
		llvm::Instruction * assignInst = createAssignment(inst, bridgeBlock);

		//redirect all branches from the source block to the bridge block
		llvm::TerminatorInst * termInst = branchBlock->getTerminator();
		for(uint i = 0; i < termInst->getNumSuccessors(); ++i)
		{
			if (termInst->getSuccessor(i) == branchTarget)
			{
				termInst->setSuccessor(i, bridgeBlock);
			}
		}

		return Preparator::AssignmentBridge(bridgeBlock, assignInst);
	}

	/*
	 * transforms PHINodes into a standardized format where
	 * no phi uses constants immediately
	 */
	void Preparator::transformPHINode(llvm::PHINode * phi)
	{
		for(uint argIdx = 0; argIdx < phi->getNumIncomingValues(); ++argIdx)
		{
			llvm::Value * val = phi->getIncomingValue(argIdx);
			llvm::BasicBlock * block = phi->getIncomingBlock(argIdx);

			if (llvm::isa<llvm::Constant>(val))
			{
				llvm::Instruction * inst = createAssignment(val, block);

				phi->setIncomingValue(argIdx, inst);

			/*
			 * PHis may not use other PHIs directly, but may use their
			 * value indirectly via a variable the PHIs value is assigned to
			 * on the edge passing the value
			 */
			/*} else if (llvm::isa<llvm::PHINode>(val)) {
				llvm::PHINode * operandPHI = llvm::cast<llvm::PHINode>(val);
				llvm::BasicBlock * phiBlock = phi->getParent();
				llvm::BasicBlock * branchBlock = operandPHI->getParent();

				//create the bridge
				AssignmentBridge bridge = insertBridgeForBlock(operandPHI, phiBlock);
				llvm::BasicBlock * bridgeBlock = bridge.bridge;
				llvm::Value * assignInst = bridge.value;

				//fix up the PHI using the value
				for(uint i = 0; i < phi->getNumIncomingValues(); ++i)
				{
					if (phi->getIncomingBlock(i) == branchBlock)
					{
						phi->setIncomingBlock(i, bridgeBlock);
						phi->setIncomingValue(i, assignInst);
					}
				}*/
			}
		}
	}

	/*void Preparator::sortTypeSymbolTable(llvm::TypeSymbolTable & symTable)
	{

	}*/

	/*
	 * calls specific instruction transform methods
	 * (for stores and PHIs so far)
	 */
	void Preparator::transformInstruction(llvm::Instruction * inst)
	{
		if (llvm::isa<llvm::PHINode>(inst))
			transformPHINode(llvm::cast<llvm::PHINode>(inst));

		/* if (llvm::isa<llvm::StoreInst>(inst))
			transformStoreConstants(llvm::cast<llvm::StoreInst>(inst)); */
	}

	/*
	 * transforms all instructions occuring in a module if necessary
	 */
	void Preparator::transformInstArguments(llvm::Module * mod)
	{
		for(llvm::Module::iterator func = mod->begin(); func != mod->end() ;++func)
			for(llvm::Function::iterator bb = func->begin(); bb != func->end(); ++bb)
				for(llvm::BasicBlock::iterator inst = bb->begin(); inst != bb->end(); ++inst)
				{
					transformInstruction(inst);
				}
	}

	void Preparator::insertIfStruct(TypeSet & types, const llvm::Type * type)
	{
		for (uint i = 0; i < type->getNumContainedTypes(); ++i)
		{
			if (types.find(type) == types.end()) {
				insertIfStruct(types, type->getContainedType(i));
			}
		}

		if (llvm::isa<const llvm::StructType>(type)) {
			types.insert(type);
		}
	}

	void Preparator::cleanInternalGlobalNames(llvm::Module & M)
	{
		typedef llvm::Module::global_iterator GlobalIter;
		GlobalIter S = M.global_begin();
		GlobalIter E = M.global_end();

		uint idx = 0;
		for (GlobalIter itGlobal = S; itGlobal != E; ++itGlobal) {
			uint globalSpace = itGlobal->getType()->getAddressSpace();
			// check if this variable is addressable from the outside
			if (globalSpace == SPACE_GLOBAL)
				continue;
			if (globalSpace == SPACE_CONSTANT && !itGlobal->hasInitializer()) //FIXME see, if this is correct
				continue;

			itGlobal->setName("internal" + str<uint>(idx++));
		}
	}

	// give a generic name to all structs in @symTable
	StringSet usedNames;
	void Preparator::cleanStructNames(llvm::Module & M)
	{
		llvm::TypeFinder SrcStructTypes;
                SrcStructTypes.run(M, true);
                std::vector<llvm::StructType*> structTypes(SrcStructTypes.begin(),
                                                 SrcStructTypes.end());

		uint idx = 0;
		for (StructTypeVector::iterator itStruct = structTypes.begin();
				itStruct != structTypes.end();
				++itStruct, ++idx)
		{
			(*itStruct)->setName("StructType" + str<uint>(idx));
		}
	}

		 const char * Preparator::getPassName() const
		{
			return "axtor - preparator pass";
		}

		 void Preparator::getAnalysisUsage(llvm::AnalysisUsage & usage) const
		{
			//usage.addRequired<llvm::PostDominatorTree>();
			//usage.addRequired<llvm::DominatorTree>();

			//usage.addRequired<ExitUnificationPass>();
			//usage.addRequired<Duplicator>();

			usage.setPreservesCFG();
#ifdef ENABLE_CNS
			usage.addPreserved<CNSPass>();
#endif
		}

		Preparator::Preparator() :
			ModulePass(ID)
		{}

		 Preparator::~Preparator()
		{}

		 bool Preparator::runOnModule(llvm::Module& M)
		{
#ifdef DEBUG_PASSRUN
		std::cerr << "\n\n##### PASS: Preparator #####\n\n";
#endif
			transformInstArguments(&M);

			//nameAllNonOpaqueTypes(M, M.getTypeSymbolTable());

			nameAllInstructions(&M);

			cleanStructNames(M);

			cleanInternalGlobalNames(M);
			//sortTypeSymbolTable(M.getTypeSymbolTable());

			#ifdef DEBUG
				verifyModule(M);
				std::cerr << "\n##### module after preparations #####\n";
				M.dump();
			#endif

			return true;
		}
}
