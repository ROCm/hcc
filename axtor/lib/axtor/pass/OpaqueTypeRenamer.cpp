/*
 * OpaqueTypeRenamer.cpp
 *
 *  Created on: 09.05.2010
 *      Author: gnarf
 */

#include <axtor/pass/OpaqueTypeRenamer.h>
#include <axtor/util/llvmDebug.h>
#include <axtor/pass/RestructuringPass.h>

namespace axtor
{

llvm::RegisterPass<OpaqueTypeRenamer> __regTypeRenamer("renamer", "opaque type renamer (quick dirty fix for the clang frontend, converts _class.NAME_ to _NAME_)", false, false);

char OpaqueTypeRenamer::ID = 0;

OpaqueTypeRenamer::OpaqueTypeRenamer()
: ModulePass(ID) {}

bool OpaqueTypeRenamer::runOnModule(llvm::Module & mod)
{
/*#ifdef DEBUG_PASSRUN
		std::cerr << "\n\n##### PASS: OpaqueTypeRenamer #####\n\n";
#endif
	llvm::TypeSymbolTable & symTable = mod.getTypeSymbolTable();
	llvm::TypeSymbolTable::iterator nextIt = symTable.begin();

	for(llvm::TypeSymbolTable::iterator it = symTable.begin(); it != symTable.end();)
	{
		llvm::TypeSymbolTable::iterator curr = it++;

		std::string name = curr->first;
		const llvm::Type * type = curr->second;

		if (llvm::isa<llvm::OpaqueType>(type)) {
#ifdef DEBUG
			std::cerr << "opaque type in TypeSymbolTable : " << name << "\n";
#endif

			std::string newName = "";
			if (name.substr(0, 6) == "class.") {
#ifdef DEBUG
				std::cerr << "has \"class.\" prefix  . ..  \n";
#endif
				newName = name.substr(6, std::string::npos);

			} else if (name.substr(0, 7) == "struct."){
#ifdef DEBUG
				std::cerr << "has \"struct.\" prefix  . ..  \n";
#endif
				newName = name.substr(7, std::string::npos);
			}

			if (newName != "")
			{
				symTable.remove(curr);
				symTable.insert(newName, type);
			}
		}
	}

#ifdef DEBUG
	verifyModule(mod);
#endif
*/
	return true;
}

void OpaqueTypeRenamer::getAnalysisUsage(llvm::AnalysisUsage & usage) const
{
	usage.addPreserved<RestructuringPass>();
	usage.setPreservesCFG();
}


/*void OpaqueTypeRenamer::replaceAllocaIntrinsics(llvm::Module & mod)
{
	for (llvm::Module::iterator func = mod.begin(); func != mod.end(); ++func)
	{
		if (func->isDeclaration() && (func->getNameStr().find("alloca_",0) != std::string::npos))
		{
			//llvm::Function * allocaIntrinsic = func;
			const llvm::Type * allocaType = getFunctionType(func)->getReturnType();

			for(llvm::Value::use_iterator itUse = func->use_begin(); itUse != func->use_end(); ++itUse)
			{
				assert(llvm::isa<llvm::CallInst>(itUse) && "invalid use of alloca intrinsic");

				llvm::CallInst * intrinsicCall = llvm::cast<llvm::CallInst>(itUse);
				llvm::AllocaInst * alloca = new llvm::AllocaInst(allocaType, intrinsicCall->getName(), intrinsicCall);
				intrinsicCall->replaceAllUsesWith(alloca);
				intrinsicCall->removeFromParent();
			}
		}
	}
}*/

}
