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
 * OpaqueTypeRenamer.h
 *
 *  Created on: 09.05.2010
 *      Author: Simon Moll
 */

#ifndef OPAQUETYPERENAMER_HPP_
#define OPAQUETYPERENAMER_HPP_

#include <axtor/config.h>

#include <iostream>

#include <llvm/LLVMContext.h>
//#include <llvm/TypeSymbolTable.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Module.h>
#include <llvm/Pass.h>
#include <llvm/Instructions.h>

#include <axtor/util/llvmShortCuts.h>

/*
 * removes the "class." or "struct." suffixes from types in the TypeSymbolTable
 */
namespace axtor
{

class OpaqueTypeRenamer : public llvm::ModulePass
{
	/*
	 * replace all calls to alloca_*() functions to AllocaInsts with the same type
	 */
	//void replaceAllocaIntrinsics(llvm::Module & mod);

public:
	static char ID;
	OpaqueTypeRenamer();

	virtual bool runOnModule(llvm::Module & mod);
	virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;
};

}

#endif /* OPAQUETYPERENAMER_HPP_ */
