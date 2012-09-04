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
 * AxtorBackend.h
 *
 *  Created on: 19.03.2010
 *      Author: Simon Moll
 */

#ifndef AXTORBACKEND_HPP_
#define AXTORBACKEND_HPP_

#include <llvm/PassManagers.h>
#include <llvm/PassAnalysisSupport.h>
#include <llvm/PassManager.h>
#include <llvm/Function.h>

#include <axtor/metainfo/ModuleInfo.h>
#include <axtor/intrinsics/PlatformInfo.h>
#include <axtor/writer/SyntaxWriter.h>
#include <axtor/CommonTypes.h>

namespace axtor {

struct AxtorBackend
{
	virtual const std::string & getName()=0;
	virtual const std::string & getLLVMDataLayout()=0;

	//verifier
	virtual bool hasValidType(ModuleInfo * moduleInfo)=0;

	//factory methods
	virtual SyntaxWriter * createModuleWriter(ModuleInfo & modInfo, const IdentifierScope & globals)=0;
	virtual SyntaxWriter * createFunctionWriter(SyntaxWriter * modWriter, llvm::Function * func)=0;
	virtual SyntaxWriter * createBlockWriter(SyntaxWriter * writer)=0;

	virtual bool implementsFunction(llvm::Function * func)=0;

	//interface for specifying passes specific to this backend
	virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const=0;
	virtual void addRequiredPasses(llvm::PassManager & pm) const=0;
};

}

#endif /* AXTORBACKEND_HPP_ */
