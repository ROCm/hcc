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
 * OCLBackend.h
 *
 *  Created on: 07.04.2010
 *      Author: Simon Moll
 */

#ifndef OCLBACKEND_HPP_
#define OCLBACKEND_HPP_

#include <axtor/backend/AxtorBackend.h>
#include <axtor/intrinsics/PlatformInfo.h>

#include "OCLWriter.h"
#include "OCLModuleInfo.h"


namespace axtor {

class OCLBackend : public AxtorBackend
{
	static PlatformInfo * platform;

	static void init();

public:
	OCLBackend();

	virtual bool hasValidType(ModuleInfo * modInfo);

	virtual const std::string & getName();
	virtual const std::string & getLLVMDataLayout();

	static const std::string & getNameString();
	static const std::string & getLLVMDataLayoutString();

	//factory methods
	virtual SyntaxWriter * createModuleWriter(ModuleInfo & modInfo, const IdentifierScope & globals);

	virtual SyntaxWriter * createFunctionWriter(SyntaxWriter * modWriter, llvm::Function * func);

	virtual SyntaxWriter * createBlockWriter(SyntaxWriter * writer);

	virtual bool implementsFunction(llvm::Function * func);

	//interface for specifying passes specific to this backend
	virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;
	virtual void addRequiredPasses(llvm::PassManager & pm) const;
};

}

#endif /* OCLBACKEND_HPP_ */
