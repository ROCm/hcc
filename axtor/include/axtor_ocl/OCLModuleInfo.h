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
 * OCLModuleInfo.h
 *
 *  Created on: 06.04.2010
 *      Author: Simon Moll
 */

#ifndef OCLMODULEINFO_HPP_
#define OCLMODULEINFO_HPP_

//#include <llvm/TypeSymbolTable.h>

#include <axtor/metainfo/ModuleInfo.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/InstructionIterator.h>

#include <vector>

namespace axtor {

/*
 * class for storing additional translation relevant information for the syntax-writer
 */
class OCLModuleInfo : public ModuleInfo
{

public:
/*	enum ArgumentModifier
	{
		IN, OUT, UNIFORM
	};*/

	//typedef std::map<std::string, ArgumentModifier> ArgumentMap;
private:

	//bitcode & arguments
	llvm::Module * mod;
	//llvm::Function * kernelFunc;
  std::vector<llvm::Function*> kernels;

	//output streams
	std::ostream & out;

	//program properties
	bool usesDoubleType;

	//scans the module for uses of the double type
	bool scanForDoubleType();

public:
	bool requiresDoubleType();

	llvm::Module * getModule();

	OCLModuleInfo(llvm::Module *mod, std::vector<llvm::Function*> kernelFunc, std::ostream &out);

	/*
	 * helper method for creating a ModuleInfo object from a module and a bind file
	 */
	static OCLModuleInfo createTestInfo(llvm::Module * mod, std::ostream & out);

	std::ostream & getStream();

	bool isKernelFunction(llvm::Function * func);

	std::vector<llvm::Function*> getKernelFunctions();

	virtual void dump();

	virtual void dumpModule();

	virtual IdentifierScope createGlobalBindings();

	virtual void runPassManager(llvm::PassManager & pm);

	/*
	 * checks whether all types are supported and
	 */
	virtual void verifyModule();

	virtual bool isTargetModule(llvm::Module * other) const;
};

}

#endif /* OCLMODULEINFO_HPP_ */
