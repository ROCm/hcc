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
 * ModuleInfo.h
 *
 *  Created on: 25.02.2010
 *      Author: Simon Moll
 */

#ifndef MODULEINFO_HPP_
#define MODULEINFO_HPP_

#include <llvm/Module.h>

#include <llvm/Support/FormattedStream.h>
#include <llvm/PassManager.h>
//#include <llvm/TypeSymbolTable.h>

#include <axtor/CommonTypes.h>
#include <axtor/util/BlockCopyTracker.h>

#include <string>

class StructType;

namespace axtor {

/*
 * base class for Backend specific ModuleInfo classes that
 * provides additional information about the target llvm::Module
 */
class ModuleInfo : public BlockCopyTracker
{
	llvm::Module & M;

public:
	ModuleInfo(llvm::Module & _M);

	virtual ~ModuleInfo();

	const llvm::Module * getModule() const;

	/*
	 * returns whether this the module info object for that module
	 */
	virtual bool isTargetModule(llvm::Module*) const=0;

	/*
	 * create global scope identifier bindings
	 */
	virtual IdentifierScope createGlobalBindings() = 0;

	/*
	 * verify the module integrity with respect to target language limitations
	 */
	virtual void verifyModule() = 0;

	/*
	 * dump information about this module descriptor
	 */
	virtual void dump()=0;

	/*
	 * dump all information contained in this module info object
	 */
	virtual void dumpModule()=0;

	virtual void runPassManager(llvm::PassManager & pm)=0;

	/*
	 * returns the type bound to that name
	 */
	//virtual const llvm::Type * lookUpType(const std::string & name) const;

	/*
	 * write all extracted data to the given output stream
	 */
	virtual void writeToLLVMStream(llvm::formatted_raw_ostream & out)
	{ assert(false && "not implemented"); }

};

}

#endif /* MODULEINFO_HPP_ */
