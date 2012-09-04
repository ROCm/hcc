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
 * llvmTools.h
 *
 *  Created on: 07.02.2010
 *      Author: Simon Moll
 */

#ifndef LLVMTOOLS_HPP_
#define LLVMTOOLS_HPP_


#include <iostream>

#include <llvm/Module.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/MemoryBuffer.h>

#include <axtor/util/SharedContext.h>

namespace axtor {

	llvm::Module* createModuleFromFile(std::string fileName);

	llvm::Module* createModuleFromFile(std::string fileName, llvm::LLVMContext & context);

	void writeModuleToFile(llvm::Module * M, const std::string & fileName);

}

#endif /* LLVMTOOLS_HPP_ */
