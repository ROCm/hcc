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
 * llvmConstant.h
 *
 *  Created on: 05.03.2010
 *      Author: Simon Moll
 */

#ifndef LLVMCONSTANT_HPP_
#define LLVMCONSTANT_HPP_

#include <string>

#include <llvm/ADT/APInt.h>
#include <llvm/Value.h>
#include <llvm/Constants.h>
#include <llvm/Instructions.h>
#include <llvm/GlobalVariable.h>

#include <axtor/util/SharedContext.h>

namespace axtor {

llvm::ConstantInt * get_uint(uint val);

llvm::ConstantInt * get_uint(uint val, llvm::LLVMContext & context, llvm::IntegerType * type);

llvm::ConstantInt * get_int(int val);

llvm::ConstantInt * get_int(uint64_t val, int bits);

llvm::ConstantInt * get_int(int val, llvm::LLVMContext & context, llvm::IntegerType * type);

llvm::Constant * get_stringGEP(llvm::Module * module, std::string content);

bool evaluateString(llvm::Value * val, std::string & out);

bool evaluateInt(llvm::Value * val, uint64_t & oValue);

}

#endif /* LLVMCONSTANT_HPP_ */
