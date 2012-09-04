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
 * SharedContext.h
 *
 *  Created on: 27.02.2010
 *      Author: Simon Moll
 */

#ifndef SHAREDCONTEXT_HPP_
#define SHAREDCONTEXT_HPP_

#include <iostream>
#include <assert.h>

#include <llvm/LLVMContext.h>

namespace axtor {

class SharedContext
{
	static llvm::LLVMContext * context;

public:
	static void init();

	static llvm::LLVMContext & get();
};

}


#endif /* SHAREDCONTEXT_HPP_ */
