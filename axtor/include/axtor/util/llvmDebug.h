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
 * llvmDebug.h
 *
 *  Created on: 16.03.2010
 *      Author: Simon Moll
 */

#ifndef LLVMDEBUG_HPP_
#define LLVMDEBUG_HPP_

#include <stdio.h>
#include <iostream>

#include <llvm/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Analysis/Verifier.h>

#include <axtor/CommonTypes.h>


namespace axtor {

	void dumpBlockVector(const BlockVector & blocks);

	std::string toString(const BlockSet & blocks);
	void dumpBlockSet(const BlockSet & blocks);

	void dumpTypeSet(const TypeSet & types);

	void verifyModule(llvm::Module & mod);

	void dumpUses(llvm::Value * val);

}

#endif /* LLVMDEBUG_HPP_ */
