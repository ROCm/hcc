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
 * llvmLoop.h
 *
 *  Created on: Aug 6, 2010
 *      Author: Simon Moll
 */

#ifndef LLVMLOOP_HPP_
#define LLVMLOOP_HPP_

#include <axtor/config.h>

#include <llvm/Analysis/LoopInfo.h>
#include <axtor/CommonTypes.h>

namespace axtor {

	/*
	 * identify the immediate sub loop of @parent that contains @block.
	 *
	 * returns NULL if no such loop exists
	 */
	llvm::Loop * getNestedLoop(llvm::LoopInfo & loopInfo, llvm::Loop * parent, llvm::BasicBlock * block);

}


#endif /* LLVMLOOP_HPP_ */
