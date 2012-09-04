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
 * BlockCopyTracker.h
 *
 *  Created on: 27.06.2010
 *      Author: Simon MOll
 */

#ifndef BLOCKCOPYTRACKER_HPP_
#define BLOCKCOPYTRACKER_HPP_

#include <map>

#include <axtor/CommonTypes.h>

namespace axtor {

class BlockCopyTracker
{
	typedef	const llvm::BasicBlock * ConstBlock;
	typedef std::map<ConstBlock,int> IndexMap;

	IndexMap indices;
	BlockVector originalBlocks;
	int getIndex(ConstBlock block) const;

public:
	BlockCopyTracker(llvm::Module & M);
	virtual ~BlockCopyTracker();

	//duplicate tracker functionality
	void identifyBlocks(ConstBlock first, ConstBlock second);
	bool equalBlocks(ConstBlock first, ConstBlock second) const;
	ConstBlockSet getEqualBlocks(const llvm::BasicBlock * block) const;
	void dump() const;

	llvm::BasicBlock * getOriginalBlock(const llvm::BasicBlock * block) const;
	bool isOriginalBlock(const llvm::BasicBlock * block) const;
};

}

#endif /* BLOCKCOPYTRACKER_HPP_ */
