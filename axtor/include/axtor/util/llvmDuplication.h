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
 * llvmDuplication.h
 *
 *  Created on: Jun 21, 2010
 *      Author: Simon Moll
 */

#ifndef LLVMDUPLICATION_HPP_
#define LLVMDUPLICATION_HPP_

#include <axtor/config.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/LoopPass.h>

#include <axtor/CommonTypes.h>
#include <axtor/util/BlockCopyTracker.h>
#include "llvmShortCuts.h"


namespace axtor {

/*
 * a wrapper for CloneBasicBlock that remaps all instructions of the clone
 */
llvm::BasicBlock * cloneBlockAndMapInstructions(llvm::BasicBlock * block, ValueMap & cloneMap);

BlockSet splitNode(BlockCopyTracker & tracker, llvm::BasicBlock * srcBlock); //dummy tracker argument (legacy code support)
BlockSet splitNode(llvm::BasicBlock * srcBlock, llvm::DominatorTree * domTree = NULL);

// splits a node for a set of branches, does ignore empty sets in predecessorSet and returns 0 for the cloned block of them
BlockVector splitNodeExt(llvm::BasicBlock * srcBlock, BlockSetVector predecessorSet, llvm::DominatorTree * domTree);

LoopSet splitLoop(BlockCopyTracker & tracker, llvm::LoopInfo & loopInfo, llvm::Loop * loop, llvm::Pass * pass); //dummy tracker argument (legacy code support)
LoopSet splitLoop(llvm::LoopInfo & loopInfo, llvm::Loop * loop, llvm::Pass * pass, llvm::DominatorTree * domTree = NULL);

llvm::BasicBlock * cloneBlockForBranch(BlockCopyTracker & tracker, llvm::BasicBlock * srcBlock, llvm::BasicBlock * branchBlock);
llvm::BasicBlock * cloneBlockForBranch(llvm::BasicBlock * srcBlock, llvm::BasicBlock * branchBlock, llvm::DominatorTree * domTree = NULL);
llvm::BasicBlock * cloneBlockForBranchSet(llvm::BasicBlock * srcBlock, BlockSet branchSet, llvm::DominatorTree * domTree);

/*
 * fixes instructions inside the cloned blocks and instruction using the original blocks, such that @branchBlock exclusively branches to the cloned Blocks
 */
void patchClonedBlocksForBranch(ValueMap & cloneMap, const BlockVector & originalBlocks, const BlockVector & clonedBlocks, llvm::BasicBlock * branchBlock);
void patchClonedBlocksForBranches(ValueMap & cloneMap, const BlockVector & originalBlocks, const BlockVector & clonedBlocks, BlockSet branchBlocks);

//FIXME

//llvm::Loop * cloneLoopForBranch(BlockCopyTracker & tracker, llvm::LPPassManager & lpm, llvm::Pass * pass, llvm::LoopInfo & loopInfo, llvm::Loop * loop, llvm::BasicBlock * branchBlock);  //dummy tracker argument (legacy code support)

//llvm::Loop * cloneLoopForBranch(llvm::LPPassManager & lpm, llvm::Pass * pass, llvm::LoopInfo & loopInfo, llvm::Loop * loop, llvm::BasicBlock * branchBlock, llvm::DominatorTree * domTree=NULL);

}

#endif /* LLVMDUPLICATION_HPP_ */
