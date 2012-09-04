/*
 * BlockCopyTracker.cpp
 *
 *  Created on: 27.06.2010
 *      Author: gnarf
 */

#include <axtor/util/BlockCopyTracker.h>

namespace axtor{

BlockCopyTracker::BlockCopyTracker(llvm::Module &M)
{
/*	int counter = -1;
	uint numBlocks = 0;

	for(llvm::Module::iterator func = M.begin(),
      e = M.end(); func != e; ++func)
	{
    std::cout << "function: " << func->getName().str() << "\n";
		numBlocks += func->getBasicBlockList().size();
	}

	//originalBlocks.resize(numBlocks, NULL);

	for(llvm::Module::iterator func = M.begin(); func != M.end(); ++func)
	{
		func->getBasicBlockList().size();
		for(llvm::Function::iterator block = func->begin(); block != func->end(); ++block)
		{
			indices[block] = ++counter;
			//originalBlocks[counter] = block;
		}
	}*/
}

BlockCopyTracker::~BlockCopyTracker()
{}

int BlockCopyTracker::getIndex(ConstBlock block) const
{
	IndexMap::const_iterator it = indices.find(block);
	if (it != indices.end())
		return it->second;

	return -1;
}


void BlockCopyTracker::identifyBlocks(const llvm::BasicBlock * known, const llvm::BasicBlock * copy)
{
	int knownIdx = getIndex(known);
	assert(knownIdx != -1 && "can only track copies of known blocks");

	indices[copy] = knownIdx;
}

bool BlockCopyTracker::equalBlocks(const llvm::BasicBlock * first, const llvm::BasicBlock * second) const
{
	if (first == second)
		return true;

	int firstIdx = getIndex(first);
	int secondIdx = getIndex(second);

	return (firstIdx != -1 && secondIdx != -1) && //both blocks are unknown
			(firstIdx == secondIdx);
}

ConstBlockSet BlockCopyTracker::getEqualBlocks(const llvm::BasicBlock * block) const
{
	ConstBlockSet set;

	IndexMap::const_iterator itBlock =  indices.find(block);

	if (itBlock == indices.end())
		return set;

	int index = itBlock->second;

	for(IndexMap::const_iterator it = indices.begin(); it != indices.end(); ++it)
	{
		if (it->second == index)
			set.insert(it->first);
	}

	return set;
}

llvm::BasicBlock * BlockCopyTracker::getOriginalBlock(const llvm::BasicBlock * block) const
{
	int i = getIndex(block);
	return originalBlocks[i];
}

bool BlockCopyTracker::isOriginalBlock(const llvm::BasicBlock * block) const
{
	return getOriginalBlock(block) == block;
}

void BlockCopyTracker::dump() const
{
	ConstBlockSet dumpedBlocks;
	std::cerr << "BlockCopyTracker (registered blocks=" << indices.size() << ") {\n";

	for(IndexMap::const_iterator it = indices.begin();dumpedBlocks.size() < indices.size(); ++it)
	{
		std::pair<ConstBlockSet::const_iterator, bool> result = dumpedBlocks.insert(it->first);

		if (result.second)
		{
			ConstBlockSet blocks = getEqualBlocks(it->first);
			dumpedBlocks.insert(blocks.begin(), blocks.end());

			for(ConstBlockSet::const_iterator itEqual = blocks.begin(); itEqual != blocks.end() ; ++itEqual)
			{
				const llvm::BasicBlock * equalBlock = *itEqual;

				std::cerr << (itEqual == blocks.begin() ? "\t{" : ",") << equalBlock->getName().str();
			}

			std::cerr << "}\n";
		}
	}
	std::cerr << "}\n";
}

}
