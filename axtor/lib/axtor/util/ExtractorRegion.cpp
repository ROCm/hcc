/*
 * ExtractorRegion.cpp
 *
 *  Created on: 20.02.2011
 *      Author: gnarf
 */

#include <axtor/util/ExtractorRegion.h>
#include <axtor/CommonTypes.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/llvmDebug.h>

namespace axtor{
	ExtractorRegion::ExtractorRegion(llvm::BasicBlock * _header, ExtractorContext & _context) :
		header(_header), context(_context)
	{
		assert(header);
	}

	bool ExtractorRegion::verify(llvm::DominatorTree & domTree) const
	{
		BlockSetPair exitInfo = computeDominanceFrontierExt(header, domTree, context.getAnticipatedExits());
		BlockSet exits = exitInfo.first;
#ifdef DEBUG
		if (exits.size() > 0) {
			std::cerr << "invalid region (!!):\n";
			dumpBlockSet(exits);
		}
#endif

		return exits.size() == 0;
	}

	void ExtractorRegion::dump() const
	{
		dump("");
	}

	void ExtractorRegion::dump(std::string prefix) const
	{
		std::cerr << prefix << "Region (header = " << header->getName().str() << ")\n"
				  << prefix << "{";
		context.dump(prefix + "\t");
		std::cerr << prefix << "}\n";
	}

	bool ExtractorRegion::contains(llvm::DominatorTree & domTree, const llvm::BasicBlock * block) const
	{
		return domTree.dominates(getHeader(), block);
	}

}
