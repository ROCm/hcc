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
 * ExtractorRegion.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef EXTRACTORREGION_HPP_
#define EXTRACTORREGION_HPP_

#include <axtor/CommonTypes.h>

#include <vector>

namespace axtor {
	/*
	 * header dominated dominance region with extractor context
	 */
	class ExtractorRegion {
		llvm::BasicBlock * header;
	public:

		ExtractorContext context;
		llvm::BasicBlock * getHeader() const { return header; }

		ExtractorRegion(llvm::BasicBlock * _header, ExtractorContext & _context);

		void dump() const;
		void dump(std::string prefix) const;
		bool verify(llvm::DominatorTree & domTree) const;
		bool contains(llvm::DominatorTree & domTree, const llvm::BasicBlock * block) const; //FIXME: does not check context-boundaries
	};

	typedef std::vector<ExtractorRegion> RegionVector;

}
#endif /* EXTRACTORREGION_HPP_ */
