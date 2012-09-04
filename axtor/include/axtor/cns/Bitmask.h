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
 * Bitmask.h
 *
 *  Created on: 30.04.2010
 *      Author: Simon Moll
 */

#ifndef BITMASK_HPP_
#define BITMASK_HPP_

#include <vector>
#include <iostream>

namespace axtor {

	typedef std::vector<bool> BoolVector;

	/*
	 * bit operations
	 */
	inline BoolVector AND(BoolVector a, BoolVector b)
	{
		BoolVector mask(a.size(), false);

		for(uint i = 0; i < a.size(); ++i)
		{
			mask[i] = a[i] && b[i];
		}

		return mask;
	}

	inline BoolVector OR(BoolVector a, BoolVector b)
	{
		BoolVector mask(a.size(), false);

		for(uint i = 0; i < a.size(); ++i)
		{
			mask[i] = a[i] || b[i];
		}

		return mask;
	}

	inline BoolVector NOT(BoolVector a)
	{
		BoolVector mask(a.size(), false);

		for(uint i = 0; i < a.size(); ++i)
		{
			mask[i] = !a[i];
		}

		return mask;
	}

	/*
	 * quantors
	 */
	inline bool EXISTS(BoolVector a)
	{
		for(uint i = 0; i < a.size(); ++i)
		{
			if (a[i])
				return true;
		}

		return false;
	}

	inline bool NONE(BoolVector a)
	{
		return !EXISTS(a);
	}

	inline bool ALL(BoolVector a)
	{
		return !EXISTS(NOT(a));
	}

	inline void dumpVector(BoolVector mask)
	{
		for(uint i = 0; i < mask.size(); ++i) {
			if (mask[i])
				std::cerr << "1";
			else
				std::cerr << "0";
		}
		std::cerr << "\n";
	}
}


#endif /* BITMASK_HPP_ */
