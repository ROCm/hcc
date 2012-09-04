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
 * ResourceGuard.h
 *
 *  Created on: 19.03.2010
 *      Author: Simon Moll
 */

#ifndef RESOURCEGUARD_HPP_
#define RESOURCEGUARD_HPP_

namespace axtor {
/*
 * required for creating local scopes for heap objects
 */
template<class T>
class ResourceGuard
{
	T * obj;
public:

	ResourceGuard(T * _obj) :
		obj(_obj)
	{
	#ifdef DEBUG
		std::cerr << "ResourceGuard ctor()" << std::endl;
	#endif
	}

	~ResourceGuard()
	{
	#ifdef DEBUG
		std::cerr << "ResourceGuard dtor()" << std::endl;
	#endif
		if (obj) {
			delete obj;
		}
	}

	inline T * get()
	{
		return obj;
	}

	inline T & operator*()
	{
		return *obj;
	}

	inline T * operator ->()
	{
		return obj;
	}
};

}


#endif /* RESOURCEGUARD_HPP_ */
