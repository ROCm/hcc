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
 * AddressSpaces.h
 *
 *  Created on: 03.05.2010
 *      Author: Simon Moll
 */

#ifndef ADDRESSSPACES_HPP_
#define ADDRESSSPACES_HPP_

//### supported address spaces
// PTX-compatible address spaces (do not work)
#define SPACE_GLOBAL   1
#define SPACE_LOCAL    2
#define SPACE_CONSTANT 3

// other address spaces
#define SPACE_PRIVATE  4
#define SPACE_POINTER  5

#define SPACE_NOPTR 100


#endif /* ADDRESSSPACES_HPP_ */
