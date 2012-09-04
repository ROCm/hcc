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
 * Assert.h
 *
 *  Created on: 08.09.2011
 *      Author: Dmitri Rubinstein
 */

#ifndef ASSERT_HPP_
#define ASSERT_HPP_

#include <assert.h>

#define AXTOR_UNUSED(expr) (void)sizeof(expr)

#ifndef NDEBUG
#define AXTOR_ASSERT(cond) assert(cond)
#else
#define AXTOR_ASSERT(cond) AXTOR_UNUSED(cond)
#endif

#endif /* ASSERT_HPP_ */
