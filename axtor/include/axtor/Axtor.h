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
 * Axtor.h
 *
 *  Created on: 28.03.2010
 *      Author: Simon Moll
 */

#ifndef AXTOR_HPP_
#define AXTOR_HPP_

#include <axtor/backend/AxtorBackend.h>
#include <axtor/metainfo/ModuleInfo.h>

namespace axtor {

/*
 * initializes the library
 */
void initialize(bool initLLVM = true);

/*
 * translates the module described by modinfo using the corresponding backend
 */
void translateModule(AxtorBackend & backend, ModuleInfo & modInfo);

/*
 * adds all required passes to @pm for translating the module
 */
void addBackendPasses(AxtorBackend & backend, ModuleInfo & modInfo, llvm::PassManager & pm);

}

#endif /* AXTOR_HPP_ */
