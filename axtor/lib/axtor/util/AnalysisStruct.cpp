/*
 * AnalysisStruct.cpp
 *
 *  Created on: 1 Feb 2012
 *      Author: v1smoll
 */


#include <axtor/util/AnalysisStruct.h>
#include <axtor/pass/AnalysisProvider.h>

namespace axtor {

	void AnalysisStruct::rebuild()
	{
		provider->rebuildAnalysisStruct(*func, *this);
	}

}
