/*
 * AnalysisProvider.h
 *
 *  Created on: 1 Feb 2012
 *      Author: v1smoll
 */

#ifndef ANALYSISPROVIDER_H_
#define ANALYSISPROVIDER_H_

namespace llvm {
	class Function;
}

namespace axtor {
	class AnalysisStruct;

// Callback interface for refreshing AnalysisStructs outside of passes
	class AnalysisProvider
	{
	public:
		virtual ~AnalysisProvider() {}

		// must only be called, if run() was invocated
		virtual void rebuildAnalysisStruct(llvm::Function & func, AnalysisStruct & analysis) = 0;
	};

}


#endif /* ANALYSISPROVIDER_H_ */
