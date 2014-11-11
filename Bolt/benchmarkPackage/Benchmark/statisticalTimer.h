/***************************************************************************                                                                                     
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.                                     
*                                                                                    
*   Licensed under the Apache License, Version 2.0 (the "License");   
*   you may not use this file except in compliance with the License.                 
*   You may obtain a copy of the License at                                          
*                                                                                    
*       http://www.apache.org/licenses/LICENSE-2.0                      
*                                                                                    
*   Unless required by applicable law or agreed to in writing, software              
*   distributed under the License is distributed on an "AS IS" BASIS,              
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         
*   See the License for the specific language governing permissions and              
*   limitations under the License.                                                   

***************************************************************************/                                                                                     

////////////////////////////////////////////
//	Copyright (C) 2012 Advanced Micro Devices, Inc. All Rights Reserved.
////////////////////////////////////////////

#pragma once
#ifndef _STATISTICALTIMER_CPU_H_
#define _STATISTICALTIMER_CPU_H_
#include <iosfwd>
#include <vector>
#include <algorithm>
#include <stdexcept> 
#include "bolt/unicode.h"


namespace bolt
{
/**
 * \file StatisticalTimer.CPU.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with high accuracy.
 *	This class is implemented entirely in the header, to facilitate inclusion into multiple
 *	projects without having to compile an object file for each project.
 */

class statTimer
{
	typedef unsigned long long ulong;
	typedef unsigned int uint;

	//	Private typedefs
	typedef std::vector< ulong > clkVector;
	typedef	std::pair< tstring, uint > labelPair;
	typedef	std::vector< labelPair > stringVector;

	//	In order to calculate statistics <std. dev.>, we must keep a history of our timings.
	stringVector	labelID;
	clkVector	clkStart;
	std::vector< clkVector >	clkTicks;

	//	How many clockticks in a second.
	ulong	clkFrequency;

	//	For LINUX; the resolution of a high-precision timer.
#if defined( __GNUC__ )
	timespec res;
#endif

	//	Saved sizes for our vectors, used in Reset() to reallocate vectors.
	clkVector::size_type	nEvents, nSamples;

	//	This setting controls whether the Timer converts samples into time by dividing by the 
	//	clock frequency.
	bool normalize;

	/**
	 * \fn StatisticalTimer()
	 * \brief Constructor for StatisticalTimer that initializes the class.
	 *	This is private, so that user code cannot create its own instantiation.  Instead, you
	 *	must go through getInstance( ) to get a reference to the class.
	 */
	statTimer( );

	/**
	 * \fn ~StatisticalTimer()
	 * \brief Destructor for StatisticalTimer that cleans up the class.
	 */
	~statTimer( );

	/**
	 * \fn StatisticalTimer(const StatisticalTimer& )
	 * \brief Copy constructors do not make sense for a singleton; disallow copies.
	 */
	statTimer( const statTimer& );

	/**
	 * \fn operator=( const StatisticalTimer& )
	 * \brief Assignment operator does not make sense for a singleton; disallow assignments.
	 */
	statTimer& operator=( const statTimer& );

	friend tstream& operator<<( tstream& os, const statTimer& s );

	/**
	 * \fn void AddSample( const size_t id, const ulong n )
	 * \brief Explicitely add a timing sample into the class
	 */
	void AddSample( const size_t id, const ulong n );

public:
	/**
	 * \fn getInstance()
	 * \brief This returns a reference to the singleton timer.  Guarantees that only one timer class is
	 *	instantiated within a compilable executable.
	 */
	static statTimer& getInstance( );

	/**
	 * \fn void Start( size_t id )
	 * \brief Start the timer
	 * \sa Stop(), Reset()
	 */
	void Start( size_t id );

	/**
	 * \fn void Stop( size_t id )
	 * \brief Stop the timer
	 * \sa Start(), Reset()
	 */
	void Stop( size_t id );

	/**
	 * \fn void Reset(void)
	 * \brief Reset the timer to 0
	 * \sa Start(), Stop()
	 */
	void Clear( );

	/**
	 * \fn void Reset(void)
	 * \brief Reset the timer to 0
	 * \sa Start(), Stop()
	 */
	void Reset( );

	void Reserve( size_t nEvents, size_t nSamples );

	size_t getUniqueID( const tstring& label, uint groupID );

	//	Calculate the average/mean of data for a given event.
	void	convert2seconds( bool norm );

	//	Calculate the average/mean of data for a given event.
	double	getMean( size_t id ) const;

	//	Calculate the variance of data for a given event.
	//	Variance - average of the squared differences between data points and the mean.
	double	getVariance( size_t id ) const;

	//	Sqrt of variance, also in units of the original data.
	double	getStdDev( size_t id ) const;

	/**
	 * \fn double getAverageTime(size_t id) const
	 * \return Return the arithmetic mean of all the samples that have been saved.
	 */
	double getAverageTime( size_t id ) const;

	/**
	 * \fn double getMinimumTime(size_t id) const
	 * \return Return the arithmetic min of all the samples that have been saved.
	 */
	double getMinimumTime( size_t id ) const;

	//	Using the stdDev of the entire population (of an id), eliminate those samples that fall
	//	outside some specified multiple of the stdDev.  This assumes that the population
	//	form a Gaussian curve.
	size_t pruneOutliers( double multiple );
	size_t pruneOutliers( size_t id , double multiple );
};

}
#endif // _STATISTICALTIMER_CPU_H_
