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

// StatTimer.cpp : Defines the exported functions for the DLL application.
//

#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <cmath>
#include <limits>

#include <bolt/statisticalTimer.h>

#if defined( _WIN32 )
	#define NOMINMAX
	#include <windows.h>
#else
	#include <sys/time.h>
typedef unsigned long cl_ulong;	
#endif

namespace bolt
{

//	Functor object to help with accumulating values in vectors
template< typename T >
struct Accumulator: public std::unary_function< T, void >
{
	T acc;

	Accumulator( ): acc( 0 ) {}
	void operator( )(T x) { acc += x; }
};

//	Unary predicate used for remove_if() algorithm
//	Currently, RangeType is expected to be a floating point type, and ValType an integer type
template< typename RangeType, typename ValType >
struct PruneRange
{
	RangeType lower, upper;

	PruneRange( RangeType mean, RangeType stdev ): lower( mean-stdev ), upper( mean+stdev ) {}

	bool operator( )( ValType val )
	{ 
		//	These comparisons can be susceptible to signed/unsigned casting problems
		//	This is why we cast ValType to RangeType, because RangeType should always be floating and signed
		if( static_cast< RangeType >( val ) < lower )
			return true;
		else if( static_cast< RangeType >( val ) > upper )
			return true;

		return false;
	}
};

statTimer&
statTimer::getInstance( )
{
	static	statTimer	timer;
	return	timer;
}

statTimer::statTimer( ): nEvents( 0 ), nSamples( 0 ), normalize( true )
{
#if defined( _WIN32 )
	//	OS call to get ticks per second2
	::QueryPerformanceFrequency( reinterpret_cast< LARGE_INTEGER* >( &clkFrequency ) );
#else
	res.tv_sec	= 0;
	res.tv_nsec	= 0;
	clkFrequency 	= 0;

	//	clock_getres() return 0 for success
	//	If the function fails (monotonic clock not supported), we default to a lower resolution timer
//	if( ::clock_getres( CLOCK_MONOTONIC, &res ) )
	{
		clkFrequency = 1000000;
	}
//	else
//	{
//	    // Turn time into frequency
//		clkFrequency = res.tv_nsec * 1000000000;
//	}

#endif
}

statTimer::~statTimer( )
{}

void
statTimer::Clear( )
{
	labelID.clear( );
	clkStart.clear( );
	clkTicks.clear( );
}

void
statTimer::Reset( )
{
	if( nEvents == 0 || nSamples == 0 )
		throw	std::runtime_error( "StatisticalTimer::Reserve( ) was not called before Reset( )" );

	clkStart.clear( );
	clkTicks.clear( );

	clkStart.resize( nEvents );
	clkTicks.resize( nEvents );

	for( statTimer::uint	i = 0; i < nEvents; ++i )
	{
		clkTicks.at( i ).reserve( nSamples );
	}

	return;
}

//	The caller can pre-allocate memory, to improve performance.  
//	nEvents is an approximate value for how many seperate events the caller will think 
//	they will need, and nSamples is a hint on how many samples we think we will take
//	per event
void
statTimer::Reserve( size_t nEvents, size_t nSamples )
{
	this->nEvents	= std::max< size_t >( 1, nEvents );
	this->nSamples	= std::max< size_t >( 1, nSamples );

	Clear( );
	labelID.reserve( nEvents );

	clkStart.resize( nEvents );
	clkTicks.resize( nEvents );

	for( statTimer::uint i = 0; i < nEvents; ++i )
	{
		clkTicks.at( i ).reserve( nSamples );
	}
}

void
statTimer::convert2seconds( bool norm )
{
	normalize = norm;
}

void
statTimer::Start( size_t id )
{
#if defined( _WIN32 )
	::QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>( &clkStart.at( id ) ) );
#else
	if( clkFrequency )
	{
		struct timeval s;
		gettimeofday( &s, 0 );
		clkStart.at( id ) = (cl_ulong)s.tv_sec * 1000000 + (cl_ulong)s.tv_usec;
	}
	else
	{
		
	}
#endif
}

void
statTimer::Stop( size_t id )
{
	statTimer::ulong n;

#if defined( _WIN32 )
	::QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>( &n ) );
#else
	struct timeval s;
	gettimeofday( &s, 0 );
	n = (cl_ulong)s.tv_sec * 1000000 + (cl_ulong)s.tv_usec;
#endif

	n		-= clkStart.at( id );
	clkStart.at( id )	= 0;
	AddSample( id, n );
}

void
statTimer::AddSample( const size_t id, const statTimer::ulong n )
{
	clkTicks.at( id ).push_back( n );
}

//	This function's purpose is to provide a mapping from a 'friendly' human readable text string
//	to an index into internal data structures.
size_t
statTimer::getUniqueID( const tstring& label, statTimer::uint groupID )
{
	//	I expect labelID will hardly ever grow beyond 30, so it's not of any use
	//	to keep this sorted and do a binary search

	labelPair	sItem	= std::make_pair( label, groupID );

	stringVector::iterator	iter;
	iter	= std::find( labelID.begin(), labelID.end(), sItem );

	if( iter != labelID.end( ) )
		return	std::distance( labelID.begin( ), iter );

	labelID.push_back( sItem );

	return	labelID.size( ) - 1;

}

double
statTimer::getMean( size_t id ) const
{
	if( clkTicks.empty( ) )
		return	0;

	size_t	N	= clkTicks.at( id ).size( );

#if _MSC_VER <  1800   

	Accumulator< statTimer::ulong > sum = std::for_each( clkTicks.at( id ).begin(), clkTicks.at( id ).end(), Accumulator< statTimer::ulong > () );

#else	
	Accumulator< statTimer::ulong > sum;
	for (std::vector<ulong>::const_iterator f1 = clkTicks.at(id).begin(); f1 != clkTicks.at(id).end(); f1++)
	{
		sum(*f1);
	}
#endif
	return	static_cast<double>(sum.acc) / N;

}


double
statTimer::getVariance( size_t id ) const
{
	if( clkTicks.empty( ) )
		return	0;

	double	mean	= getMean( id );

	size_t	N	= clkTicks.at( id ).size( );
	double	sum	= 0;

	for( statTimer::uint i = 0; i < N; ++i )
	{
		double	diff	= clkTicks.at( id ).at( i ) - mean;
		diff	*= diff;
		sum		+= diff;
	}

	return	 sum / N;
}

double
statTimer::getStdDev( size_t id ) const
{
	double	variance	= getVariance( id );

	return	sqrt( variance );
}

double
statTimer::getAverageTime( size_t id ) const
{
	if( normalize )
		return getMean( id ) / clkFrequency;
	else
		return getMean( id );
}

double
statTimer::getMinimumTime( size_t id ) const
{
	clkVector::const_iterator iter	= std::min_element( clkTicks.at( id ).begin( ), clkTicks.at( id ).end( ) );

	if( iter != clkTicks.at( id ).end( ) )
	{
		if( normalize )
			return static_cast<double>( *iter ) / clkFrequency;
		else
			return static_cast<double>( *iter );
	}
	else
		return	0;
}

size_t
statTimer::pruneOutliers( size_t id , double multiple )
{
	if( clkTicks.empty( ) )
		return	0;

	double	mean	= getMean( id );
	double	stdDev	= getStdDev( id );

	clkVector&	clks = clkTicks.at( id );

	//	Look on p. 379, "The C++ Standard Library"
	//	std::remove_if does not actually erase, it only copies elements, it returns new 'logical' end
	clkVector::iterator	newEnd	= std::remove_if( clks.begin( ), clks.end( ), PruneRange< double, statTimer::ulong >( mean, multiple*stdDev ) );

	size_t dist = std::distance( newEnd, clks.end( ) );

	if( dist != 0 )
		clks.erase( newEnd, clks.end( ) );

	assert( dist < std::numeric_limits< statTimer::uint >::max( ) );

	return dist;
}

size_t
statTimer::pruneOutliers( double multiple )
{
	size_t	tCount	= 0;

	for( size_t l = 0; l < labelID.size( ); ++l )
	{
		size_t lCount	= pruneOutliers( l , multiple );
//		tout << _T( "[StatisticalTimer]: Pruned " ) << lCount << _T( " samples from " ) << labelID[l].first << std::endl;
		tCount += lCount;
	}

	tout << std::endl;

	return	tCount;
}

//	Defining an output print operator
tstream&
operator<<( tstream& os, const statTimer& st )
{
	if( st.clkTicks.empty( ) )
		return	os;

	std::ios::fmtflags bckup	= os.flags( );

	for( statTimer::uint l = 0; l < st.labelID.size( ); ++l )
	{
		statTimer::ulong min	= 0;
		statTimer::clkVector::const_iterator iter	= std::min_element( st.clkTicks.at( l ).begin( ), st.clkTicks.at( l ).end( ) );

		if( iter != st.clkTicks.at( l ).end( ) )
			min		= *iter;

		os << st.labelID[l].first << _T( ": " ) << st.labelID[l].second << std::fixed << std::endl;
		os << _T( "Min: " ) << min << std::endl;
		os << _T( "Mean: " ) << st.getMean( l ) << std::endl;
		os << _T( "StdDev: " ) << st.getStdDev( l ) << std::endl;
		os << _T( "AvgTime: " ) << st.getAverageTime( l ) << std::endl;
		os << _T( "MinTime: " ) << st.getMinimumTime( l ) << std::endl;

		//for( cl_uint	t = 0; t < st.clkTicks[l].size( ); ++t )
		//{
		//	os << st.clkTicks[l][t]<< ",";
		//}
		os << _T( "\n" ) << std::endl;

	}

	os.flags( bckup );

	return	os;
}

}
