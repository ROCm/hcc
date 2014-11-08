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

#include "stdafx.h"

#include <bolt/amp/functional.h>
#include <bolt/amp/transform_reduce.h>
#include <bolt/unicode.h>
#include <bolt/countof.h>
#include <bolt/statisticalTimer.h>

const std::streamsize colWidth = 26;

void printAccelerator( unsigned int num, const concurrency::accelerator& dev )
{
	std::wcout << std::left << std::boolalpha;
	std::wcout << std::setw( colWidth ) << _T( "Device: " ) << num << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Description: " ) << dev.get_description( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Device Path: " ) << dev.get_device_path( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Version: " ) << dev.get_version( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Dedicated Memory: " ) << std::showpoint << dev.get_dedicated_memory( ) / 1024 << _T( " MB" ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Double support: " ) << dev.get_supports_double_precision( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Limited Double Support: " ) << dev.get_supports_limited_double_precision( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Emulated: " ) << dev.get_is_emulated( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Debug: " ) << dev.get_is_debug( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "Display: " ) << dev.get_has_display( ) << std::endl;

	concurrency::accelerator_view av = dev.get_default_view( );
	std::wcout << std::setw( colWidth ) << _T( "Default accelerator view: " ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "    Version: " ) << av.get_version( ) << std::endl;
	std::wcout << std::setw( colWidth ) << _T( "    Debug: " ) << av.get_is_debug( ) << std::endl;

	concurrency::queuing_mode qm = av.get_queuing_mode( );
	bolt::tout << std::setw( colWidth ) << _T( "    Queueing mode: " );
	switch( qm )
	{
		case concurrency::queuing_mode_immediate:
			std::wcout << _T( "immediate" );
			break;
		case concurrency::queuing_mode_automatic:
			std::wcout << _T( "automatic" );
			break;
		default:
			std::wcout << _T( "unknown" );
			break;
	}
	std::wcout << std::endl;


	std::wcout << std::internal << std::noboolalpha << std::endl;
}

int _tmain( int argc, _TCHAR* argv[] )
{
	size_t length = 0;
	size_t iDevice = 0;
	size_t numLoops = 0;
	bool defaultDevice = true;

	try
	{
		// Declare the supported options.
		po::options_description desc( "AMP Transform Reduce command line options" );
		desc.add_options()
			( "help,h",			"produces this help message" )
			( "version,v",		"Print queryable version information from the Bolt AMP library" )
			( "ampInfo,i",		"Print queryable information of the AMP runtime" )
			( "device,d",		po::value< size_t >( &iDevice ), "Choose specific AMP device, otherwise system default (AMP choose)" )
			( "length,l",		po::value< size_t >( &length )->default_value( 4096 ), "Specify the length of scan array" )
			( "profile,p",		po::value< size_t >( &numLoops )->default_value( 1 ), "Time and report Scan speed GB/s (default: profiling off)" )
			;

		po::variables_map vm;
		po::store( po::parse_command_line( argc, argv, desc ), vm );
		po::notify( vm );

		if( vm.count( "version" ) )
		{
			//	TODO:  Query Bolt for its version information
			size_t libMajor, libMinor, libPatch;
			libMajor = 0;
			libMinor = 0;
			libPatch = 1;

			const int indent = countOf( "Bolt version: " );
			bolt::tout << std::left << std::setw( indent ) << _T( "Bolt version: " )
				<< libMajor << _T( "." )
				<< libMinor << _T( "." )
				<< libPatch << std::endl;
		}

		if( vm.count( "help" ) )
		{
			//	This needs to be 'cout' as program-options does not support wcout yet
			std::cout << desc << std::endl;
			return 0;
		}

		if( vm.count( "ampInfo" ) )
		{
			concurrency::accelerator default_acc;
			std::wcout << std::left;
			std::wcout << std::setw( colWidth ) << _T( "Default device: " ) << default_acc.description << std::endl;
			std::wcout << std::setw( colWidth ) << _T( "Default device path: " ) << default_acc.device_path << std::endl << std::endl;

			//std::for_each( allDevices.begin( ), allDevices.end( ), printAccelerator );
			std::vector< concurrency::accelerator > allDevices = concurrency::accelerator::get_all( );
			for( unsigned int i = 0; i < allDevices.size( ); ++i )
				printAccelerator( i, allDevices.at( i ) );

			return 0;
		}

		if( vm.count( "device" ) )
		{
			defaultDevice = false;
		}

	}
	catch( std::exception& e )
	{
		bolt::terr << _T( "Bolt AMP error reported:" ) << std::endl << e.what() << std::endl;
		return 1;
	}

//	bolt::control::getDefault( );
	std::vector< int > input( length, 1 );

	bolt::statTimer& myTimer = bolt::statTimer::getInstance( );
	myTimer.Reserve( 1, numLoops );

	size_t reduceId	= myTimer.getUniqueID( _T( "reduce" ), 0 );

	for( unsigned i = 0; i < numLoops; ++i )
	{
		myTimer.Start( reduceId );
		int res = bolt::amp::transform_reduce( input.begin( ), input.end( ),bolt::amp::square<int>(), 0, bolt::amp::plus<int>() );
		myTimer.Stop( reduceId );
	}

	//	Remove all timings that are outside of 2 stddev (keep 65% of samples); we ignore outliers to get a more consistent result
	size_t pruned = myTimer.pruneOutliers( 1.0 );
	double scanTime = myTimer.getAverageTime( reduceId );
	double scanGB = ( input.size( ) * sizeof( int ) ) / (1024.0 * 1024.0 * 1024.0);

	bolt::tout << std::left;
	bolt::tout << std::setw( colWidth ) << _T( "Transform Reduce profile: " ) << _T( "[" ) << numLoops-pruned << _T( "] samples" ) << std::endl;
	bolt::tout << std::setw( colWidth ) << _T( "    Size (GB): " ) << scanGB << std::endl;
	bolt::tout << std::setw( colWidth ) << _T( "    Time (s): " ) << scanTime << std::endl;
	bolt::tout << std::setw( colWidth ) << _T( "    Speed (GB/s): " ) << scanGB / scanTime << std::endl;
	bolt::tout << std::endl;

//	bolt::tout << myTimer;

	return 0;
}
