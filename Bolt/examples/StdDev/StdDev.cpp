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

#include <bolt/unicode.h>

#include "bolt/cl/reduce.h"
#include "bolt/cl/transform_reduce.h"

#include <math.h>
#include <algorithm>
#include <iomanip>

BOLT_FUNCTOR( Variance< int >,
    template< typename T >
    struct Variance
    {
        T m_mean;

        Variance( T mean ): m_mean( mean ) {};
        T operator( )( const T& elem )
        {
            return (elem - m_mean) * (elem - m_mean);
        };
    };
);

int _tmain( int argc, _TCHAR* argv[ ] )
{
    const cl_uint vecSize = 1024;
    bolt::cl::device_vector< cl_int > boltInput( vecSize );
    
    std::cout << "\n\n Calculate Standard deviation \n";
    std::cout << "\n\nThis example calculates the standard deviation of input device_vector \n";
    std::cout << "with the BOLT APIS and STL. and displays the result. \n\n";
    //  Initialize random data in device_vector
    std::generate( boltInput.begin( ), boltInput.end( ), rand );

    //  Calculate standard deviation on the Bolt device
    cl_int boltSum = bolt::cl::reduce( boltInput.begin( ), boltInput.end( ), 0 );
    cl_int boltMean = boltSum / vecSize;

    cl_int boltVariance  = bolt::cl::transform_reduce( boltInput.begin( ), boltInput.end( ), Variance< cl_int >( boltMean ), 0, bolt::cl::plus< cl_int >( ) );
    cl_double boltStdDev = sqrt( static_cast< double >( boltVariance ) / vecSize );

    //  Calculate standard deviation with std algorithms (using device_vector!)
    cl_int stdSum = std::accumulate( boltInput.begin( ), boltInput.end( ), 0 );
    cl_int stdMean = stdSum / vecSize;

    std::transform( boltInput.begin( ), boltInput.end( ), boltInput.begin( ), Variance< cl_int >( stdMean ) );
    cl_uint stdVariance = std::accumulate( boltInput.begin( ), boltInput.end( ), 0 );
    cl_double stdStdDev = sqrt( static_cast< double >( stdVariance ) / vecSize );

    std::cout << std::setw( 40 ) << std::right << "Bolt Standard Deviation: " << boltStdDev << std::endl;
    std::cout << std::setw( 40 ) << std::right << "STD Standard Deviation: " << stdStdDev << std::endl;
    std::cout << "\nCOMPLETED. ...\n";
    return 0;
}
