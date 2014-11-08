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

template< typename iType,
          typename iIterType,
          typename mapType,
          typename mapIterType,
          typename stencilType, 
          typename stencilIterType,
          typename oType, 
          typename oIterType,
          typename Predicate >
__kernel
void scatterIfTemplate (
            global iType *input_naked,
            iIterType input,
            global mapType* map_naked,
            mapIterType map,
            global stencilType* stencil_naked,
            stencilIterType stencil,
            global oType* output_naked,
            oIterType output,
            const uint length,
            global Predicate* pred )
{
    typedef typename stencilIterType::value_type stencilValueType;
    typedef typename mapIterType::value_type mapValueType;

    int gid = get_global_id( 0 );
    if ( gid >= length ) return;

    input.init( input_naked );
    map.init( map_naked );
    stencil.init( stencil_naked );
    output.init( output_naked );
    
    mapValueType m = map[ gid ];
    stencilValueType s = stencil[ gid ];

    if ( (*pred)( s ) )
    {
        output [ m ] = input [ gid ] ;
    }
    
}


template< typename iType,
          typename iIterType,
          typename mapType,
          typename mapIterType,
          typename oType, 
          typename oIterType >
__kernel
void scatterTemplate (
            global iType *input_naked,
            iIterType input,
            global mapType* map_naked,
            mapIterType map,
            global oType* output_naked,
            oIterType output,
            const uint length )
{
    typedef typename mapIterType::value_type mapValueType;
    int gid = get_global_id( 0 );
    if ( gid >= length ) return;

    input.init( input_naked );
    map.init( map_naked );
    output.init( output_naked );

    // Store in registers
    mapValueType m = map[ gid ];

    output [ m ] = input [ gid ] ;

}