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

/* COMPILE Command: "cl /EHsc StringifyKernels.cpp"
 * The generated exe is StringifyKernels.exe
 * Usage : StringifyKernels.exe <source-kernel-file> <destination-kernel-file> <kernel-string-name>
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;

//  This is used for debug; prints a string to stdout
void printString( const std::string& stringRef )
{
    std::cout << "String value: " << stringRef << std::endl;
};

//  This is the core logic of this program; it executes for each input file name
//  Opens a file a .cl file for reading, reads the contents and writes an .hpp file to output.
void writeHeaderFile( const std::string& inputFileName, const std::string& destDir )
{
    std::string headerPath;
    std::string baseName;

    std::string::size_type posPeriod = inputFileName.find_last_of( "." );
    std::string::size_type posSlash = inputFileName.find_last_of( "/\\" );

    if( posPeriod != std::string::npos )
    {
        // std::string::npos == -1, so adding 1 makes it index 0
        std::string::size_type strLength = posPeriod - (posSlash+1);
        baseName = inputFileName.substr( posSlash + 1, strLength );

        headerPath = destDir + baseName + ".hpp";
    }
    else
    {
        std::cerr << "Input filenames must have an extention defined; skipping file processing" << std::endl;
        return;
    }

    std::cout << "Input file: " << inputFileName << std::endl;
    std::cout << "Output path: " << headerPath << std::endl;

    //Open the Kernel file
    std::ifstream f_kernel( inputFileName, std::fstream::in );
    std::ofstream f_dest( headerPath, std::fstream::out );

    //    std::string fileContent( (std::istreambuf_iterator<char>( f_kernel ) ), std::istreambuf_iterator<char>( ) );

    if( f_kernel.is_open( ) )
    {
        //  We use the STRINGIFY_CODE macro to encode the kernel code into a string; this provides several benefits:
        //  1) multi-line strings keeping syntax highlighting, with minimal per-line string mangling
        //  2) properly handles single line c++ comments //
        //  3) C macro preprocessor strips out all comments from the text
#if defined(_WIN32)
        const std::string startLine = "#include <string>\n#include \"bolt/cl/clcode.h\"\n\n const std::string bolt::cl::" + baseName + " = STRINGIFY_CODE(";
#else
        const std::string startLine = "#include <string>\n#include \"bolt/cl/clcode.h\"\n\n const std::string bolt::cl::" + baseName + " = \"\\";
#endif
        //const std::string startLine = "#include <string>\n#include \"bolt/cl/clcode.h\"\n\n const char* const bolt::cl::" + baseName + " = STRINGIFY_CODE(";
        f_dest << startLine << std::endl;

        //  We have to emit the text files line by line, because the OpenCL compiler is dependent on newlines
        std::string line;
        while( getline( f_kernel, line ) )
        {
            //  Debug code to watch string substitution
            //std::string substLine = boost::replace_all_copy( line, "\\", "\\\\" );
            //std::string substLine = boost::replace_all_copy( line, "\"", "\\\"" );
            //std::cout << line << std::endl;
            //std::cout << substLine << std::endl << std::endl;

            //  Escape the \ and " characters, except for printf statements
#if defined(_WIN32)
            if( line.find( "printf" ) == std::string::npos )
#endif
            {
                boost::replace_all( line, "\\", "\\\\" );
                boost::replace_all( line, "\"", "\\\"" );
            }

            //  For every line of code, we append a '\n' that will be preserved in the string passed into the ::clBuildProgram API
            //  This makes debugging kernels in debuggers easier
#if defined(_WIN32)
            f_dest << line << " \\n" << std::endl;
#else
            f_dest << line << " \\n\\" << std::endl;
#endif
        }

#if defined(_WIN32)
        const std::string endLine = ");";
#else
        const std::string endLine = "\";";
#endif
        f_dest << endLine;
    }
    else
    {
        std::cerr << "Failed to open the specified file " << inputFileName << std::endl;
        return;
    }
};

int main( int argc, char *argv[] )
{
    std::string destDir;
    std::vector< std::string > kernelFiles;

    try
    {
        // Declare supported options below, describe what they do
        po::options_description desc( "StringifyKernels command line options" );
        desc.add_options()
            ( "help,h",			"produces this help message" )
            ( "destinationDir,d", po::value< std::string >( &destDir ), "Destination directory to write output files" )
            ( "kernelFiles,k", po::value< std::vector< std::string > >( &kernelFiles ), "Input .cl kernel files to be transformed; can specify multiple" )
            ;

        //  All positional options (un-named) should be interpreted as kernelFiles
        po::positional_options_description p;
        p.add("kernelFiles", -1);

        po::variables_map vm;
        po::store( po::command_line_parser( argc, argv ).options( desc ).positional( p ).run( ), vm );
        po::notify( vm );

        if( vm.count( "help" ) )
        {
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "kernelFiles" ) )
        {
            //std::for_each( kernelFiles.begin( ), kernelFiles.end( ), &printString );
        }
        else
        {
            std::cerr << "StringifyKernels requires files to process; use --help to browse command line options" << std::endl;
            return 1;
        }
    }
    catch( std::exception& e )
    {
        std::cout << "StringifyKernels parsing error reported:" << std::endl << e.what() << std::endl;
        return 1;
    }

    //  Main loop of the program
    std::for_each( kernelFiles.begin( ), kernelFiles.end( ), boost::bind( &writeHeaderFile, _1, destDir ) );
}
