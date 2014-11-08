############################################################################                                                                                     
#   Copyright 2012 - 2013 Advanced Micro Devices, Inc.                                     
#                                                                                    
#   Licensed under the Apache License, Version 2.0 (the "License");   
#   you may not use this file except in compliance with the License.                 
#   You may obtain a copy of the License at                                          
#                                                                                    
#       http://www.apache.org/licenses/LICENSE-2.0                      
#                                                                                    
#   Unless required by applicable law or agreed to in writing, software              
#   distributed under the License is distributed on an "AS IS" BASIS,              
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         
#   See the License for the specific language governing permissions and              
#   limitations under the License.                                                   

############################################################################                                                                                     

Hessian is an example ISV kernel from MotionDSP which is part of their motion estimation pipeline.  For this project, we implemented the same kernel in several different 
programming models so that we could compare the lines-of-code, and the performance for the different implementations.  

The program uses CMAKE.  The root CMAKE file is in examples/CMakeLists.txt.  Point the
cmake gui at examples/CMakeLists.txt to create a build directory.  The build has
only been tested in Visual Studio (and likely requires VS because it includes
C++ AMP).

Installation requires:
  * Visual Studio 2011 (For C++AMP)
  * AMD OpenCL SDK 2.6-beta or 2.7 (including the support for C++ kernel features including templates).  Set the env variable AMDAPPSDKROOT (ie " C:\Program Files (x86)\AMD APP\").
  * TBB (we tested with TBB 4.0).  Set the env variable TBBROOT (ie "C:\Program Files (x86)\TBB\tbb40_297oss"


Running the program:
    * The resulting executable is placed in examples/build/staging/Release.
    * Several commandline switches are supported.  The executable does not print a help
      message - see the file "hessian.cpp" for a list of supported options.
      Also see the file "hessian.h" for a decoder ring for the valuable "-mode=" option.

    * Some useful command lines:
       ./examples.Hessian.exe -mode=0x8000 -iters=1000  -ampAccelerator=1  -computeUnits=5 -wgPerComputeUnit=12 -zeroCopy=0



