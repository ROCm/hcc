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

boltProject:

This directory contains the "Bolt" project.  
  * Bolt is a C++ template function library similar in spirit to the STL 
    "algorithm" header.  It includes a hybrid of functions from STL (ie 
    transform, reduce, sort, etc) as well as functions optimized for use 
    on HSA APUS (ie pipeline, parallel_do).
  * Bolt is optimized for HSA APUs and leverages features including 
    Shared Virual Memory (smooth programming model and performance), 
    GPU and CPU, and the advanced queueing features of HSA GPUs.
  * Bolt currently runs on C++AMP and eventually will run on OpenCL. 
    To run the samples, you need the "Developer Preview" version of 
    Visual Studio Dev11, available via MSDN or here:
    http://msdn.microsoft.com/en-us/vstudio/hh127353
    

* Directory Structure:
    * /bolt : Header files for bolt template function library
    * /tests : Projects that demonstrate simple use cases and functional tests.
