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

========================================================================
    FOR BENCHMARKING REDUCE API : Reduce Project Overview
========================================================================

Copy the puthon scripts to a folder along with the executable clBolt.Bench.Reduce.exe
Then Use the following commands
>>>>>>>
python measurePerformance.py --label reduce_tbb_host --library=TBB --routine=reduce --memory host -l 4096-67108864:x2 --tablefile reduce_tbb_host.txt
python measurePerformance.py --label reduce_tbb_device --library=TBB --routine=reduce --memory device -l 4096-67108864:x2 --tablefile reduce_tbb_device.txt
python measurePerformance.py --label reduce_bolt_host --library=BOLT --routine=reduce --memory host -l 4096-67108864:x2 --tablefile reduce_bolt_host.txt
python measurePerformance.py --label reduce_bolt_device --library=BOLT --routine=reduce --memory device -l 4096-67108864:x2 --tablefile reduce_bolt_device.txt
python measurePerformance.py --label reduce_stl_host --library=STL --routine=reduce --memory host -l 4096-67108864:x2 --tablefile reduce_stl_host.txt
>>>>>>>
Run this command to plot the graph
python plotPerformance.py --y_axis_label "MKeys/sec" --title "Reduce Performance" --x_axis_scale log2 -d reduce_tbb_host.txt -d reduce_tbb_device.txt -d reduce_bolt_host.txt -d reduce_bolt_device.txt -d reduce_stl_host.txt --outputfile reducePerfAll4096.pdf

/////////////////////////////////////////////////////////////////////////////
