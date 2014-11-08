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
    FOR BENCHMARKING sort API : sort Project Overview
========================================================================

Copy the python scripts to a folder along with the executable clBolt.Bench.sort.exe
Then Use the following commands
>>>>>>>
python measurePerformance.py --label sort_tbb_host --library=TBB --routine=sort --memory host -l 4096-67108864:x2 --tablefile sort_tbb_host.txt
python measurePerformance.py --label sort_tbb_device --library=TBB --routine=sort --memory device -l 4096-67108864:x2 --tablefile sort_tbb_device.txt
python measurePerformance.py --label sort_bolt_host --library=BOLT --routine=sort --memory host -l 4096-67108864:x2 --tablefile sort_bolt_host.txt
python measurePerformance.py --label sort_bolt_device --library=BOLT --routine=sort --memory device -l 4096-67108864:x2 --tablefile sort_bolt_device.txt
python measurePerformance.py --label sort_stl_host --library=STL --routine=sort --memory host -l 4096-67108864:x2 --tablefile sort_stl_host.txt
>>>>>>>
Run this command to plot the graph
python plotPerformance.py --y_axis_label "GBeys/sec" --title "sort Performance" --x_axis_scale log2 -d sort_tbb_host.txt -d sort_tbb_device.txt -d sort_bolt_host.txt -d sort_bolt_device.txt -d sort_stl_host.txt --outputfile sortPerfAll4096.pdf

/////////////////////////////////////////////////////////////////////////////
python plotPerformance.py --y_axis_label "GBeys/sec" --title "sort Performance" --x_axis_scale log2 -d sort_tbb_host.txt -d sort_bolt_device_orig_radix.txt -d sort_bolt_device_orig_bitonic.txt --outputfile sortPerfBitonic_Radix.pdf

/////////////////////////////////////////////////////////////////////////////
copy D:\Project\bolt\GitHub\Bolt-new\bin\vs2012-tbb-SuperBuild\Bolt-build\staging\Release\clbolt.bench.sort.exe .
python measurePerformance.py --label sort_bolt_device_radix_cont --library=BOLT --routine=sort --memory device -l 4096-67108864:x2 --tablefile sort_bolt_device_radix_cont.txt
python measurePerformance.py --label sort_bolt_host_bitonic_cont --library=BOLT --routine=sort --memory host -l 4096-67108864:x2 --tablefile sort_bolt_host_bitonic_cont.txt
python plotPerformance.py --y_axis_label "Unsigned int MKeys/sec" --title "Sort Performance" --x_axis_scale log2 -d sort_bolt_device.txt -d sort_tbb_host.txt -d stable_sort_tbb_host.txt --outputfile newSortPerfRadixUint.pdf
