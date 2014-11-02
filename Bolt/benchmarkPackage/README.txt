This is the Independent Benchmark project.
It Works for both Windows and Linux Machine(only need to change the value of MACRO named Bolt_Benchmark in benchmark/Benchmark.cpp file
                                       ( = 1 means for Bolt on windows, = 0 means for Thrust on Linux))

To do cmake :
      1) Set the Environment Variable BOLT_ROOT -  to  Bolt Package location i.e.(C:\Users\xxxxxx\Desktop\Benchmark\Bolt-1.1-VS2012\Bolt-1.1-VS2012)
			where Bolt-x.x-VS2012 is Package, we have downloaded from http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/bolt-c-template-library/
									                        or  from https://github.com/HSA-Libraries 
															
      2) Set the TBB_ROOT - to the TBB package location i.e.(C:\Users\xxxxxx\Downloads\tbb41_20130314oss_win\tbb41_20130314oss)
	  3) Install the AMDAPPSDK - amd OpenCL Package and make sure AMDAPPSDKROOT is set to C:\Program Files (x86)\AMD APP in windows,
	                                                              OPENCL_ROOT is set to /opt/AMDAPP/ in Linux
	  
After cmake :
      1) cmake gives a Build directory, go to that directory and build the solution to get executable file i.e.(Bolt.Bench.Benchmark.exe)
	  2) To benchamrk - run the executable file with the appropriate command line arguments (use: Bolt.Bench.Benchmark.exe --help for more options)
