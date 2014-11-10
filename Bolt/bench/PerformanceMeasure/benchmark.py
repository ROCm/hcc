'''
print("###IMPORTANT###: Copy the Bolt.Bench.Benchmark.exe to a separate folder along with all python scripts and run this batch script")  

print("# This will benchmark the Routines for DATA_TYPE which is a MACRO in the Benchmark.cpp file. Set it to the Appropriate\n\
\tdata type for which you wan to Benchmark. ie unsigned int, float, double etc.\n\
\tAnd build Benchmark project to get the Required Bolt.Bench.Benchmark.exe")
print("# This will create the some comparison graphs ")
print("# BUT Before Running this script make sure your system must have PYTHON installed ( python-2.7.3.msi )")
print("# Installed all these required useful package as well - \n\
          \t\t\tnumpy-1.8.0-win32-superpack-python2.7.exe\n\
		  \t\tscipy-0.13.1-win32-superpack-python2.7.exe\n\
 		  \t\twxPython2.8-win32-unicode-2.8.12.1-py27.exe\n\
 		  \t\tmatplotlib-1.3.1.win32-py2.7.exe\n\
 		  \t\tpython-tesseract_0.8-1.5.win32-py2.7.exe\n\
 		  \t\tpython-dateutil-2.2.win32-py2.7.exe\n\
 		  \t\tpyparsing-2.0.1.win32-py2.7.exe\n\
 		  \t\tsix-1.4.1.win32-py2.7.exe")
 
print("# In command-line-arguments -l a1-a1:x2  --> means for each iteration double the length of buffer w.r.t. previous length.");
print("                            -l a1-a2:a3  --> means for each iteration increase the buffer size by a3.");
'''
import os
import sys
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument("-r","--routine", help="routine name")
parser.add_argument("-l","--length", help="Length of buffer")
parser.add_argument("-i","--iterations", help="no. of iteration per routine")
parser.add_argument("-lib","--lib_option", help="Library to uses - (options: bolt or thrust)")
args = parser.parse_args()
if (args.iterations and args.length and args.routine and args.lib_option):

	txtpdf_files = glob.glob('*.txt*')
	##print txtpdf_files
	deleteFile = "no"
	#deleteFile = raw_input("Would you like to delete all .txt and .pdf files?(yes/no) ")
	if deleteFile == "yes":
		# need to loop through txtpdf_files and use os.remove
		length = len(txtpdf_files);
		while length > 0:
			os.remove(txtpdf_files[length-1])
			length = length -1;
	elif deleteFile == "no":
		print "\nCAVEAT: If .txt file of same routine exist, it will create new .txt file with some random number appended to it\
		\n\tWhich lead to wrong pdf file creation"
			
	#python commonPerformance.py --label reduce_bolt_device --library=BOLT --routine=reduce --memory device -l 1024-33554432:x2 --tablefile reduce_bolt_device.txt
	if sys.argv[8] == 'bolt':
		string = " --label "\
		"boltOCL_Tahiti7970 --library=BOLT --routine="\
		+sys.argv[2]+" --memory device -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_boltOCL_device.txt";
		os.system('python commonPerformance.py%s'%string)
		'''
		string = " --label "\
		+sys.argv[2]+"_boltOCL_host --library=BOLT --routine="\
		+sys.argv[2]+" --memory host -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_boltOCL_host.txt";
		os.system('python commonPerformance.py%s'%string)'''
		
		string = " --label "\
		"tbb --library=TBB --routine="\
		+sys.argv[2]+" --memory host -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_tbb_host.txt";
		os.system('python commonPerformance.py%s'%string)
		
		string = " --label "\
		+sys.argv[2]+"_stl_host --library=STL --routine="\
		+sys.argv[2]+" --memory host -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_stl_host.txt";
		os.system('python commonPerformance.py%s'%string)
	
		#python plotPerformance.py --y_axis_label "MKeys/sec" --title "sort Performance" --x_axis_scale log2 -d sort_bolt_device.txt -d sort_bolt_host.txt -d sort_tbb_host.txt -d sort_stl_host.txt --outputfile sortPerf_boltDevice_boltHost_tbbHost_stl_Holst.pdf	
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_boltOCL_device.txt -d "\
		+sys.argv[2]+"_tbb_host.txt -d "\
		+sys.argv[2]+"_stl_host.txt --outputfile "\
		+sys.argv[2]+"Perf_boltOCLDevice'_tbbHost_stl_Host.pdf";
		os.system('python plotPerformance.py%s'%string)
		
		print("generate the graph for Bolt_With_DeviceVector VS Tbb_With_HostVector")
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_boltOCL_device.txt -d "\
		+sys.argv[2]+"_tbb_host.txt --outputfile "\
		+sys.argv[2]+"Perf_boltOCLDevice_tbbHost.pdf";
		os.system('python plotPerformance.py%s'%string)
		
		print("generate the individual graph for Bolt_With_DeviceVector")
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_boltOCL_device.txt --outputfile "\
		+sys.argv[2]+"Perf_boltOCLDevice.pdf";
		os.system('python plotPerformance.py%s'%string)
	elif sys.argv[8] == 'thrust':
		string = " --label "\
		"thrust_GTX680 --library=THRUST --routine="\
		+sys.argv[2]+" --memory device -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_thrust_device.txt";
		os.system('python commonPerformance.py%s'%string)
		'''
		string = " --label "\
		+sys.argv[2]+"_thrust_host --library=THRUST --routine="\
		+sys.argv[2]+" --memory host -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_thrust_host.txt";
		os.system('python commonPerformance.py%s'%string)
				
			
		#python plotPerformance.py --y_axis_label "MKeys/sec" --title "sort Performance" --x_axis_scale log2 -d sort_bolt_device.txt -d sort_bolt_host.txt -d sort_tbb_host.txt -d sort_stl_host.txt --outputfile sortPerf_boltDevice_boltHost_tbbHost_stl_Holst.pdf	
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_thrust_device.txt -d "\
		+sys.argv[2]+"_thrust_host.txt --outputfile "\
		+sys.argv[2]+"Perf_thrustDevice_thrustHost.pdf";
		os.system('python plotPerformance.py%s'%string)'''
		
		print("generate the individual graph for thrust_With_DeviceVector")
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_thrust_device.txt --outputfile "\
		+sys.argv[2]+"Perf_thrustDevice.pdf";
		os.system('python plotPerformance.py%s'%string)
	else:
		string = " --label "\
		"BoltAmp_tahiti7970 --library=AMP --routine="\
		+sys.argv[2]+" --memory device -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_Boltamp_device.txt";
		os.system('python commonPerformance.py%s'%string)
		'''
		string = " --label "\
		+sys.argv[2]+"_amp_host --library=AMP --routine="\
		+sys.argv[2]+" --memory host -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_amp_host.txt";
		os.system('python commonPerformance.py%s'%string)'''
		
		string = " --label "\
		+sys.argv[2]+"_tbb_host --library=TBB --routine="\
		+sys.argv[2]+" --memory host -l "\
		+sys.argv[4]+" -i "\
		+sys.argv[6]+" --tablefile "\
		+sys.argv[2]+"_tbb_host.txt";
		os.system('python commonPerformance.py%s'%string)				
			
		#python plotPerformance.py --y_axis_label "MKeys/sec" --title "sort Performance" --x_axis_scale log2 -d sort_bolt_device.txt -d sort_bolt_host.txt -d sort_tbb_host.txt -d sort_stl_host.txt --outputfile sortPerf_boltDevice_boltHost_tbbHost_stl_Holst.pdf	
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_Boltamp_device.txt -d "\
		+sys.argv[2]+"_tbb_host.txt --outputfile "\
		+sys.argv[2]+"Perf_BoltampDevice_tbbHost.pdf";
		os.system('python plotPerformance.py%s'%string)
		
		print("generate the individual graph for amp_With_DeviceVector")
		string = " --y_axis_label \"MKeys/sec\" --title \""\
		+sys.argv[2]+" Performance\" --x_axis_scale log2 -d "\
		+sys.argv[2]+"_Boltamp_device.txt --outputfile "\
		+sys.argv[2]+"Perf_ampDevice.pdf";
		os.system('python plotPerformance.py%s'%string)
		
		
		
else:
	print "Too few arguments: Use -h or --help as input argument to HELP you"


















