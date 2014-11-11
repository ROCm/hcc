#Copy the cl.Bolt.benchmark.exe to a seprate folder along with the python scripts and run this batch script
python commonPerformance.py --label sort_bolt_device --library=BOLT --routine=sort --memory device -l 4096-33554432:x2 --tablefile sort_bolt_device.txt
python commonPerformance.py --label sort_bolt_host   --library=BOLT --routine=sort --memory host   -l 4096-33554432:x2 --tablefile sort_bolt_host.txt
python commonPerformance.py --label sort_tbb_host    --library=TBB  --routine=sort --memory host   -l 4096-33554432:x2 --tablefile sort_tbb_host.txt
python plotPerformance.py --y_axis_label "MKeys/sec" --title "Sort Performance" --x_axis_scale log2 -d sort_bolt_device.txt -d sort_bolt_host.txt -d sort_tbb_host.txt --outputfile sortPerformance.pdf

python commonPerformance.py --label stablesort_bolt_device --library=BOLT --routine=stablesort --memory device -l 4096-33554432:x2 --tablefile stablesort_bolt_device.txt
python commonPerformance.py --label stablesort_bolt_host   --library=BOLT --routine=stablesort --memory host   -l 4096-33554432:x2 --tablefile stablesort_bolt_host.txt
python commonPerformance.py --label stablesort_tbb_host    --library=TBB  --routine=stablesort --memory host   -l 4096-33554432:x2 --tablefile stablesort_tbb_host.txt
python plotPerformance.py --y_axis_label "MKeys/sec" --title "Stable Sort Performance" --x_axis_scale log2 -d stablesort_bolt_device.txt -d stablesort_bolt_host.txt -d stablesort_tbb_host.txt --outputfile stablesortPerformance.pdf

python commonPerformance.py --label inclusivescan_bolt_device --library=BOLT --routine=inclusivescan --memory device -l 4096-33554432:x2 --tablefile inclusivescan_bolt_device.txt
python commonPerformance.py --label inclusivescan_bolt_host   --library=BOLT --routine=inclusivescan --memory host   -l 4096-33554432:x2 --tablefile inclusivescan_bolt_host.txt
python commonPerformance.py --label inclusivescan_tbb_host    --library=TBB  --routine=inclusivescan --memory host   -l 4096-33554432:x2 --tablefile inclusivescan_tbb_host.txt
python plotPerformance.py --y_axis_label "MKeys/sec" --title "Inclusive Scan Performance" --x_axis_scale log2 -d inclusivescan_bolt_device.txt -d inclusivescan_bolt_host.txt -d inclusivescan_tbb_host.txt --outputfile inclusivescanPerformance.pdf