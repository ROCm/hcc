'''
-lib bolt -> to run the Bolt Benchmark 
-lib thrust -> to run the thrust Benchmark 
'''
import os
os.system("python benchmark.py -r reduce -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r reducebykey -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r transformreduce -l 4096-33554432:x2 -i 1000 -lib bolt")

os.system("python benchmark.py -r sort -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r sortbykey -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r stablesort -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r stablesortbykey -l 4096-33554432:x2 -i 1000 -lib bolt")

os.system("python benchmark.py -r scan -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r scanbykey -l 4096-33554432:x2 -i 1000 -lib bolt")
os.system("python benchmark.py -r transformscan -l 4096-33554432:x2 -i 1000 -lib bolt")

os.system("python benchmark.py -r unarytransform -l 4096-33554432:x2 -i 1000 -lib bolt")
