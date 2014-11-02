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

import argparse
import subprocess
import itertools
import re#gex
import os
from threading import Thread
from datetime import datetime

# Our own code import statements
import errorHandler
from performanceTesting import executable, tableHeader, TestCombination

LOG_FILE_NAME = {
                'Bolt':{'ER':'boltExecuteRunsLog.txt',#execute runs
                        'MP':'boltMeasurePerfLog.txt'} #measure performance
                }

LOG_FILE_LOC = {
                'Bolt':'perflog'
               }

TIMOUT_VAL = 900  #In seconds

def log(filename, txt):
    with open(filename, 'a') as f:
        f.write(datetime.now().ctime()+'# '+txt+'\n')
        
IAM = 'Bolt'

precisionvalues = ['single', 'double']
libraryvalues = ['STL','BOLT','TBB', 'null']
memoryvalues = ['device','host', 'null']
routinevalues = ['copy','scan','sort', 'stablesort', 'transform', 'reduce','null']
backEndValues = ['cl','amp']

parser = argparse.ArgumentParser(description='Measure performance of a Bolt library')
parser.add_argument('--device',
    dest='device', default='default',
    help='device(s) to run on; may be a comma-delimited list. choices are index values as reported by the library. (default: AMP default)')
parser.add_argument('-l', '--lengthx',
    dest='lengthx', default='1',
    help='length(s) of x to test; must be factors of 1, 2, 3, or 5 with clAmdFft; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,2048-32768 (default 1)')

parser.add_argument('-r', '--precision',
    dest='precision', default='single',
    help='may enter multiple in a comma-delimited list. choices are ' + str(precisionvalues) + '. (default single)')
parser.add_argument('--library',

    dest='library', default='null', choices=libraryvalues,
    help='indicates the library to use for testing on this run')
parser.add_argument('--routine',
    dest='routine', default='null', choices=routinevalues,
    help='indicates the routine to use for testing on this run')
parser.add_argument('--memory',
    dest='memory', default='null', choices=memoryvalues,
    help='indicates the memory subsystem to choose from')
parser.add_argument('--label',
    dest='label', default=None,
    help='a label to be associated with all transforms performed in this run. if LABEL includes any spaces, it must be in \"double quotes\". note that the label is not saved to an .ini file. e.g., --label cayman may indicate that a test was performed on a cayman card or --label \"Windows 32\" may indicate that the test was performed on Windows 32')

parser.add_argument('--createini',
    dest='createIniFilename', default=None,
    help='create an .ini file with the given name that saves the other parameters given at the command line, then quit. e.g., \'clAmdFft.performance.py -x 2048 --createini my_favorite_setup.ini\' will create an .ini file that will save the configuration for a 2048-datapoint 1D FFT.')
parser.add_argument('--ini',
    dest='iniFilename', default=None,
    help='use the parameters in the named .ini file instead of the command line parameters.')
parser.add_argument('--tablefile',
    dest='tableOutputFilename', default=None,
    help='save the results to a plaintext table with the file name indicated. this can be used with clAmdFft.plotPerformance.py to generate graphs of the data (default: table prints to screen)')
parser.add_argument('--test',
    dest='test', default=1,
    help='Algorithm used [1,2]  1:SORT_BOLT, 2:SORT_AMP_SHOC')
parser.add_argument('--backend',
    dest='backend', default='cl', choices=backEndValues,
    help='Which Bolt backend to use')

args = parser.parse_args()

lab = str(args.label)
subprocess.call('mkdir '+LOG_FILE_LOC[IAM], shell = True)
logfile = os.path.join(LOG_FILE_LOC[IAM], (lab+'-'+LOG_FILE_NAME[IAM]['ER']))

def printLog(txt):
    print txt
    log(logfile, txt)

printLog("=========================MEASURE PERFORMANCE START===========================")
printLog("Process id of Measure Performance:"+str(os.getpid()))

currCommandProcess = None

printLog('Executing measure performance for label: '+str(lab))

#Spawns a separate thread to execute the library command and wait for that thread to complete
#This wait is of 900 seconds (15 minutes). If still the thread is alive then we kill the thread
def checkTimeOutPut(args):
    global currCommandProcess
    global stde
    global stdo
    stde = None
    stdo = None
    def executeCommand():
        global currCommandProcess
        global stdo
        global stde
        try:
            stdo, stde = currCommandProcess.communicate()
            printLog('stdout:\n'+str(stdo))
            printLog('stderr:\n'+str(stde))
        except:
            printLog("ERROR: UNKNOWN Exception - +checkWinTimeOutPut()::executeCommand()")

    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    thread = Thread(target=executeCommand)
    thread.start()
    thread.join(TIMOUT_VAL) #wait for the thread to complete 
    if thread.is_alive():
        printLog('ERROR: Killing the process - terminating thread because it is taking too much of time to execute')
        currCommandProcess.kill()
        printLog('ERROR: Timed out exception')
        raise errorHandler.ApplicationException(__file__, errorHandler.TIME_OUT)
    if stdo == "" or stdo==None:
        errCode = currCommandProcess.poll()
        printLog('ERROR: @@@@@Raising Called processor exception')
        raise subprocess.CalledProcessError(errCode, args, output=stde)
    return stdo

# don't try to create and use an .ini file at the same time (it will open a portal through which demons will emerge)
if args.iniFilename and args.createIniFilename:
    printLog('ERROR: --ini and --createini are mutually exclusive. Please choose only one.')
    quit()

#read in .ini parameters if --ini is used
if args.iniFilename != None:
    if not os.path.isfile(args.iniFilename):
        printLog("No file with the name \'{}\' exists. Please indicate another filename.".format(args.iniFilename))
        quit()
    
    ini = open(args.iniFilename, 'r')
    iniContents = ini.read()
    iniContents = iniContents.split(';')
    for i in range(0,len(iniContents)):
        line = iniContents.pop()
        line = line.partition(' ')
        parameter = line[0]
        value = line[2]
        value = value.replace('\'','').replace('[','').replace(']','').replace(' ','')
        
        if parameter == 'lengthx':
            args.lengthx = value
        elif parameter == 'device':
            args.device = value
        elif parameter == 'precision':
            args.precision = value
        else:
            printLog('{} corrupted. Please re-create a .ini file with the --createini flag.'.format(args.iniFilename))
            quit()

#create ini file if requested
if args.createIniFilename != None:
    printLog('Creating Ini files')
    if os.path.isfile(args.createIniFilename):
        printLog('A file with the name \'{}\' already exists. Please delete the file or choose another name.'.format(args.createIniFilename))
        quit()
    printLog('Creating Ini file:'+args.createIniFilename+'\n')
    ini = open(args.createIniFilename, 'w')
    ini.write('lengthx {} ;'.format(args.lengthx))
    ini.write('device {} ;'.format(args.device))
    ini.write('precision {} ;'.format(args.precision))
    printLog('Created Ini file:'+args.createIniFilename+'\n')
    printLog("=========================MEASURE PERFORMANCE START===========================\n")
    quit()


#split up comma-delimited lists
args.device = args.device.split(',')
args.lengthx = args.lengthx.split(',')
args.precision = args.precision.split(',')
args.library = str(args.library)
args.routine = str(args.routine)
printLog('Executing for label: '+str(args.label))
printLog('Executing for routine: '+str(args.routine))
printLog('Executing for library: '+str(args.library))
#check parameters for sanity

# check for valid values in precision
for n in args.precision:
    if n != 'single' and n != 'double':
        printLog('ERROR: invalid value for precision')
        quit()

if not os.path.isfile(executable(args.routine, args.backend)):
    printLog("ERROR: Could not find client named {0}".format(executable(args.routine)))
    quit()
   
#expand ranges
#example inputs
#16-65536:x2   for all powers of 2 
#16-65536:16   for multiples of 16 
class Range:
    def __init__(self, ranges, defaultStep='+1'):
        self.expanded = []
        for thisRange in ranges:
            if thisRange != 'max' and thisRange != 'adapt' :
                if thisRange.count(':'):
                    self._stepAmount = thisRange.split(':')[1]
                else:
                    self._stepAmount = defaultStep
                thisRange = thisRange.split(':')[0]

                if self._stepAmount.count('x'):
                    self._stepper = '_mult'
                    self._stepAmount = self._stepAmount.lstrip('+x')
                    self._stepAmount = int(self._stepAmount)
                else:
                    self._stepper = '_add'
                    self._stepAmount = self._stepAmount.lstrip('+x')
                    self._stepAmount = int(self._stepAmount)

                if thisRange.count('-'):
                    self.begin = int(thisRange.split('-')[0])
                    self.end = int(thisRange.split('-')[1])
                else:
                    self.begin = int(thisRange.split('-')[0])
                    self.end = int(thisRange.split('-')[0])
                self.current = self.begin

            _thisRangeExpanded = []
            if thisRange == 'max':
                self.expanded = self.expanded + ['max']
            elif thisRange == 'adapt':
                self.expanded = self.expanded + ['adapt']
            elif self.begin == 0 and self._stepper == '_mult':
                self.expanded = self.expanded + [0]
            else:
                while self.current <= self.end:
                    self.expanded = self.expanded + [self.current]
                    self._step()

            # now we want to uniquify and sort the expanded range
            self.expanded = list(set(self.expanded))
            self.expanded.sort()

    # advance current value to next
    def _step(self):
        getattr(self, self._stepper)()

    def _mult(self):
        self.current = self.current * self._stepAmount

    def _add(self):
        self.current = self.current + self._stepAmount


args.lengthx = Range(args.lengthx, '4096').expanded

#create final list of all transformations (with problem sizes and transform properties)
test_combinations = itertools.product( args.lengthx, args.device, args.precision )
test_combinations = list( itertools.islice(test_combinations, None) )
test_combinations = [TestCombination( params[0], params[1], params[2], args.label) for params in test_combinations]

#turn each test combination into a command, run the command, and then stash the gflops
result = [] # this is where we'll store the results for the table


#open output file and write the header

if args.tableOutputFilename == None:
   args.tableOutputFilename = 'results' + datetime.now().isoformat().replace(':','.') + '.txt'
else:
   if os.path.isfile(args.tableOutputFilename):
       oldname = args.tableOutputFilename
       args.tableOutputFilename = args.tableOutputFilename + datetime.now().isoformat().replace(':','.')
       message = 'A file with the name ' + oldname + ' already exists. Changing filename to ' + args.tableOutputFilename
       printLog(message)


printLog('table header---->'+ str(tableHeader))

table = open(args.tableOutputFilename, 'w')
table.write(tableHeader + '\n')
table.flush()

printLog('Total combinations =  '+str(len(test_combinations)))

vi = 0
#test_combinations = test_combinations[825:830]
for params in test_combinations:
    vi = vi+1

    printLog("")
    printLog('preparing command: '+ str(vi))    
    device = params.device
    lengthx = str(params.x)
    test = args.test
    
    if params.precision == 'single':
        precision = ''
    elif params.precision == 'double':
        precision = '--double'
    else:
        printLog('ERROR: invalid value for precision when assembling client command')

    #set up arguments here
    if params.device == 'default':
        arguments = [executable(args.routine, args.backend),
                     '-l', lengthx,
    #                     precision,
                     '-i', '50']
    else:
        arguments = [executable(args.routine, args.backend),
                     '-a', # Enumerate all devices in system, so that we can benchmark any device on command line
                     '-d', device,
                     '-l', lengthx,
    #                     precision,
                     '-i', '50']
    #if args.routine == 'sort':
    #    arguments.append( '-t' )
    #    arguments.append( str( test ) )
    #    arguments.append( '-m' )
    ###Set the Library Selection 
    if args.library == 'TBB':
        arguments.append( '-T' )
    if args.library == 'BOLT':
        arguments.append( '-B' )
    if args.library == 'STL':
        arguments.append( '-E' )
    ###Set the Memory Selection     
    if args.memory == 'host':
        arguments.append( '-S' )
    if args.memory == 'device':
        arguments.append( '-D' )
    writeline = True
    try:
        printLog('Executing Command: '+ str(arguments))
        output = checkTimeOutPut(arguments)
        output = output.split(os.linesep);
        printLog('Execution Successfull---------------\n')

    except errorHandler.ApplicationException as ae:
        writeline = False
        printLog('ERROR: Command is taking too much of time '+ae.message+'\n'+'Command: \n'+str(arguments))
        continue
    except subprocess.CalledProcessError as clientCrash:
        print 'Command execution failure--->'
        if clientCrash.output.count('CLFFT_INVALID_BUFFER_SIZE'):
            writeline = False
            printLog('Omitting line from table - problem is too large')
        else:
            writeline = False
            printLog('ERROR: client crash. Please report the following error message (with \'CLFFT_*\' error code, if given, and the parameters used to invoke clAmdFft.measurePerformance.py) to Geoff\n'+clientCrash.output+'\n')
            printLog('IN ORIGINAL WE CALL QUIT HERE - 1\n')
            continue

    for x in output:
        if x.count('CUDA driver version is insufficient'):
            printLog('ERROR: CUDA is not operational on this system. Check for a CUDA-capable card and up-to-date drivers')
            #dh.write('ERROR: CUDA is not operational on this system. Check for a CUDA-capable card and up-to-date drivers\n')
            quit()

        elif x.count('out of memory'):
            writeline = False
            printLog('ERROR: Omitting line from table - problem is too large')

    speedStr = 'GB/s'
    if args.routine == 'stablesort':
        speedStr = 'MKeys/s'
    if args.routine == 'sort':
        speedStr = 'MKeys/s'

    if writeline:
        try:
            output = itertools.ifilter( lambda x: x.count( speedStr ), output)
            output = list(itertools.islice(output, None))
            thisResult = re.search('\d+\.*\d*e*-*\d*$', output[-1])
            thisResult = float(thisResult.group(0))
            thisResult = (params.x, params.device, params.precision, params.label, thisResult)

            outputRow = ''
            for x in thisResult:
                outputRow = outputRow + str(x) + ','
            outputRow = outputRow.rstrip(',')
            table.write(outputRow + '\n')
            table.flush()
        except:
            printLog('ERROR: Exception occurs in GFLOP parsing')
    else:
        if(len(output) > 0):
            if output[0].find('nan') or output[0].find('inf'):
                printLog( 'WARNING: output from client was funky for this run. skipping table row')
            else:
                printLog('ERROR: output from client makes no sense')
                printLog(str(output[0]))
                printLog('IN ORIGINAL WE CALL QUIT HERE - 2\n')
        else:
            printLog('ERROR: output from client makes no sense')
            #quit()
printLog("=========================MEASURE PERFORMANCE ENDS===========================\n")
#
#"""
#print a pretty table
#"""
#if args.tableOutputFilename == None:
#   args.tableOutputFilename = 'results' + datetime.now().isoformat().replace(':','.') + '.txt'
#else:
#   if os.path.isfile(args.tableOutputFilename):
#       oldname = args.tableOutputFilename
#       args.tableOutputFilename = args.tableOutputFilename + datetime.now().isoformat().replace(':','.')
#       message = 'A file with the name ' + oldname + ' already exists. Changing filename to ' + args.tableOutputFilename
#       print message
#
#table = open(args.tableOutputFilename, 'w')
#table.write(tableHeader + '\n')
#for x in result:
#   row = ''
#   for y in x:
#       row = row + str(y) + ','
#   row = row[:-1] #chomp off the trailing comma
#   table.write(row + '\n')
