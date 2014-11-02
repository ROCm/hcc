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

import os
import string
from os.path import join, getsize
from string import Template

license_template = "\
$header                                                                                     \n\
$comment   Copyright $YEAR Advanced Micro Devices, Inc.                                     \n\
$comment                                                                                    \n\
$comment   Licensed under the Apache License, Version $LICENCE_VERSION (the \"License\");   \n\
$comment   you may not use this file except in compliance with the License.                 \n\
$comment   You may obtain a copy of the License at                                          \n\
$comment                                                                                    \n\
$comment       http://www.apache.org/licenses/LICENSE-$LICENCE_VERSION                      \n\
$comment                                                                                    \n\
$comment   Unless required by applicable law or agreed to in writing, software              \n\
$comment   distributed under the License is distributed on an \"AS IS\" BASIS,              \n\
$comment   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         \n\
$comment   See the License for the specific language governing permissions and              \n\
$comment   limitations under the License.                                                   \n\n\
$footer                                                                                     \n\n"

##########################################################################
def add_licence_cmake(file_name):
    print "Adding Licence to cmake file " + file_name
    cmake_licence = Template(license_template).substitute(
                        header='############################################################################', 
                        comment='#',  
                        YEAR='2012',
                        LICENCE_VERSION='2.0',                        
                        footer='############################################################################');
    #print cmake_licence    

    orig_file = ''
    f_read = open(file_name, 'r') 
    lines = f_read.readlines()
    for line in lines:
        orig_file = orig_file + line    
    f_read.close()

    f_write = open(file_name, "w") 
    f_write.write(cmake_licence + orig_file)
    f_write.close()

##########################################################################
def add_licence_c(file_name):
    print "Adding Licence to C and CPP file " + file_name
    c_licence = Template(license_template).substitute(
                            header='/***************************************************************************', 
                            comment='*',  
                            YEAR='2012',                        
                            LICENCE_VERSION='2.0',
                            footer='***************************************************************************/');
    orig_file = ''
    f_read = open(file_name, 'r') 
    lines = f_read.readlines()
    for line in lines:
        orig_file = orig_file + line    
    f_read.close()

    f_write = open(file_name, "w") 
    f_write.write(c_licence + orig_file)
    f_write.close()    
    
##########################################################################
#MAIN
files_modified = 0
files_not_modified = 0
for root, dirs, files in os.walk('../'):
    for fname in files:
        file_name = '{0}/{1}'.format(root, fname)
        if file_name.endswith(".dox") or file_name.endswith(".inl") or file_name.endswith(".c") or file_name.endswith(".cpp") or file_name.endswith(".h") or file_name.endswith(".hpp") or file_name.endswith(".cl") or file_name.endswith(".h.in"): 
            files_modified += 1
            add_licence_c(file_name) 
        elif file_name.endswith(".cfg") or file_name.endswith(".txt") or file_name.endswith(".cmake") or file_name.endswith(".py") :
            files_modified += 1
            add_licence_cmake(file_name)
        else:    
            files_not_modified += 1
            print "Not Processing file " + file_name

print "Processed " 
print files_modified 
print "files"
print "Not Modified " 
print files_not_modified 
print "files" 
#END
##########################################################################


