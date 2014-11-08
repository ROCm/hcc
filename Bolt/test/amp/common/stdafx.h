/***************************************************************************                                                                                     
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.                                     
*                                                                                    
*   Licensed under the Apache License, Version 2.0 (the "License");   
*   you may not use this file except in compliance with the License.                 
*   You may obtain a copy of the License at                                          
*                                                                                    
*       http://www.apache.org/licenses/LICENSE-2.0                      
*                                                                                    
*   Unless required by applicable law or agreed to in writing, software              
*   distributed under the License is distributed on an "AS IS" BASIS,              
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         
*   See the License for the specific language governing permissions and              
*   limitations under the License.                                                   

***************************************************************************/                                                                                     

// stdafx.h : include file for standard system include files,
// or project-specific include files that frequently, but
// changed infrequently.
//

#pragma once

#define NOMINMAX
#include "targetver.h"
#ifdef UNICODE
#include <tchar.h>
#else
#include <bolt/unicode.h>
#endif
#include <numeric>
#include <limits>
#include <tuple>
#include <iterator>
#include <list>	// For debugging purposes, to prove that we can reject lists


// TODO: reference additional headers here that your program requires.
