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

//#include "stdafx.h"

#include "stdio.h"
#include <string.h>

#include "image_utils.h"


void writeNumber(unsigned char* data, int line, float num, int xoff, int yoff)
{
	static const unsigned char digits[9][13*6] = {
		{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{0,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
		{1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
		{1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0},
		{1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
		{1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
		{0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
		{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
	};

	char s[320];
	sprintf(s, "%+12.6f", num);
	int x_off = xoff;
	int y_off = yoff;

	int len = strlen(s);
	if(xoff + len*6 > line)
		return;

	for(int c = 0; c < (int)strlen(s); c++)
	{
		char sign = s[c];

		if(sign == ' ') continue;

		int doff;
		if( sign == '-' ) 
			doff = 10*6;
		else if( sign == '+' ) 
			doff = 11*6;
		else if( sign == '.' ) 
			doff = 12*6;
		else if( sign >= '0' && sign <= '9')
			doff = (sign - '0')*6;
		else 
			continue;

		for(int i = 0; i<9; i++)
			for(int j = 0; j<6; j++) {				
				if( digits[8-i][doff+j] )
					data[(y_off+i)*line + (x_off+j)] = 0;
				else
					data[(y_off+i)*line + (x_off+j)] = 255;
			}
			x_off += 6;
	}	
}


