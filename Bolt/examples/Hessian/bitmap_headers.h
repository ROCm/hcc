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

#pragma once
#ifndef bitmap_header_h__
#define bitmap_header_h__

#pragma pack(1)

typedef struct _RGBQUAD { // rgbq 
    unsigned char rgbBlue; 
    unsigned char rgbGreen; 
    unsigned char rgbRed; 
    unsigned char rgbReserved; 
} RGBQuad; 

typedef struct _BITMAPFILEHEADER { // bmfh 
	unsigned short	bfType;
    unsigned long   bfSize; 
    unsigned short  bfReserved1; 
    unsigned short  bfReserved2; 
    unsigned long   bfOffBits; 
} BitmapFileHeader; 

typedef struct _BITMAPINFOHEADER{ // bmih 
    unsigned long  biSize; 
    long		   biWidth; 
    long		   biHeight; 
    unsigned short biPlanes; 
    unsigned short biBitCount; 
    unsigned long  biCompression; 
    unsigned long  biSizeImage; 
    long		   biXPelsPerMeter; 
    long		   biYPelsPerMeter; 
    unsigned long  biClrUsed; 
    unsigned long  biClrImportant; 
} BitmapInfoHeader; 

#pragma pack()


#endif