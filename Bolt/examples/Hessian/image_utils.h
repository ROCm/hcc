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

#include "matrix_utils.h"
#include "bitmap_headers.h"
#include <tchar.h>

extern void writeNumber(unsigned char* data, int line, float num, int xoff, int yoff);

template<typename T>
utils::Matrix<T>* BMP2matrix( const char* imFileName, utils::Range<T> range )
{
	std::fstream file;//(imFileName, std::ios_base::in|std::ios_base::binary);
	file.open(imFileName, std::ios_base::in|std::ios_base::binary);

	if (file.fail()) {
		TCHAR cCurrentPath[FILENAME_MAX];
		if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) {
			std::wcout <<  _T( "CWD=" ) << cCurrentPath << std::endl;
		};
		std::cout << "error: failed to open file '" << imFileName << std::endl;
		throw;
		//if(file.is_open() == false)
		return 0;
	};



	file.seekp(0, std::ios_base::end);
	int bufSize = (int)file.tellp() + 1;
	file.seekp(0, std::ios_base::beg);
	unsigned char* buf = new unsigned char[bufSize];

	file.read((char*)buf, bufSize);
	file.close();

	BitmapFileHeader* pBFH;
	BitmapInfoHeader* pBIH;
	RGBQuad* pPalette;
	unsigned char* pData;

	pBFH = (BitmapFileHeader*)buf;
	pBIH = (BitmapInfoHeader*) &(buf[ sizeof(BitmapFileHeader) ]);
	pPalette = (RGBQuad*) &(buf[sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader)]);
	pData = &(buf[pBFH->bfOffBits]);

	utils::Size imSize(pBIH->biHeight, pBIH->biWidth);

	int line; 
	int i, j;
	double pix, k;
	int index;
	utils::Matrix<T>* mx;

	k = (double)range.span()/256;

	switch(pBIH->biBitCount)
	{
	case 8:
		mx = new utils::Matrix<T>(imSize);

		line = (pBIH->biWidth+0x03) & (~0x03);

		for(i = 0; i < (int)imSize.height; i++)
		{
			for(j = 0; j < (int)imSize.width; j++) 
			{
				pix = (double)(pData[line*(imSize.height - i - 1) + j]) ;
				(*mx)(i, j) = (T)( k*pix ) + range._min;
			}
		}

		break;

	case 24:

		mx = new utils::Matrix<T>( utils::Size(imSize.height*3, imSize.width) );

		line = (pBIH->biWidth*3 + 0x03) & (~0x03);

		for(i = 0; i < (int)imSize.height; i++) 
		{
			index = 0;
			for( j = 0; j < (int)imSize.width; j++ ) 
			{
				// B
				pix = (double)(pData[line*(imSize.height - i - 1) + index]) ;
				(*mx)(i+2*imSize.height, j) = (T)( k*pix ) + range._min; 

				// G
				pix = (double)(pData[line*(imSize.height - i - 1) + index + 1]) ;
				(*mx)(i + imSize.height, j) = (T)( k*pix ) + range._min; 

				// R
				pix = (double)(pData[line*(imSize.height - i - 1) + index + 2]) ;
				(*mx)(i, j) = (T)( k*pix ) + range._min; 

				index += 3;
			}
		}
	}


	delete [] buf;
	return mx;
}

template<typename T>
void matrix2BMP( const utils::Matrix<T>& mx, utils::Range<T> range, const char* imFileName, bool jet = false, bool writeRange = false )
{
	BitmapFileHeader* pBFH;
	BitmapInfoHeader* pBIH;
	RGBQuad* pPalette;
	unsigned char* pData;
	int line; 
	unsigned int i, j; 

	if( mx.getSize().width % 4 )
		line = (mx.getSize().width/4 + 1)*4;
	else
		line = mx.getSize().width;

	int headers = sizeof(BitmapFileHeader)+
		sizeof(BitmapInfoHeader)+
		256*sizeof(RGBQuad);

	int imHeight =  (writeRange) ? (mx.getSize().height+32) : mx.getSize().height;

	unsigned char* buf = new unsigned char[headers + imHeight*line];

	memset( buf, 0, headers);
	memset( buf+headers, 255, imHeight*line ); 

	pBFH	 = (BitmapFileHeader*) buf ;
	pBIH	 = (BitmapInfoHeader*)( &(buf[sizeof(BitmapFileHeader)]) );
	pPalette = (RGBQuad*)(&(buf[sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader)])) ;
	pData	 = (unsigned char*) (&pPalette[256]);	

	pBFH->bfType	= 'MB';
	pBFH->bfSize	= headers + imHeight*line;
	pBFH->bfOffBits = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + 256*sizeof(RGBQuad);

	pBIH->biSize      = sizeof(BitmapInfoHeader);
	pBIH->biWidth     = mx.getSize().width;
	pBIH->biHeight    = imHeight;
	pBIH->biPlanes    = 1;
	pBIH->biBitCount  = 8;            
	pBIH->biSizeImage =  line*imHeight;    
	pBIH->biClrUsed   = 256;

	if(jet) {

		//black
		pPalette[0].rgbBlue	 = 0;
		pPalette[0].rgbGreen = 0;
		pPalette[0].rgbRed	 = 0;

		for( i = 1; i<32; i++ ) {
			pPalette[i].rgbBlue	 = 128 + 4*i;
			pPalette[i].rgbGreen = 0;
			pPalette[i].rgbRed	 = 0;
		}
		int index = 0;
		for( i = 32; i<96; i++ ) {
			pPalette[i].rgbBlue	 = 255;
			pPalette[i].rgbGreen = 4*index++;
			pPalette[i].rgbRed   = 0;
		}
		index = 0;
		for( i = 96; i<160; i++ ) {
			pPalette[i].rgbBlue	 = 255 - 4*index;
			pPalette[i].rgbGreen = 255;
			pPalette[i].rgbRed   = 4*index++;
		}
		index = 0;
		for( i = 160; i<224; i++ ) {
			pPalette[i].rgbBlue	 = 0;
			pPalette[i].rgbGreen = 255 - 4*index++;
			pPalette[i].rgbRed   = 255;
		}

		index = 0;
		for( i = 224; i<255; i++ ) {
			pPalette[i].rgbBlue	 = 0;
			pPalette[i].rgbGreen = 0;
			pPalette[i].rgbRed   = 255 - 4*index++;
		}

		// white
		pPalette[255].rgbBlue  = 255;
		pPalette[255].rgbGreen = 255;
		pPalette[255].rgbRed   = 255;
	}
	else {
		for(i = 0; i<256; i++)
		{
			pPalette[i].rgbRed	 = i;
			pPalette[i].rgbGreen = i;
			pPalette[i].rgbBlue	 = i;
		}
	}

	float k = 256.0f / range.span();

	unsigned long imrow = mx.getSize().height - 1 + ((writeRange) ? 32 : 0);

	for(i = 0; i < mx.getSize().height; i++) {
		for(j = 0; j < mx.getSize().width; j++)
		{
			float pix = k*(mx(i,j) - range._min);
			if(pix > 254.0f) pix = 254.0f;
			if(pix < 1)      pix = 1.0f;
			pData[imrow*line + j] = (unsigned char)pix;
		}
		imrow--;
	}

	if(writeRange)
	{
		writeNumber(pData, line, (float)range._min, 4, 4);
		writeNumber(pData, line, (float)range._max, 4, 20);
	}

	std::fstream file;
	file.open(imFileName, std::ios_base::out|std::ios_base::binary);
	file.write((const char*)buf, headers + line*imHeight);
	file.close();
	delete [] buf;
}
