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
#include <assert.h>
#include <math.h>

namespace utils {

	template <typename T>
	inline bool isFloat()
	{
		T t = (T)0.5f;
		return (0.5f == (float)t);
	}; // usage for T: isFloat<T>()

	template <typename T>
	struct Range
	{
		const T _max;
		const T _min;
		inline Range(T min, T max): _min(min), _max(max) { /*assert( _min < _max );*/ }; 
		inline float span() const throw() { return (isFloat<T>())?(_max - _min):((float)_max - (float)_min + 1); };
	};

	template<typename T>
	struct SizeT
	{
		T height;
		T width;

		SizeT<T>(): width((T)0), height((T)0) { };
		SizeT<T>( T Height, T Width ) :
			height(Height), width(Width) {}; 
		inline bool operator==( const SizeT<T>& s )  { return (height == s.height) && (width == s.width); };
		inline bool operator!=( const SizeT<T>& s )  { return (height != s.height) || (width != s.width); };
		inline T all() const { return height*width; };
	};
	typedef SizeT<unsigned long> Size;

	struct Point {
		Point(int xx=0, int yy=0) restrict(cpu,amp) : x(xx), y(yy)  {};
		int x;
		int y;
	};

	template<typename T>
	struct Rectangle
	{
		T top;
		T left;
		T bottom;
		T right;

		Rectangle<T>() :
			top((T)1), left((T)1), bottom((T)0), right((T)0) {};

		Rectangle<T>( T Top, T Left, T Bottom, T Right )  restrict(amp,cpu) :
			top(Top), left(Left), bottom(Bottom), right(Right) {};

		inline T height() const {
			return isFloat<T>() ? bottom - top : bottom - top + 1;
		};

		inline T width() const {
			return isFloat<T>() ? right - left : right - left + 1;
		};

		inline T area() const {
			return  isFloat<T>() ? (bottom-top)*(right-left) : (bottom-top+1)*(right-left+1);
		};
	};
	typedef Rectangle<int> Rect;


#define DEFAULT_ALIGNMENT 32
	typedef unsigned int uint;

	template < typename T >
	class Matrix
	{
	public: 

		Matrix( Size size=Size(0,0), T* data=NULL);
		~Matrix( );

		inline T& operator()  (uint row, uint col) const throw() {
			assert( row < _size.height && col < _size.width );
			return _data[_lineLength*row + col]; 
		};

		inline const T* operator[] ( uint row ) const throw() {
			assert( row < _size.height );
			return &(_data[_lineLength*row]);
		};
		inline T* operator[] ( uint row ) throw() {
			assert( row < _size.height );
			return &(_data[_lineLength*row]);
		};

		inline Size getSize() const throw() { return _size; };
		inline size_t sizeBytes() const { return _size.all() * sizeof(T); };
		inline int pitch() const { return _size.width; };
		T *data() const { return _data; };
		T *wdata() { return _data; };


	private:
		static size_t Matrix<T>::allocSize( Size size );
		int calculateLineLength( int cnt, int size, int alignment=0 ) throw();

	private:
		Size	_size;
		int		    _lineLength;
		T*		    _data;
	};


	// Implementation
	template<typename T>
	Matrix<T>::Matrix( Size size, T* data)
		: _size( size ), _lineLength(0), _data(data)
	{
		if (size.all() != 0) {
			_lineLength = calculateLineLength( size.width, sizeof(T) );
			size_t data   = sizeof(T)*( size.height * calculateLineLength(size.width, sizeof(T)) );
			if (_data == NULL) {
				_data = reinterpret_cast<T*> (malloc(data));
			}
		} 
	}

	template<typename T>
	Matrix<T>::~Matrix()
	{
		if (_data != 0) {
			free(_data);
			_data = 0;
		}
	}

	template<typename T>
	int Matrix<T>::calculateLineLength( int cnt, int size, int alignment ) throw()
	{
		if(!alignment)
			alignment = DEFAULT_ALIGNMENT;

		int len = cnt*size;
		int mod = len % alignment;
		if( !mod )
			return cnt;
		return (len + alignment - mod)/size;

	}
};

#define M_PI       3.14159265358979323846
class H3 {
public:
	H3(); 
	void zeros();

	inline double& operator[](int index) throw() {
		assert(index < 9);
		return _H[index]; 
	};
	inline const double& operator[](int index) const throw() {
		assert(index < 9);
		return _H[index]; 
	};
	inline double& operator()(int row, int col) throw() { 
		assert(row < 3 && col < 3);
		return _H[ row*3 + col]; 
	};
	inline const double& operator()(int row, int col) const throw() { 
		assert(row < 3 && col < 3);
		return _H[ row*3 + col]; 
	};

	inline void H3::eye()
	{
		_H[0] = 1.0; _H[1] = 0.0; _H[2] = 0.0;
		_H[3] = 0.0; _H[4] = 1.0; _H[5] = 0.0;
		_H[6] = 0.0; _H[7] = 0.0; _H[8] = 1.0;
	}

	inline void H3::rot( double angle )
	{
		eye();
		angle *= M_PI/180.0;
		_H[0] = _H[4] = cos(angle);
		_H[1] = sin(angle);
		_H[3] = -_H[1];
	}
private:
	double _H[9];
};

inline H3::H3()
{
	_H[0] = 1.0; _H[1] = 0.0; _H[2] = 0.0;
	_H[3] = 0.0; _H[4] = 1.0; _H[5] = 0.0;
	_H[6] = 0.0; _H[7] = 0.0; _H[8] = 1.0;
}

inline void H3::zeros()
{
	_H[0] = 0.0; _H[1] = 0.0; _H[2] = 0.0;
	_H[3] = 0.0; _H[4] = 0.0; _H[5] = 0.0;
	_H[6] = 0.0; _H[7] = 0.0; _H[8] = 0.0;
}
