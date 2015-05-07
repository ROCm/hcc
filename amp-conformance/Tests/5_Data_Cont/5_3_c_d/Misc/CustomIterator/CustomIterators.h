// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

#include "amptest.h"

namespace CustomIterator
{
	template<typename T> class CustomContainer;
	template<typename T> class InputIterator;
	template<typename T> class ForwardIterator;

	template<typename T>
	class CustomContainer
	{
		friend class InputIterator<T>;
		friend class ForwardIterator<T>;

	public:
		typedef  InputIterator<T> input_iterator;
		typedef  ForwardIterator<T> forward_iterator;

		std::vector<T> container;

		CustomContainer<T>() { container = std::vector<T>(); }
		CustomContainer<T>(int size) {	container = std::vector<T>(size); }

		input_iterator read_begin()
		{
			auto var = container.begin();
			return InputIterator<T>(var);
		}
		input_iterator read_end()
		{
			auto var = container.end();
			return InputIterator<T>(var);
		}

		forward_iterator write_begin()
		{
			auto var = container.begin();
			return ForwardIterator<T>(var);
		}
		
		forward_iterator write_end()
		{
			auto var = container.end();
			return ForwardIterator<T>(container.end());
		}

	};

	template<typename T>
	class InputIterator : public std::iterator<std::input_iterator_tag, T>
	{
		friend class CustomContainer<T>;

	private:
		typename std::vector<T>::iterator actual_iterator;	
		InputIterator(typename std::vector<T>::iterator& actual_iterator) { this->actual_iterator = actual_iterator; }

	public:
		const T& operator*() { return *actual_iterator; }

		const InputIterator<T>& operator++()
		{
			++(this->actual_iterator);
			return *this;
		}

		const InputIterator<T> operator++(int)
		{
			InputIterator<T> temp(this->actual_iterator);
			operator++();
			return temp;
		}

		bool operator!=(const InputIterator<T>& other) const { return (this->actual_iterator != other.actual_iterator); }
		bool operator==(const InputIterator<T>& other) const { return (this->actual_iterator == other.actual_iterator); }
	};

	template<typename T>
	class ForwardIterator : public std::iterator<std::forward_iterator_tag, T>
	{
		friend class CustomContainer<T>;

	private:
		typename std::vector<T>::iterator actual_iterator;

		ForwardIterator(typename std::vector<T>::iterator& actual_iterator)
		{
			this->actual_iterator = actual_iterator;
		}

	public:
		T& operator*() { return *actual_iterator; }
		
		const ForwardIterator<T>& operator++()
		{
			++(this->actual_iterator);
			return *this;
		}

		const ForwardIterator<T> operator++(int)
		{
			ForwardIterator temp(*this);
			operator++();
			return temp;
		}

		bool operator!=(const ForwardIterator<T>& other) const { return (this->actual_iterator != other.actual_iterator); }
		bool operator==(const ForwardIterator<T>& other) const { return (this->actual_iterator == other.actual_iterator); }
	};
}