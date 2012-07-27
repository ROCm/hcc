//
// Copyright (C) Microsoft Corporation
// All rights reserved.
//
// Code in Details namespace is for internal usage within the library code
//

#ifndef _PLATFORM_AGILE_H_
#define _PLATFORM_AGILE_H_

#ifdef _MSC_VER
#pragma once
#endif  // _MSC_VER

#include <wrl\client.h>

#if !defined(__cplusplus_winrt)
#error agile.h can only be used with /ZW
#endif

namespace Platform
{
	namespace Details
	{
		__declspec(dllimport) IUnknown* __stdcall GetObjectContext();
		__declspec(dllimport) HRESULT __stdcall GetProxyImpl(IUnknown*, REFIID, IUnknown*, IUnknown**);
		__declspec(dllimport) HRESULT __stdcall ReleaseInContextImpl(IUnknown*, IUnknown*);
		
		template <typename T>
		inline HRESULT GetProxy(T ^ObjectIn, IUnknown *ContextCallBack, T ^*Proxy)
		{
			return GetProxyImpl(*reinterpret_cast<IUnknown**>(&const_cast<T^>(ObjectIn)), __uuidof(T^), ContextCallBack, reinterpret_cast<IUnknown**>(Proxy));
		}

		template <typename T>
		inline HRESULT ReleaseInContext(T *ObjectIn, IUnknown *ContextCallBack)
		{
			return ReleaseInContextImpl(ObjectIn, ContextCallBack);
		}

		template <typename T>
		class AgileHelper
		{
			__abi_IUnknown* _p;
			bool _release;
		public:
			AgileHelper(__abi_IUnknown* p, bool release = true) : _p(p), _release(release)
			{
			}
			AgileHelper(AgileHelper&& other) : _p(other._p), _release(other._release)
			{
				_other._p = nullptr;
				_other._release = true;
			}
			AgileHelper operator=(AgileHelper&& other)
			{
				_p = other._p;
				_release = other._release;
				_other._p = nullptr;
				_other._release = true;
				return *this;
			}

			~AgileHelper()
			{
				if (_release && _p)
				{
					_p->__abi_Release();
				}
			}

			__declspec(no_refcount) __declspec(no_release_return)
				T^ operator->()
			{
				return reinterpret_cast<T^>(_p);
			}

			__declspec(no_refcount) __declspec(no_release_return)
				operator T^()
			{
				return reinterpret_cast<T^>(_p);
			}
		private:
			AgileHelper(const AgileHelper&);
			AgileHelper operator=(const AgileHelper&);
		};
	} // namespace Details

#pragma warning(push)
#pragma warning(disable: 4451) // Usage of ref class inside this context can lead to invalid marshaling of object across contexts

	template <typename T>
	class Agile
	{
		T^ _object;
		::Microsoft::WRL::ComPtr<IUnknown> _contextCallback;    
		ULONG_PTR _contextToken;

		void CaptureContext()
		{
			_contextCallback = Details::GetObjectContext();
			__abi_ThrowIfFailed(CoGetContextToken(&_contextToken));
		}

		void SetObject(T^ object)
		{
			// Capture context before setting the pointer
			// If context capture fails then nothing to cleanup
			Release();
			CaptureContext();
			_object = object;
		}

	public:
		Agile() throw() : _object(nullptr), _contextToken(0)
		{
		}

		Agile(T^ object) throw() : _object(nullptr), _contextToken(0)
		{
			// Assumes that the source object is from the current context
			// Cannot assert
			SetObject(object);
		}

		Agile(const Agile<T>& object) throw() : _object(nullptr), _contextToken(0)
		{
			// Get returns pointer valid for current context
			SetObject(object.Get());
		}

		Agile(Agile<T>&& object) throw() : _object(nullptr), _contextToken(0)
		{
			// Assumes that the source object is from the current context
			_contextToken = object._contextToken;
			_contextCallback.Attach(object._contextCallback.Detach());
			*(IUnknown**)(&_object) = *(IUnknown**)(&object._object);
			*(IUnknown**)(&object._object) = nullptr;
		}

		~Agile() throw()
		{
			Release();
		}

		Details::AgileHelper<T> Get() const
		{
			T^ localObject;
			::Microsoft::WRL::ComPtr<IUnknown> currentContext;
			currentContext.Attach(Details::GetObjectContext());
			if (currentContext.Get() == _contextCallback.Get())
			{
				return Details::AgileHelper<T>(reinterpret_cast<__abi_IUnknown*>(_object), false);
			}
			__abi_ThrowIfFailed(Details::GetProxy(_object, _contextCallback.Get(), &localObject));
			return Details::AgileHelper<T>(reinterpret_cast<__abi_IUnknown*>(__detach_as_voidptr(reinterpret_cast<void**>(&localObject))));
		}

		T^* GetAddressOf() throw()
		{
			Release();
			CaptureContext();
			return &_object;
		}

		T^* GetAddressOfForInOut() throw()
		{
			CaptureContext();
			return &_object;
		}

		Details::AgileHelper<T> operator->() const throw()
		{
			return Get();
		}

		Agile<T> operator=(T^ object) throw()
		{
			if(_object != object)
			{
				SetObject(object);
			}
			return _object;
		}

		Agile<T> operator=(const Agile<T>& object) throw()
		{
			if(_object != object)
			{
				// Get returns pointer valid for current context
				SetObject(object.Get());
			}
			return _object;
		}

		Agile<T> operator=(Agile<T>&& object) throw()
		{
			// Assumes that the source object is from the current context
			_contextToken = object._contextToken;
			_contextCallback.Attach(object._contextCallback.Detach());
			*(IUnknown**)(&_object) = *(IUnknown**)(&object._object);
			*(IUnknown**)(&object._object) = nullptr;
		}

		T^ operator=(IUnknown* lp) throw()
		{
			Release();
			CaptureContext();

			// bump ref count
			::Microsoft::WRL::ComPtr<IUnknown> spObject(lp);

			// put it into Platform Object
			Platform::Object object;
			*(IUnknown**)(&object) = spObject.Detach();

			// QI for the right type
			_object = object;
			return _object;
		}

		// Release the interface and set to NULL
		void Release() throw()
		{
			if (_object)
			{
				// Cast to IInspectable (no QI)
				IUnknown* pObject = *(IUnknown**)(&_object);
				// Set ^ to null without release
				*(IUnknown**)(&_object) = nullptr;

				::Microsoft::WRL::ComPtr<IUnknown> currentContext;
				currentContext.Attach(Details::GetObjectContext());
				if (currentContext.Get() == _contextCallback.Get())
				{
					pObject->Release();
				}
				else
				{
					Details::ReleaseInContext(pObject, _contextCallback.Get());
				}
				_contextToken = 0;
			}
		}
	};

#pragma warning(pop)

} // namespace Platform

#endif // _PLATFORM_AGILE_H_
