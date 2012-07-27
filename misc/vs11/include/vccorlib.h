//
// Copyright (C) Microsoft Corporation
// All rights reserved.
//
// This header is included by the compiler using /FI when /ZW is specified
// Do not include any headers in this file
#ifndef _VCCORLIB_H_
#define _VCCORLIB_H_

#ifdef _MSC_VER
#pragma once
#endif  // _MSC_VER

#if !defined(__cplusplus_winrt)
#error vccorlib.h can only be used with /ZW
#endif

#if defined(VCWINRT_DLL)
#include <stdio.h>
#include <windows.h>
#include <inspectable.h>
#include <WinString.h>
#endif

// All WinRT types should have a packing (the default C++ packing).
#ifdef  _WIN64
#pragma pack(push, 16)
#else
#pragma pack(push, 8)
#endif

// <InternalComment>
// READ THIS BEFORE MAKING CHANGES TO THIS FILE:
//    This is a force-include file which is used by all /ZW compilations, and akin to a typesrc file.
//    /ZW should be usable to build any existing body of C++ code (including Windows, SQL, Office etc.)
//    As such, the following rules should be observed:
//        * Do not include any header files that have any behavior that can be changed by the user (e.g. #ifdef)
//        * Do not declare a method or typename that can conflict with an existing method or type that comes from a header
//          if the header may modify that type based on user #defines.
//    General rules:
//        * Keep PDB sizes small. Don't overuse templates, and keep identifiers short.
// </InternalComment>

// <InternalComment>
// Postconditions: ParsingInitTypes is set
// </InternalComment>

#if defined(__VCCORLIB_H_ENABLE_ALL_WARNINGS)
#pragma warning(push)
#endif

// Following warnings disabled globally
// To enable these warnings define __VCCORLIB_H_ENABLE_ALL_WARNINGS
#pragma warning(disable: 4400) // const/volatile qualifiers on ^ are not supported
#pragma warning(disable: 4514) // unreferenced inline function has been removed
#pragma warning(disable: 4710) // function not inlined
#pragma warning(disable: 4711) // selected for automatic inline expansion

// Following warnings disabled for this file
#pragma warning( push )
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4483) // Allows us to use __identifier
#pragma warning(disable: 4820) // bytes padding added after data member

#pragma initialize_winrt_types_start

struct HSTRING__;
typedef HSTRING__* __abi_HSTRING;
typedef HSTRING__* HSTRING;

__declspec(noreturn) void __stdcall __abi_WinRTraiseException(long);

inline void __abi_ThrowIfFailed(long __hrArg)
{
	if (__hrArg < 0)
	{
		__abi_WinRTraiseException(__hrArg);
	}
}

#if !defined(VCWINRT_DLL)
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_FailFast();
#else
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_FailFast();
#endif

namespace __winRT
{
	long __stdcall __windowsCreateString(const __wchar_t*, int, HSTRING*);
	long __stdcall __getActivationFactoryByPCWSTR(void*, ::Platform::Guid&, void**);
	long __stdcall __getIids(int, unsigned long*, const __s_GUID*, ::Platform::Guid**);
}

namespace Windows
{
	namespace Foundation
	{
	}
}

struct __abi_WinClassInstrumentation
{
	__abi_WinClassInstrumentation* callback;

	int numcalls_QueryInterface;
	int numcalls_AddRef;
	int numcalls_Release;
	int numcalls_GetIids;
	int numcalls_GetRuntimeClassName;
	int numcalls_GetTrustLevel;
	int numcalls_Other;
	int destructed;
	int refcount;

	__abi_WinClassInstrumentation()
	{
		callback = nullptr;
		numcalls_QueryInterface = 0;
		numcalls_AddRef = 0;
		numcalls_Release = 0;
		numcalls_GetIids = 0;
		numcalls_GetRuntimeClassName = 0;
		numcalls_GetTrustLevel = 0;
		numcalls_Other = 0;
		destructed = 0;
		refcount = 0;
	}

	void __abi_SetInstrumentationData(__abi_WinClassInstrumentation* __callbackArg)
	{
		callback = __callbackArg;
		__abi_CopyToAttached();
	}

	void __abi_CopyToAttached()
	{
		if (callback)
		{
			callback->numcalls_QueryInterface     = numcalls_QueryInterface;
			callback->numcalls_AddRef             = numcalls_AddRef;
			callback->numcalls_Release            = numcalls_Release;
			callback->numcalls_GetIids            = numcalls_GetIids;
			callback->numcalls_GetRuntimeClassName= numcalls_GetRuntimeClassName;
			callback->numcalls_GetTrustLevel      = numcalls_GetTrustLevel;
			callback->numcalls_Other              = numcalls_Other;
			callback->destructed                  = destructed;
			callback->refcount                    = refcount;
		}
	}
};

//
//// Don't want to define the real IUnknown from unknown.h here. That would means if the user has
//// any broken code that uses it, compile errors will take the form of e.g.:
////     predefined C++ WinRT types (compiler internal)(41) : see declaration of 'IUnknown::QueryInterface'
//// This is not helpful. If they use IUnknown, we still need to point them to the actual unknown.h so
//// that they can see the original definition.
////
//// For WinRT, we'll instead have a parallel COM interface hierarchy for basic interfaces starting with _.
//// The type mismatch is not an issue. COM passes types through GUID / void* combos - the original type
//// doesn't come into play unless the user static_casts an implementation type to one of these, but
//// the WinRT implementation types are hidden.
__interface __declspec(uuid("00000000-0000-0000-C000-000000000046")) __abi_IUnknown
{
public:
	virtual long __stdcall __abi_QueryInterface(::Platform::Guid&, void**) = 0;
	virtual unsigned long __stdcall __abi_AddRef() = 0;
	virtual unsigned long __stdcall __abi_Release() = 0;
};

enum __abi_TrustLevel
{
	__abi_BaseTrust        = 0,
	__abi_PartialTrust    = (__abi_BaseTrust    + 1) ,
	__abi_FullTrust      = (__abi_PartialTrust + 1)
};

__interface __declspec(uuid("3C5C94E8-83BB-4622-B76A-B505AE96E0DF")) __abi_Module
{
public:
	virtual unsigned long __stdcall __abi_IncrementObjectCount() = 0;
	virtual unsigned long __stdcall __abi_DecrementObjectCount() = 0;
};

extern __abi_Module* __abi_module;

extern "C" long __cdecl _InterlockedIncrement(long volatile *);
extern "C" long __cdecl _InterlockedDecrement(long volatile *);
extern "C" long __cdecl _InterlockedCompareExchange(long volatile *, long, long);

#pragma intrinsic(_InterlockedIncrement)
#pragma intrinsic(_InterlockedDecrement)
#pragma intrinsic(_InterlockedCompareExchange)

// A class that represents a volatile refcount, that gets initialized to 0.
class __abi_MultiThreadedRefCount
{
	long __refcount;
public:
	__abi_MultiThreadedRefCount() : __refcount(1)
	{
		__abi_module->__abi_IncrementObjectCount();
	}

	inline unsigned long Increment() volatile
	{
		return static_cast<unsigned long>(_InterlockedIncrement(&__refcount));
	}

	inline unsigned long Decrement() volatile
	{
		unsigned long __refCountLoc = static_cast<unsigned long>(_InterlockedDecrement(&__refcount));
		if (__refCountLoc == 0)
		{
			// When destructing objects at the end of the program, we might be freeing
			// objects across dlls, and the dll this object is in might have already freed its module object.
			if (__abi_module != nullptr) {
				__abi_module->__abi_DecrementObjectCount();
			}
		}
		return __refCountLoc;
	}

	inline unsigned long Get() volatile
	{
		return static_cast<unsigned long>(__refcount);
	}
};

__interface __declspec(uuid("AF86E2E0-B12D-4c6a-9C5A-D7AA65101E90")) __abi_IInspectable : public __abi_IUnknown
{
	virtual long __stdcall __abi_GetIids(unsigned long*, ::Platform::Guid**) = 0;
	virtual long __stdcall __abi_GetRuntimeClassName(HSTRING*) = 0;
	virtual long __stdcall __abi_GetTrustLevel(__abi_TrustLevel*) = 0;
};

__interface __declspec(uuid("00000001-0000-0000-C000-000000000046")) __abi_IClassFactory : public __abi_IUnknown
{
	virtual long __stdcall __abi_CreateInstance(__abi_IUnknown*, ::Platform::Guid&, void**) = 0;
	virtual long __stdcall __abi_LockServer(int) = 0;
};

__interface __declspec(uuid("00000035-0000-0000-C000-000000000046")) __abi_IActivationFactory : public __abi_IInspectable
{
	virtual long __stdcall __abi_ActivateInstance(__abi_IInspectable**) = 0;
};

#if !defined(VCWINRT_DLL)
typedef struct __Platform_Details_HSTRING_HEADER
{
	int             __flags;             // Bit flags which used for storing extra information
	unsigned int    __length;            // length of string's unicode code point
	unsigned int    __padding;           // padding for future use
	unsigned int    __morepadding;       // padding for future use
	__wchar_t*      __stringRef;         // An address pointer which points to a string buffer.
} __Platform_Details_HSTRING_HEADER;
#else
typedef HSTRING_HEADER __Platform_Details_HSTRING_HEADER;
#endif

namespace Platform { namespace Details {
	struct EventLock
	{
		void* __targetsLock;
		void* __addRemoveLock;
	};

	template<typename T>
	EventLock* GetStaticEventLock()
	{
		static EventLock __eventLock = { nullptr, nullptr };
		return &__eventLock;
	}
}} // namespace Platform::Details

// <InternalComment>
// Initialize a set of PCH global roots from some of the types defined above this point.
// Preconditions: The following types must be defined before this point:
//                     __abi_IUnknown
//                     __abi_IInspectable
//                     __abi_IClassFactory
//                     __abi_IActivationFactory
//                     HSTRING
//                     __abi_TrustLevel
//                     ::Platform::Guid
//                     __abi_MultiThreadedRefCount
// Postconditions: * The following PCH global roots are initialized
//                     pWinRTIUnknown
//                     pWinRTIInspectable
//                     pWinRTIClassFactory
//                     pWinRTIActivationFactory
//                     pWinRTHSTRING
//                     pWinRTTrustLevel
//                     pWindowsFoundationGuid
//                     pWinRTMultiThreadedRefCount
//                  * Windows.Foundation.winmd is loaded
//                  * From this point on WinRT types can be declared using 'ref class', 'interface class' etc. BUT must have __declspec(no_weak_ref)
//                  * ParsingInitTypes is still set
// </InternalComment>
#pragma initialize_winrt_types_phase1

namespace __abi_details
{
	// String^
	__declspec(non_user_code) __declspec(no_refcount) void __abi_delete_String(::Platform::String^);

#if !defined(VCWINRT_DLL)
	__declspec(dllimport) __declspec(non_user_code) __declspec(no_refcount)
		::Platform::Object ^ __stdcall __abi_cast_String_to_Object(::Platform::String^);

	__declspec(dllimport) __declspec(non_user_code) __declspec(no_refcount)
		::Platform::String ^ __stdcall __abi_cast_Object_to_String(bool, ::Platform::Object^);

	__declspec(dllimport) __declspec(non_user_code) ::Platform::String^ __stdcall __abi_ObjectToString(::Platform::Object^ o, bool useIPrintable);

#else
	__declspec(dllexport) __declspec(non_user_code) __declspec(no_refcount)
		::Platform::Object ^ __stdcall __abi_cast_String_to_Object(::Platform::String^);

	__declspec(dllexport) __declspec(non_user_code) __declspec(no_refcount)
		::Platform::String ^ __stdcall __abi_cast_Object_to_String(bool, ::Platform::Object^);

	__declspec(dllexport) __declspec(non_user_code) ::Platform::String^ __stdcall __abi_ObjectToString(::Platform::Object^ o, bool useIPrintable);
#endif
} // namespace __abi_details

__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	inline void* __abi_winrt_ptr_ctor(const volatile ::Platform::Object^ const __objArg)
{
	__abi_IUnknown* __pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Object^>(__objArg));
	if (__pUnknown) {
		__pUnknown->__abi_AddRef();
	}
	return __pUnknown;
}

__declspec(non_user_code) __declspec(no_refcount)
	inline void __abi_winrt_ptr_dtor(const volatile ::Platform::Object^ const __objArg)
{
	__abi_IUnknown* __pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Object^>(__objArg));
	if (__pUnknown) {
		__pUnknown->__abi_Release();
	}
}

__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	inline void* __abi_winrt_ptr_assign(void** __ppTargetArg, const volatile ::Platform::Object^ __objArg)
{
	__abi_IUnknown* __pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Object^>(__objArg));
	__abi_IUnknown** __ppTargetUnknown = reinterpret_cast<__abi_IUnknown**>(__ppTargetArg);
	if (__pUnknown != *__ppTargetUnknown)
	{
		if (__pUnknown) {
			__pUnknown->__abi_AddRef();
		}
		if (*__ppTargetUnknown) {
			(*__ppTargetUnknown)->__abi_Release();
		}
		*__ppTargetUnknown = __pUnknown;
	}
	return __pUnknown;
}

// Used for handle which is inside '__declspec(no_refcount)' function but still needs Release.
struct __abi_dtor_helper {
private:
	__abi_IUnknown *__pUnknown;
public:
	__declspec(non_user_code) __abi_dtor_helper(const volatile ::Platform::Object^ __objArg) {
		__pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Object^>(__objArg));
	}
	__declspec(non_user_code) ~__abi_dtor_helper() {
		if (__pUnknown) {
			__pUnknown->__abi_Release();
		}
	}
};

// The exceptions are split out explicitly in order to make them obvious from callstacks.
#if !defined(VCWINRT_DLL)
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseNotImplementedException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseInvalidCastException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseNullReferenceException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOperationCanceledException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseFailureException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseAccessDeniedException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOutOfMemoryException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseInvalidArgumentException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOutOfBoundsException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseChangedStateException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseClassNotRegisteredException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseWrongThreadException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseDisconnectedException();
__declspec(dllimport) __declspec(noreturn) void __stdcall __abi_WinRTraiseCOMException(long);
#else
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseNotImplementedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseInvalidCastException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseNullReferenceException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOperationCanceledException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseFailureException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseAccessDeniedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOutOfMemoryException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseInvalidArgumentException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOutOfBoundsException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseChangedStateException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseClassNotRegisteredException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseWrongThreadException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseDisconnectedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseCOMException(long);
#endif

__declspec(non_user_code) __declspec(no_refcount)
	inline ::Platform::Object^ __abi_winrt_cast_to(bool __isDynamicCastArg, ::Platform::Object^ __objArg, const _GUID& __guidArg)
{
	void* __pTo = nullptr;
	__abi_IUnknown* __pUnknown = reinterpret_cast<__abi_IUnknown*>(__objArg);

	if (__pUnknown) {
		// Cast to ::Platform::Guid instead of using conversion in order to avoid copy to temporary
		long __hr = __pUnknown->__abi_QueryInterface(reinterpret_cast< ::Platform::Guid&>(const_cast<_GUID&>(__guidArg)), &__pTo);

		if (__isDynamicCastArg && __hr != 0)
			return nullptr;
		// It will throw InvalidCastException on failure
		__abi_ThrowIfFailed(__hr);
	}

	return reinterpret_cast< ::Platform::Object^>(__pTo);
}

__declspec(non_user_code) __declspec(no_refcount)
	inline ::Platform::String^ __abi_winrt_cast_to_string(bool __isDynamicCast, ::Platform::Object^ __objArg)
{
	return __abi_details::__abi_cast_Object_to_String(__isDynamicCast, __objArg);
}

__declspec(non_user_code) __declspec(no_refcount)
	inline ::Platform::Object^ __abi_winrt_cast_from_string_to_object(bool /*__isDynamicCastArg*/, ::Platform::String^ __objArg)
{
	return __abi_details::__abi_cast_String_to_Object(__objArg);
}

__declspec(non_user_code) __declspec(no_refcount)
	inline ::Platform::Object^ __abi_winrt_cast_from_string_to_other(bool /*__isDynamicCastArg*/, ::Platform::String^ /*__objArg*/)
{
	__abi_WinRTraiseInvalidCastException();
}

inline void* __detach_as_voidptr(void** __ppObjArg)
{
	void* __pObj = *__ppObjArg;
	*__ppObjArg = nullptr;
	return __pObj;
}

__declspec(non_user_code) __declspec(no_refcount)
	inline void __abi_winrt_ptrto_string_dtor(const volatile ::Platform::String^ const __objArg)
{
	__abi_details::__abi_delete_String(const_cast< ::Platform::String^>(__objArg));
}

// Function decleration for types we use from Windows and CRT
// This prevents pulling in the headers
#if !defined(VCWINRT_DLL)
extern "C" long __stdcall __Platform_CoCreateFreeThreadedMarshaler(::Platform::Object^, ::Platform::Object^*);
#endif

namespace Platform {
	template <typename __TArg, unsigned int __dimension = 1>
	ref class WriteOnlyArray;

	template <typename __TArg, unsigned int __dimension = 1>
	ref class Array;
}

template <typename __TArg, unsigned int __dimension>
__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	void* __abi_winrt_ptr_ctor(const volatile ::Platform::Array<__TArg, __dimension>^ const);

template<typename __TArg, unsigned int __dimension>
__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	void* __abi_winrt_ptr_assign(void**, const volatile ::Platform::Array<__TArg, __dimension>^);

__declspec(non_user_code) __declspec(no_refcount)
	inline ::Platform::Object^ __abi_winrt_cast_use_helper(bool __isDynamicArg, void* __fromArg, const _GUID& __guidArg, __abi_IUnknown* __useresultArg)
{
	if (__useresultArg)
	{
		return reinterpret_cast< ::Platform::Object^>(__useresultArg);
	}

	return __abi_winrt_cast_to(__isDynamicArg, reinterpret_cast< ::Platform::Object^>(__fromArg), __guidArg);
}

__declspec(selectany) void * __forceInstantiate1 = &__abi_winrt_cast_use_helper;

__declspec(non_user_code) __declspec(no_refcount)
	inline void __abi_winrt_ptr_dispose_dtor(const volatile ::Platform::Object^ const __objArg)
{
	::Platform::IDisposable ^__dispose = dynamic_cast< ::Platform::IDisposable ^>(const_cast< ::Platform::Object^>(__objArg));
	if (__dispose) {
		__dispose->__identifier("<Dispose>")();
		reinterpret_cast<__abi_IUnknown*>(__dispose)->__abi_Release();
	}

	__abi_winrt_ptr_dtor(__objArg);
}

namespace Platform { namespace Details
{
    [::Windows::Foundation::Metadata::MarshalingBehaviorAttribute(::Windows::Foundation::Metadata::MarshalingType::Standard)]
	ref class __declspec(no_weakreferencesource)
		__declspec(no_empty_identity_interface)
		WeakReferenceSource : public ::Platform::Details::IWeakReference
	{
	private:
		volatile long __strongRefCount;
		__abi_IUnknown* __target; // we shouldn't hold a strong reference to target, so grab an unaddref'd reference here.

	public:
		virtual ::Platform::Object^ Resolve(::Platform::Guid* __riidArg)
		{
			for(;;)
			{
				long __ref = __strongRefCount;
				if (__ref == 0)
				{
					return nullptr;
				}

				// InterlockedCompareExchange calls _InterlockedCompareExchange intrinsic thus we call directly _InterlockedCompareExchange to save the call
				if (::_InterlockedCompareExchange(&__strongRefCount, __ref + 1, __ref) == __ref)
				{
					break;
				}
			}

			::Platform::Object^ __obj;
			long __hr = __target->__abi_QueryInterface(*__riidArg, reinterpret_cast<void**>(&__obj));
			__abi_ThrowIfFailed(__hr);

			// Undo the "addref" that happened above during the _InterlockedCompareExchange call
			__target->__abi_Release();

			return __obj;
		}

	internal:
		WeakReferenceSource(Object^ __targetArg) : __target( reinterpret_cast<__abi_IUnknown*>(__targetArg) ), __strongRefCount(1)
		{
		}

		unsigned long __stdcall IncrementStrongReference()
		{
			// InterlockedIncrement calls _InterlockedIncrement intrinsic thus we call directly _InterlockedIncrement to save the call
			return static_cast<unsigned long>(_InterlockedIncrement(&__strongRefCount));
		}

		unsigned long __stdcall DecrementStrongReference()
		{
			// InterlockedDecrement calls _InterlockedDecrement intrinsic thus we call directly _InterlockedDecrement to save the call
			unsigned long __ref = static_cast<unsigned long>(_InterlockedDecrement(&__strongRefCount));
			if (__ref == 0)
			{
				// The IWeakReferenceSource interface is being destroyed thus the reference is no longer valid.
				// This shouldn'->Release() the object.
				__target = nullptr;
			}
			return __ref;
		}

		inline long __stdcall GetRefcount()
		{
			return __strongRefCount;
		}
	};
}} // ::Platform::Details

// A class that represents a volatile refcount, that gets initialized to 0.
class __abi_FTMWeakRefData
{
	__abi_IUnknown* __weakRefSource;
	__abi_IUnknown* __pUnkMarshal;

public:
	__abi_FTMWeakRefData(::Platform::Object^ __targetArg)
	{
#if !defined(VCWINRT_DLL)
		long __hr = ::__Platform_CoCreateFreeThreadedMarshaler(__targetArg,  reinterpret_cast< ::Platform::Object^*>(&__pUnkMarshal));
#else
		long __hr = ::CoCreateFreeThreadedMarshaler(reinterpret_cast<IUnknown*>(__targetArg),  reinterpret_cast<IUnknown**>(&__pUnkMarshal));
#endif
		__abi_ThrowIfFailed(__hr);

		auto __weakRef = ref new ::Platform::Details::WeakReferenceSource(__targetArg);
		__weakRefSource = reinterpret_cast<__abi_IUnknown*>(__weakRef);
		*reinterpret_cast<__abi_IUnknown**>(&__weakRef) = nullptr;

		__abi_module->__abi_IncrementObjectCount();
	}

	__abi_FTMWeakRefData(::Platform::Object^ __targetArg, ::Platform::CallbackContext __contextArg)
	{
		if (__contextArg == ::Platform::CallbackContext::Any)
		{
#if !defined(VCWINRT_DLL)
			long __hr = ::__Platform_CoCreateFreeThreadedMarshaler(__targetArg,  reinterpret_cast< ::Platform::Object^*>(&__pUnkMarshal));
#else
			long __hr = ::CoCreateFreeThreadedMarshaler(reinterpret_cast<IUnknown*>(__targetArg),  reinterpret_cast<IUnknown**>(&__pUnkMarshal));
#endif
			__abi_ThrowIfFailed(__hr);
		}

		auto __weakRef = ref new ::Platform::Details::WeakReferenceSource(__targetArg);
		__weakRefSource = reinterpret_cast<__abi_IUnknown*>(__weakRef);
		*reinterpret_cast<__abi_IUnknown**>(&__weakRef) = nullptr;

		__abi_module->__abi_IncrementObjectCount();
	}

	inline unsigned long __stdcall Increment() volatile
	{
		if (!__weakRefSource)
		{
			return static_cast<unsigned long>(-1); // Called during destruction
		}

		return reinterpret_cast< ::Platform::Details::WeakReferenceSource^>(__weakRefSource)->IncrementStrongReference();
	}

	inline long __stdcall Decrement() volatile
	{
		if (!__weakRefSource)
		{
			return static_cast<unsigned long>(-1); // Called during destruction
		}

		unsigned long __refCount = reinterpret_cast< ::Platform::Details::WeakReferenceSource^>(__weakRefSource)->DecrementStrongReference();
		if (__refCount == 0)
		{
			__weakRefSource->__abi_Release();
			__weakRefSource = nullptr;

			if (__pUnkMarshal)
			{
				__pUnkMarshal->__abi_Release();
				__pUnkMarshal = nullptr;
			}

			// When destructing objects at the end of the program, we might be freeing
			// objects across dlls, and the dll this object is in might have already freed its module object.
			if (__abi_module != nullptr) {
				__abi_module->__abi_DecrementObjectCount();
			}
		}

		return __refCount;
	}

	inline __abi_IUnknown* GetFreeThreadedMarshaler()
	{
		return __pUnkMarshal;
	}

	inline ::Platform::Details::IWeakReference^ GetWeakReference()
	{
		return reinterpret_cast< ::Platform::Details::IWeakReference^>(__weakRefSource);
	}

	inline long __stdcall Get() volatile
	{
		return reinterpret_cast< ::Platform::Details::WeakReferenceSource^>(__weakRefSource)->GetRefcount();
	}
};

namespace Platform { namespace Details
{
	struct __abi_CaptureBase
	{
	protected:
		virtual __stdcall ~__abi_CaptureBase()  {}

	public:
		static const size_t __smallCaptureSize = 4 * sizeof(void*);
		void* operator new(size_t __sizeArg, void* __pSmallCaptureArg)
		{
			if (__sizeArg > __smallCaptureSize)
			{
				return reinterpret_cast<__abi_CaptureBase*>( ::Platform::Details::Heap::Allocate( __sizeArg ) );
			}

			return __pSmallCaptureArg;
		}

		void operator delete(void* __ptrArg, void* __pSmallCaptureArg)
		{
			__abi_CaptureBase* __pThis = static_cast<__abi_CaptureBase*>(__ptrArg);
			__pThis->Delete(__pThis, __pSmallCaptureArg);
		}

		inline void* GetVFunction(int __slotArg)
		{
			return (*reinterpret_cast<void***>(this))[__slotArg];
		}

		void Delete(__abi_CaptureBase* __pThisArg, void* __pSmallCaptureArg)
		{
			__pThisArg->~__abi_CaptureBase();
			if (__pThisArg != __pSmallCaptureArg)
			{
				::Platform::Details::Heap::Free(__pThisArg);
			}
		}
	};

	struct __abi_CapturePtr
	{
		char* smallCapture[__abi_CaptureBase::__smallCaptureSize];
		__abi_CaptureBase* ptr;
		__abi_CapturePtr() : ptr( reinterpret_cast<__abi_CaptureBase*>(smallCapture) )  {}
		~__abi_CapturePtr()
		{
			ptr->Delete(ptr, smallCapture);
		}
	};
}} // namespace Platform::Details

// <InternalComment>
// Initialize a set of PCH global roots from some of the types defined above this point.
// Preconditions: See initialize_winrt_types_phase1 preconditions:
//                     __abi_FTMWeakRefData is now defined
// Postconditions:  * From this point on WinRT types can be declared using 'ref class', 'interface class' etc.
//                  * ParsingInitTypes is still set
// </InternalComment>

#pragma initialize_winrt_types_phase2

namespace Platform 
{
	template <typename __TArg, unsigned int __dimension = 1>
	class ArrayReference;

	namespace Details
	{
		template <typename __HighLevelType, unsigned int __dimension>
		::Platform::Array<__HighLevelType, __dimension>^ __abi_array_attach(void* __src, unsigned int __size, bool __isFastPass, bool __needsInit);

		template <typename __HighLevelType, unsigned int __dimension>
		void __abi_array_copy_to_and_release(::Platform::Array<__HighLevelType, __dimension>^ __arr, void** __dest, unsigned int* __size);

		template <typename __LowLevelType, typename __HighLevelType, unsigned int __dimension>
		__LowLevelType* __abi_array_to_raw(const ::Platform::Array<__HighLevelType, __dimension>^);

		template <typename __TArg, bool = __is_enum(__TArg)>
		struct array_helper;
	} // namespace Details

#pragma warning(push)
#pragma warning(disable: 4487)
	// Partial specialization of one-dimensional Array
	template <typename __TArg>
	private ref class WriteOnlyArray<__TArg, 1>
	{
	protected private:
		unsigned int   __size;         // number of elements
		bool           __fastpassflag; // true if "fast pass", else false
		__TArg*        __data;        // actual data buffer, alloc'd if not "fast-pass"

	internal:
		__TArg& set(unsigned int __indexArg, __TArg __valueArg);
		property unsigned int Length {unsigned int get() const; }
		property __TArg* Data { __TArg* get() const; }
		property bool FastPass { bool get() const; }

		__TArg* begin() const;
		__TArg* end() const;

	protected private:
		WriteOnlyArray();
		WriteOnlyArray(unsigned int __sizeArg);
		WriteOnlyArray(__TArg* __dataArg, unsigned int __sizeArg);
		void Clear();

		__TArg& get(unsigned int __indexArg) const;

		static __TArg* AllocateAndZeroInitialize(unsigned int __countArg);
		static __TArg* AllocateAndCopyElements(const __TArg* __srcArg, unsigned int __countArg);
	private:
		virtual ~WriteOnlyArray();
	};

	template <typename __TArg>
	private ref class Array<__TArg,1> sealed :
		public WriteOnlyArray<__TArg, 1>,
			public [::Windows::Foundation::Metadata::Default] [::Platform::Metadata::RuntimeClassName] ::Platform::IBoxArray<__TArg>
		{
		public:
			virtual property Array^ Value { virtual Array^ get(); }

        internal:
            Array(const Array<__TArg, 1>^ __source);
            Array(unsigned int __sizeArg);
			Array(__TArg* __dataArg, unsigned int __sizeArg);
			__TArg& get(unsigned int __indexArg) const;
		private:
			Array();
			void Attach(__TArg* __srcArg, unsigned int __sizeArg);
			void AttachFastPass(__TArg* __srcArg, unsigned int __sizeArg);
			void CopyToOrDetach(__TArg** __destArg, unsigned int* __sizeArg);

			template <typename __HighLevelType, unsigned int __dimension>
			friend ::Platform::Array<__HighLevelType, __dimension>^ ::Platform::Details::__abi_array_attach(void* __src, unsigned int __size, bool __isFastPass, bool __needsInit);

			template <typename __HighLevelType, unsigned int __dimension>
			friend void ::Platform::Details::__abi_array_copy_to_and_release(::Platform::Array<__HighLevelType, __dimension>^ __arrArg, void** __destArg, unsigned int* __sizeArg);
			template <typename __TArg, unsigned int __dimension> friend class ArrayReference;

			void ArrayReferenceInit()
			{
				__vtable_initialize(Array<__TArg, 1>);
			}
		};

#pragma warning(pop)

} // namespace Platform

template <typename __TArg, unsigned int __dimension>
__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	inline void* __abi_winrt_ptr_ctor(const volatile ::Platform::Array<__TArg, __dimension>^ const __arrArg)
{
	__abi_IUnknown* __pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Array<__TArg, __dimension>^>(__arrArg));
	if (__pUnknown)
	{
		auto __localArray = const_cast< ::Platform::Array<__TArg, __dimension>^>(const_cast< ::Platform::Array<__TArg, __dimension>^>(__arrArg));
		if (__localArray->FastPass)
		{
			auto __ret = ref new ::Platform::Array<__TArg, __dimension>(__localArray->Data, __localArray->Length);
			__pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Array<__TArg, __dimension>^>(__ret));
		}
		else
		{
			__pUnknown->__abi_AddRef();
		}
	}
	return __pUnknown;
}

template<typename __TArg, unsigned int __dimension>
__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	void* __abi_winrt_ptr_assign(void** __ppTarget, const volatile ::Platform::Array<__TArg, __dimension> ^__arrArg)
{
	__abi_IUnknown* __pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Array<__TArg, __dimension>^>(__arrArg));
	__abi_IUnknown** __ppTargetUnknown = reinterpret_cast<__abi_IUnknown**>(__ppTarget);
	if (__pUnknown != *__ppTargetUnknown)
	{
		if (__pUnknown)
		{
			auto __localArray = const_cast< ::Platform::Array<__TArg, __dimension>^>(__arrArg);
			if (__localArray->FastPass)
			{
				auto __ret = ref new ::Platform::Array<__TArg>(__localArray->Data, __localArray->Length);
				__pUnknown = reinterpret_cast<__abi_IUnknown*>(const_cast< ::Platform::Array<__TArg, __dimension>^>(__ret));
			}
			else
			{
				__pUnknown->__abi_AddRef();
			}
		}
		if (*__ppTargetUnknown)
		{
			(*__ppTargetUnknown)->__abi_Release();
		}
		*__ppTargetUnknown = __pUnknown;
	}
	return __pUnknown;
}

template <typename __TArg>
inline __TArg __winrt_Empty_Struct()
{
	unsigned char __bytes[sizeof(__TArg)];
	__Platform_memset(__bytes, 0, sizeof(__TArg));

	return (__TArg&)__bytes;
}

struct __abi___FactoryCache
{
	__abi_IUnknown* __factory;
	void* __cookie;
};

__declspec(selectany) __abi___FactoryCache __abi_no_factory_cache = { nullptr, 0 };

struct __abi___classObjectEntry
{
	// Factory creator function
	long (__stdcall *__factoryCreator)(unsigned int*, __abi___classObjectEntry*, ::Platform::Guid&, __abi_IUnknown**);
	// Object id
	const __wchar_t* (__stdcall *__getRuntimeName)();
	// Trust level for WinRT otherwise nullptr
	int (__stdcall *__getTrustLevel)();
	// Factory cache, group id data members
	__abi___FactoryCache* __factoryCache;
	const __wchar_t* __serverName;
};

// Section r is used to put WinRT objects to creator map
#pragma section("minATL$__r", read)

__declspec(noreturn) inline void __stdcall __abi_WinRTraiseException(long __hrArg)
{
	switch (__hrArg)
	{
	case 0x80004001L: // E_NOTIMPL
		__abi_WinRTraiseNotImplementedException();

	case 0x80004002L: // E_NOINTERFACE
		__abi_WinRTraiseInvalidCastException();

	case 0x80004003L: // E_POINTER
		__abi_WinRTraiseNullReferenceException();

	case 0x80004004L: // E_ABORT
		__abi_WinRTraiseOperationCanceledException();

	case 0x80004005L: // E_FAIL
		__abi_WinRTraiseFailureException();

	case 0x80070005L: // E_ACCESSDENIED
		__abi_WinRTraiseAccessDeniedException();

	case 0x8007000EL: // E_OUTOFMEMORY
		__abi_WinRTraiseOutOfMemoryException();

	case 0x80070057L: // E_INVALIDARG
		__abi_WinRTraiseInvalidArgumentException();

	case 0x8000000BL: // E_BOUNDS
		__abi_WinRTraiseOutOfBoundsException();

	case 0x8000000CL: // E_CHANGED_STATE
		__abi_WinRTraiseChangedStateException();

	case 0x80040154L: // REGDB_E_CLASSNOTREG
		__abi_WinRTraiseClassNotRegisteredException();

	case 0x8001010EL: // RPC_E_WRONG_THREAD
		__abi_WinRTraiseWrongThreadException();
		
	case 0x80010108L: // RPC_E_DISCONNECTED
		__abi_WinRTraiseDisconnectedException();

	default:
		__abi_WinRTraiseCOMException(__hrArg);
		break;
	}
}

__declspec(non_user_code)
	::Platform::String^ __abi_winrt_CreateSystemStringFromLiteral(const __wchar_t*);
__declspec(non_user_code)
	::Platform::String^ __abi_winrt_CreateSystemStringFromLiteral(const unsigned short*);

#if defined(VCWINRT_DLL)
#define __Platform_CoCreateFreeThreadedMarshaler(__punkOuter, __ppunkMarshal) CoCreateFreeThreadedMarshaler(reinterpret_cast<IUnknown*>(__punkOuter), reinterpret_cast<IUnknown**>(__ppunkMarshal))
#endif

#if defined(VCWINRT_DLL)
#define __Platform_CoCreateFreeThreadedMarshaler(__punkOuter, __ppunkMarshal) CoCreateFreeThreadedMarshaler(reinterpret_cast<IUnknown*>(__punkOuter), reinterpret_cast<IUnknown**>(__ppunkMarshal))
#endif

// Postconditions: * ParsingInitTypes is cleared
#pragma initialize_winrt_types_phase3

#pragma region Define Common EnumResourceTypes
// Define common types if not building vccorlib.dll
#if !defined(VCWINRT_DLL)

// Function decleration for types we use from Windows and CRT
// This prevents pulling in the headers
extern "C"
{
	long __stdcall __Platform_WindowsCreateString(const ::default::char16*, unsigned int, HSTRING*);
	long __stdcall __Platform_WindowsDeleteString(HSTRING);
	long __stdcall __Platform_WindowsDuplicateString(HSTRING, HSTRING*);
	const ::default::char16* __stdcall __Platform_WindowsGetStringRawBuffer(HSTRING, unsigned int*);
	unsigned int __stdcall __Platform_WindowsGetStringLen(HSTRING);
	int __stdcall __Platform_WindowsIsStringEmpty(HSTRING);
	long __stdcall __Platform_WindowsStringHasEmbeddedNull(HSTRING, int*);
	long __stdcall __Platform_WindowsCompareStringOrdinal(HSTRING, HSTRING, int*);
	long __stdcall __Platform_WindowsCreateStringReference(const ::default::char16*, unsigned int, __Platform_Details_HSTRING_HEADER*, HSTRING*);
	long __stdcall __Platform_WindowsConcatString(HSTRING, HSTRING, HSTRING*);
	void* __stdcall __Platform_CoTaskMemAlloc(size_t);
	void __stdcall __Platform_CoTaskMemFree(void*);
	size_t __cdecl __Platform_wcslen(const ::default::char16 *);
	void * __cdecl __Platform_memset(void *, int, size_t);
}
#else // VCWINRT_DLL
#define __Platform_WindowsCreateString              WindowsCreateString
#define __Platform_WindowsDeleteString              WindowsDeleteString
#define __Platform_WindowsDuplicateString           WindowsDuplicateString
#define __Platform_WindowsGetStringRawBuffer        WindowsGetStringRawBuffer
#define __Platform_WindowsGetStringLen              WindowsGetStringLen
#define __Platform_WindowsIsStringEmpty             WindowsIsStringEmpty
#define __Platform_WindowsStringHasEmbeddedNull     WindowsStringHasEmbeddedNull
#define __Platform_WindowsCompareStringOrdinal      WindowsCompareStringOrdinal
#define __Platform_WindowsCreateStringReference     WindowsCreateStringReference
#define __Platform_WindowsConcatString              WindowsConcatString
#define __Platform_CoTaskMemAlloc                   CoTaskMemAlloc
#define __Platform_CoTaskMemFree                    CoTaskMemFree

#define __Platform_wcslen                           wcslen
#define __Platform_memset                           memset

#endif // VCWINRT_DLL

#pragma endregion

#pragma region String^ helpers
namespace Platform
{
	// Convert failure HRESULT from Windows String API's to Exception
	namespace Details
	{
		inline void CreateString(const ::default::char16* __bufferArg, unsigned int __lengthArg, HSTRING* __destArg)
		{
			__abi_ThrowIfFailed( __Platform_WindowsCreateString((const ::default::char16 *)__bufferArg, __lengthArg, __destArg) );
		}

		inline void CreateString(const ::default::char16* __sourceStringArg, HSTRING* __destArg)
		{
			__abi_ThrowIfFailed( __Platform_WindowsCreateString((const ::default::char16 *)__sourceStringArg, __sourceStringArg ? static_cast<unsigned int>(__Platform_wcslen((const ::default::char16 *)__sourceStringArg)) : 0u, __destArg) );
		}

		inline void DuplicateString(HSTRING __sourceArg, HSTRING* __destArg)
		{
			if (__sourceArg == nullptr)
			{
				*__destArg = __sourceArg;
				return;
			}

			__abi_ThrowIfFailed( __Platform_WindowsDuplicateString(__sourceArg, __destArg) );
		}

		inline void CreateStringReference(const ::default::char16* __sourceStringArg, unsigned int __lengthArg, __Platform_Details_HSTRING_HEADER* __hstringHeaderArg, HSTRING* __stringArg)
		{
			__abi_ThrowIfFailed(__Platform_WindowsCreateStringReference(__sourceStringArg, __lengthArg, __hstringHeaderArg, __stringArg));
		}
	} // namepsace Details

	// StringReference is used to hold onto a fast pass HSTRING.
	class StringReference
	{
	public:
		~StringReference()
		{
			Free();
		}
		StringReference()
		{
			Init();
		}
		StringReference(const StringReference& __fstrArg)
		{
			Init(__fstrArg);
		}
		StringReference& operator=(const StringReference& __fstrArg)
		{
			Free();
			Init(__fstrArg);
			return *this;
		}

		StringReference(const ::default::char16* __strArg)
		{
			Init(__strArg, __Platform_wcslen(__strArg));
		}
		StringReference& operator=(const ::default::char16* __strArg)
		{
			Free();
			Init(__strArg, __Platform_wcslen(__strArg));
			return *this;
		}
		StringReference(const ::default::char16* __strArg, size_t __lenArg)
		{
			Init(__strArg, __lenArg);
		}

		const ::default::char16 * Data() const
		{
			return __Platform_WindowsGetStringRawBuffer(GetHSTRING(), nullptr);
		}
		unsigned int Length() const
		{
			return __Platform_WindowsGetStringLen(GetHSTRING());
		}

		__declspec(no_release_return) __declspec(no_refcount)
			operator ::Platform::String^() const
		{
			return reinterpret_cast< ::Platform::String^>(__hString);
		}
		__declspec(no_release_return) __declspec(no_refcount)
			::Platform::String^ GetString() const
		{
			return reinterpret_cast< ::Platform::String^>(__hString);
		}
		HSTRING GetHSTRING() const
		{
			return __hString;
		}
	private:
		void Free()
		{
			__Platform_WindowsDeleteString(__hString);
		}
		void Init()
		{
			__Platform_memset(this, 0, sizeof(StringReference));
		}
		void Init(const StringReference& __fstrArg)
		{
			unsigned int __length = 0;
			const ::default::char16* __source = __Platform_WindowsGetStringRawBuffer(__fstrArg.GetHSTRING(), &__length);
			Init(__source, __length);
		}
		void Init(const ::default::char16* __strArg, unsigned __int64 __lenArg)
		{
			if ((__strArg == nullptr) || (__lenArg == 0))
				Init();
			else if (__lenArg > 0xffffffffLL) // check if it exceeds the size of an integer
				__abi_WinRTraiseInvalidArgumentException();
			else
			{
				unsigned int __length = (unsigned int) (__lenArg & 0xffffffffLL);
				::Platform::Details::CreateStringReference(__strArg, __length, &__header, &__hString);
			}
		}
		void Init(const ::default::char16* __strArg, unsigned int __lenArg)
		{
			if ((__strArg == nullptr) || (__lenArg == 0))
				Init();
			else
			{
				::Platform::Details::CreateStringReference(__strArg, __lenArg, &__header, &__hString);
			}
		}

		__Platform_Details_HSTRING_HEADER __header;
		HSTRING __hString;
	};
} // namespace Platform

__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	inline void* __abi_winrt_ptrto_string_ctor(const volatile ::Platform::String ^__strArg)
{
	if (__strArg)
	{
		HSTRING __hstr;
		auto __pRaw = reinterpret_cast<HSTRING>((const_cast< ::Platform::String^>(__strArg)));
		::Platform::Details::DuplicateString(__pRaw, &__hstr);
		return __hstr;
	}
	return nullptr;
}

__declspec(non_user_code) __declspec(no_refcount) __declspec(no_release_return)
	inline void* __abi_winrt_ptrto_string_assign(void** __ppTargetArg, const volatile ::Platform::String ^__pSourceArg)
{
	auto __pRaw = reinterpret_cast<HSTRING>((const_cast< ::Platform::String^>(__pSourceArg)));
	if ( *__ppTargetArg != reinterpret_cast<void*>(__pRaw) )
	{
		if (*__ppTargetArg)
		{
			__abi_details::__abi_delete_String(reinterpret_cast< ::Platform::String^>(*__ppTargetArg));
		}
		*__ppTargetArg = nullptr;
		if (__pSourceArg)
		{
			HSTRING __hstr;
			::Platform::Details::DuplicateString(__pRaw, &__hstr);
			*__ppTargetArg = __hstr;
		}
	}
	return *__ppTargetArg;
}

namespace __abi_details
{

	__declspec(non_user_code) __declspec(no_refcount)
		inline void __abi_delete_String(::Platform::String^ __strArg)
	{
		__Platform_WindowsDeleteString(reinterpret_cast<HSTRING>(__strArg));
	}

} // namespace __abi_details

#pragma endregion


struct __abi_type_descriptor
{
	const __wchar_t* __typeName;
	int __typeId;
};

#if !defined(VCWINRT_DLL)
__declspec(dllimport) ::Platform::Type^ __stdcall __abi_make_type_id(const __abi_type_descriptor&);
#else
__declspec(dllexport) ::Platform::Type^ __stdcall __abi_make_type_id(const __abi_type_descriptor&);
#endif

inline Platform::String^ __abi_CustomToString(void*)
{
    return nullptr;
}

namespace Platform
{
	namespace Details
	{
		__declspec(dllexport) long __stdcall ReCreateFromException(::Platform::Exception^);
		__declspec(dllexport) ::Platform::Object^ __stdcall CreateValue(::Platform::Type^, const void*);
		__declspec(dllexport) void* __stdcall GetIBoxArrayVtable(void*);
		__declspec(dllexport) void* __stdcall GetIBoxVtable(void*);

        template<typename T>
        ref class 
            __declspec(no_empty_identity_interface)
        CustomBox sealed :
        public [::Windows::Foundation::Metadata::Default] [::Platform::Metadata::RuntimeClassName] ::Platform::IBox<T>,
        public ::Platform::Details::IPrintable
        {
            T value_;
        public:
            CustomBox(T value) : value_(value)
            {
                *reinterpret_cast<void**>(static_cast< ::Platform::IValueType^>(this)) =
                    GetIBoxVtable(reinterpret_cast<void*>(static_cast< ::Platform::IBox<T>^>(this)));
            }

            virtual property T Value
            {
                T get()
                {
                    return value_;
                }
            }

            virtual Platform::String^ ToString()
            {
                if (__is_enum(T))
                {
                    String^ s = ::__abi_CustomToString(&value_);
                    if (s)
                    {
                        return s;
                    }
                    return T::typeid->FullName;
                }
                else
                {
                    return ::__abi_details::__abi_ObjectToString(this, false);
                }
            }
        };

		ref class CustomValueType : public ::Platform::ValueType
		{
		};

		ref class CustomEnum : public ::Platform::Enum
		{
		};

		template<bool __isEnum>
		struct BoxValueType
		{
			typedef CustomValueType Type;
		};

		template<>
		struct BoxValueType<true>
		{
			typedef CustomEnum Type;
		};

		template<typename __TArg>
		struct RemoveConst
		{
			typedef __TArg Type;
		};

		template<typename __TArg>
		struct RemoveConst<const __TArg>
		{
			typedef __TArg Type;
		};
		
		template<typename __TArg>
		struct RemoveVolatile
		{
			typedef __TArg Type;
		};

		template<typename __TArg>
		struct RemoveVolatile<volatile __TArg>
		{
			typedef __TArg Type;
		};
		
		template<typename __TArg>
		struct RemoveCV
		{
			typedef typename RemoveConst<typename RemoveVolatile<__TArg>::Type>::Type Type;
		};
	} // namespace Details

	template<typename __TArg>
	ref class
		__declspec(one_phase_constructed)
		__declspec(layout_as_external)
		__declspec(no_empty_identity_interface)
		Box abstract :
	public ::Platform::IBox<typename ::Platform::Details::RemoveCV<__TArg>::Type>,    
		public Details::BoxValueType<__is_enum(__TArg)>::Type   
	{
		static_assert(__is_enum(__TArg) || __is_value_class(__TArg) || __is_trivial(__TArg), "__TArg type of Box<__TArg> must be either value type or enum type");
		
		typedef typename ::Platform::Details::RemoveCV<__TArg>::Type __TBoxValue;
	internal:
		Box(__TBoxValue __valueArg)
		{
			::Platform::Object ^__boxValue = Details::CreateValue(__TBoxValue::typeid, &__valueArg);
			if (__boxValue == nullptr)
			{
				__boxValue = ref new Details::CustomBox<__TBoxValue>(__valueArg);
				return reinterpret_cast<Box^>(__boxValue);
			}

			return dynamic_cast<Box^>(__boxValue);
		}

		operator __TBoxValue()
		{
			return safe_cast< ::Platform::IBox<__TBoxValue>^>(this)->Value;
		}

		operator Box<const __TBoxValue>^()
		{
			return reinterpret_cast<Box<const __TBoxValue>^>(this);
		}

		operator Box<volatile __TBoxValue>^()
		{
			return reinterpret_cast<Box<volatile __TBoxValue>^>(this);
		}

		operator Box<const volatile __TBoxValue>^()
		{
			return reinterpret_cast<Box<const volatile __TBoxValue>^>(this);
		}

		static operator Box<__TArg>^( ::Platform::IBox<__TArg>^ __boxValueArg)
		{
			return reinterpret_cast<Box<__TArg>^>(__boxValueArg);
		}

	public:
		virtual property __TBoxValue Value
		{
			__TBoxValue get()
			{
				return safe_cast< ::Platform::IBox<__TBoxValue>^>(this)->Value;
			}
		}
	};

	////////////////////////////////////////////////////////////////////////////////
	inline Guid::Guid() : __a(0), __b(0), __c(0), __d(0), __e(0), __f(0), __g(0), __h(0), __i(0), __j(0), __k(0)
	{
	}

	inline Guid::Guid(__rcGUID_t __guid) :
	__a(reinterpret_cast<const __s_GUID&>(__guid).Data1),
		__b(reinterpret_cast<const __s_GUID&>(__guid).Data2),
		__c(reinterpret_cast<const __s_GUID&>(__guid).Data3),
		__d(reinterpret_cast<const __s_GUID&>(__guid).Data4[0]),
		__e(reinterpret_cast<const __s_GUID&>(__guid).Data4[1]),
		__f(reinterpret_cast<const __s_GUID&>(__guid).Data4[2]),
		__g(reinterpret_cast<const __s_GUID&>(__guid).Data4[3]),
		__h(reinterpret_cast<const __s_GUID&>(__guid).Data4[4]),
		__i(reinterpret_cast<const __s_GUID&>(__guid).Data4[5]),
		__j(reinterpret_cast<const __s_GUID&>(__guid).Data4[6]),
		__k(reinterpret_cast<const __s_GUID&>(__guid).Data4[7])
	{
	}

	inline Guid::operator ::__rcGUID_t()
	{
		return reinterpret_cast<__rcGUID_t>(*this);
	}

	inline bool ::Platform::Guid::Equals(::Platform::Guid __guidArg)
	{
		return (
			((unsigned long *) this)[0] == ((unsigned long *) &__guidArg)[0] &&
			((unsigned long *) this)[1] == ((unsigned long *) &__guidArg)[1] &&
			((unsigned long *) this)[2] == ((unsigned long *) &__guidArg)[2] &&
			((unsigned long *) this)[3] == ((unsigned long *) &__guidArg)[3]);
	}

	inline bool ::Platform::Guid::Equals(__rcGUID_t __guidArg)
	{
		return (
			((unsigned long *) this)[0] == ((unsigned long *) &__guidArg)[0] &&
			((unsigned long *) this)[1] == ((unsigned long *) &__guidArg)[1] &&
			((unsigned long *) this)[2] == ((unsigned long *) &__guidArg)[2] &&
			((unsigned long *) this)[3] == ((unsigned long *) &__guidArg)[3]);
	}

	inline bool ::Platform::Guid::operator==(::Platform::Guid __aArg, ::Platform::Guid __bArg)
	{
		return (
			((unsigned long *) &__aArg)[0] == ((unsigned long *) &__bArg)[0] &&
			((unsigned long *) &__aArg)[1] == ((unsigned long *) &__bArg)[1] &&
			((unsigned long *) &__aArg)[2] == ((unsigned long *) &__bArg)[2] &&
			((unsigned long *) &__aArg)[3] == ((unsigned long *) &__bArg)[3]);
	}

	inline bool ::Platform::Guid::operator!=(::Platform::Guid __aArg, ::Platform::Guid __bArg)
	{
		return !(
			((unsigned long *) &__aArg)[0] == ((unsigned long *) &__bArg)[0] &&
			((unsigned long *) &__aArg)[1] == ((unsigned long *) &__bArg)[1] &&
			((unsigned long *) &__aArg)[2] == ((unsigned long *) &__bArg)[2] &&
			((unsigned long *) &__aArg)[3] == ((unsigned long *) &__bArg)[3]);
	}

	inline Guid::Guid(unsigned int __aArg, unsigned short __bArg, unsigned short __cArg, unsigned __int8 __dArg, 
		unsigned __int8 __eArg, unsigned __int8 __fArg, unsigned __int8 __gArg, unsigned __int8 __hArg, 
		unsigned __int8 __iArg, unsigned __int8 __jArg, unsigned __int8 __kArg) :
	__a(__aArg), __b(__bArg), __c(__cArg), __d(__dArg), __e(__eArg), __f(__fArg), __g(__gArg), __h(__hArg), __i(__iArg), __j(__jArg), __k(__kArg)
	{
	}

	inline Guid::Guid(unsigned int __aArg, unsigned short __bArg, unsigned short __cArg, ::Platform::Array<unsigned __int8>^ __dArg) :
	__a(__aArg), __b(__bArg), __c(__cArg)
	{
		if (__dArg->Length != 8)
		{
			__abi_WinRTraiseInvalidArgumentException();
		}
		__d = __dArg[0];
		__e = __dArg[1];
		__f = __dArg[2];
		__g = __dArg[3];
		__h = __dArg[4];
		__i = __dArg[5];
		__j = __dArg[6];
		__k = __dArg[7];
	}

	__declspec(selectany) ::Platform::Guid __winrt_GUID_NULL(0x00000000, 0x0000, 0x0000, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

	////////////////////////////////////////////////////////////////////////////////
	inline void* Details::Heap::Allocate(::Platform::SizeT /*__sizeArg*/, void* __pPlacementArg)
	{
		return __pPlacementArg;
	}

	inline void Details::Heap::PlacementFree(void* /*__pArg*/, void* /*__placementArg*/)
	{
	}

} // namespace Platform

template <typename __TArg>
Platform::Box<typename ::Platform::Details::RemoveCV<__TArg>::Type>^ __abi_create_box(__TArg __boxValueArg)
{
	return ref new ::Platform::Box<__TArg>(__boxValueArg);
}

template <typename __TArg>
__TArg __abi_unbox(::Platform::Object^ __objArg)
{
	return safe_cast< ::Platform::Box<__TArg>^>(__objArg);
}

#pragma region String
#pragma region String^ API
namespace Platform
{
	__declspec(no_refcount)
		__declspec(no_release_return)
		inline String::String()
	{
		return nullptr;
	}

	__declspec(no_refcount)
		inline String::String(HSTRING __hstrArg)
	{
		HSTRING __hstr;
		Details::DuplicateString(__hstrArg, &__hstr);
		return *reinterpret_cast<String^*>(&__hstr);
	}

	__declspec(no_refcount)
		inline String::String(const ::default::char16* __strArg)
	{
		if (__strArg == nullptr)
			return nullptr;

		HSTRING __hstr = nullptr;
		::Platform::Details::CreateString(__strArg, (unsigned int)__Platform_wcslen(__strArg), &__hstr);
		return *reinterpret_cast<String^*>(&__hstr);
	}

	__declspec(no_refcount)
		inline String::String(const ::default::char16* __strArg, unsigned int __lenArg)
	{
		if ((__strArg == nullptr) || (__lenArg == 0))
			return nullptr;

		HSTRING __hstr = nullptr;
		::Platform::Details::CreateString(__strArg, __lenArg, &__hstr);
		return *reinterpret_cast<String^*>(&__hstr);
	}

	inline const ::default::char16* String::Begin()
	{
		return Data();
	}
	inline const ::default::char16* String::End()
	{
		return Data() + Length();
	}

	inline Platform::String^ Object::ToString()
	{
		return ::__abi_details::__abi_ObjectToString(this, true);
	}

	inline const ::default::char16 * String::Data()
	{
		return __Platform_WindowsGetStringRawBuffer(reinterpret_cast<HSTRING>(this), nullptr);
	}

	inline unsigned int String::Length()
	{
		return __Platform_WindowsGetStringLen(reinterpret_cast<HSTRING>(this));
	}
	inline bool String::IsEmpty()
	{
		return __Platform_WindowsIsStringEmpty(reinterpret_cast<HSTRING>(this)) ? true : false;
	}

	inline bool String::IsFastPass()
	{
		return false;
	}

	inline bool String::Equals(Object^ __strArg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(this), reinterpret_cast<HSTRING>(((String^)__strArg)), &__result);
		return (__result == 0);
	}

	inline bool String::Equals(String^ __str1Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(this), reinterpret_cast<HSTRING>(__str1Arg), &__result);
		return (__result == 0);
	}
	inline int String::CompareOrdinal(String^ __str1Arg, String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return __result;
	}
	inline int String::GetHashCode()
	{
		int __hash = 0;
		for (auto i = Begin(); i != End(); ++i)
			__hash += *i;
		return __hash;
	}

	inline bool ::Platform::String::operator==(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return (__result == 0);
	}

	inline bool ::Platform::String::operator!=(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return (__result != 0);
	}

	inline bool ::Platform::String::operator<(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return (__result < 0);
	}

	inline bool ::Platform::String::operator>(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return (__result > 0);
	}

	inline bool ::Platform::String::operator<=(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return (__result <= 0);
	}

	inline bool ::Platform::String::operator>=(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		int __result = 0;
		__Platform_WindowsCompareStringOrdinal(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__result);
		return (__result >= 0);
	}

	inline class ::Platform::String ^ ::Platform::String::Concat(::Platform::String^ __str1Arg, ::Platform::String^ __str2Arg)
	{
		HSTRING __hstr = nullptr;
		__abi_ThrowIfFailed(__Platform_WindowsConcatString(reinterpret_cast<HSTRING>(__str1Arg), reinterpret_cast<HSTRING>(__str2Arg), &__hstr));
		return *reinterpret_cast<String^*>(&__hstr);
	}

	inline String^ String::ToString()
	{
		return ref new ::Platform::String(Data());
	}

#pragma region string iterators
	inline const ::default::char16 * begin(::Platform::String^ __strArg)
	{
		return __strArg->Begin();
	}

	inline const ::default::char16 * end(::Platform::String^ __strArg)
	{
		return __strArg->End();
	}
#pragma endregion
} // namespace Platform
#pragma endregion

__declspec(non_user_code)
	inline ::Platform::String^ __abi_winrt_CreateSystemStringFromLiteral(const ::default::char16* __strArg)
{
	return ref new ::Platform::String(__strArg);
}

template<typename __TArg>
inline ::Platform::Array<__TArg, 1>^ __abi_winrt_CreateArray(unsigned int __sizeArg)
{
	return ref new ::Platform::Array<__TArg, 1>(__sizeArg);
}

inline __declspec(no_refcount) __declspec(no_release_return) ::Platform::String^ __abi_winrt_CreateFastPassSystemStringFromLiteral(const ::default::char16* __strArg, unsigned int __lenArg, __Platform_Details_HSTRING_HEADER* __psheaderArg)
{
	HSTRING __hstr;
	::Platform::Details::CreateStringReference(__strArg, __lenArg, __psheaderArg, &__hstr);
	return *reinterpret_cast< ::Platform::String^*>(&__hstr);
}

inline __declspec(no_refcount) __declspec(no_release_return) ::Platform::String^ __abi_winrt_CreateFastPassSystemStringFromLiteral(const unsigned short* __strArg, unsigned int __lenArg, __Platform_Details_HSTRING_HEADER* __psheaderArg)
{
	return __abi_winrt_CreateFastPassSystemStringFromLiteral(reinterpret_cast<const ::default::char16*>(__strArg), __lenArg, __psheaderArg);
}

#pragma endregion

#pragma region Array

namespace Platform 
{ 
	namespace Details 
	{
		// Attach to the external buffer
		template <typename __HighLevelType, unsigned int __dimension>
		__declspec(no_refcount) ::Platform::Array<__HighLevelType, __dimension>^ __abi_array_attach(void* __srcArg, unsigned int __elementcountArg, bool __isFastPassArg, bool __needsInitArg)
		{
			if (static_cast<unsigned int>(-1) / sizeof(__HighLevelType) < __elementcountArg)
			{
				__abi_WinRTraiseInvalidArgumentException();
			}
			auto __arr = ref new ::Platform::Array<__HighLevelType, __dimension>(nullptr, 0);

			if (__needsInitArg)
			{
				__Platform_memset(__srcArg, 0, sizeof(__HighLevelType) * __elementcountArg);
			}
			if (!__isFastPassArg)
			{
				__arr->Attach(reinterpret_cast<__HighLevelType*>(__srcArg), __elementcountArg);
			}
			else
			{
				__arr->AttachFastPass(reinterpret_cast<__HighLevelType*>(__srcArg), __elementcountArg);
			}

			return __arr;
		}

		template <typename __HighLevelType, unsigned int __dimension>
		void __abi_array_copy_to_and_release(::Platform::Array<__HighLevelType, __dimension>^ __arrArg, void** __destArg, unsigned int* __sizeArg)
		{
			if (__arrArg == nullptr)
			{
				*__destArg = nullptr;
				*__sizeArg = 0;
				return;
			}

			__HighLevelType **__destHigh = reinterpret_cast<__HighLevelType **>(__destArg);
			__arrArg->CopyToOrDetach(__destHigh, __sizeArg);
			// The caller will not use arr after this function returns
		}

		// Convert ::Platform::Array to raw buffer
		// It is called when we are converting in Array from high level to low level
		template <typename __LowLevelType, typename __HighLevelType, unsigned int __dimension>
		__LowLevelType* __abi_array_to_raw(const ::Platform::Array<__HighLevelType, __dimension>^ __arrArg)
		{
			if (__arrArg == nullptr)
			{
				return nullptr;
			}
			return reinterpret_cast<__LowLevelType*>(__arrArg->Data);
		}

		template <typename __TArg>
		struct array_helper<__TArg, true>
		{
			static void DestructElementsAndFree(__TArg* __srcArg, unsigned int)
			{
				__Platform_CoTaskMemFree(__srcArg);
			}
		};

		template <typename __TArg>
		struct array_helper<__TArg, false>
		{
			static void DestructElementsAndFree(__TArg* __srcArg, unsigned int __countArg)
			{
				typedef __TArg typeT;
				for (unsigned int __i = 0; __i < __countArg; __i++)
				{
					(&__srcArg[__i])->~typeT();
				}
				__Platform_CoTaskMemFree(__srcArg);
			}
		};    
	} // namespace Details

	template <typename __TArg>
	class ArrayReference<__TArg, 1>
	{
		default::uint8 __data[sizeof(Array<__TArg>)];
			void Init(__TArg* __dataArg, unsigned int __sizeArg, bool __needsInitArg = false)
			{
				__Platform_memset(__data, 0, sizeof(Array<__TArg>));
				ArrayReference* __pThis = this;
				Array<__TArg>^* __pArrayThis = reinterpret_cast<Array<__TArg>^*>(&__pThis);
				(*__pArrayThis)->ArrayReferenceInit();

				if (__needsInitArg)
				{
					if (static_cast<unsigned int>(-1) / sizeof(__TArg) < __sizeArg)
					{
						__abi_WinRTraiseInvalidArgumentException();
					}
					__Platform_memset(__dataArg, 0, sizeof(__TArg) * __sizeArg);
				}

				(*__pArrayThis)->AttachFastPass(__dataArg, __sizeArg);
			}
	public:
		ArrayReference(__TArg* __dataArg, unsigned int __sizeArg, bool __needsInitArg = false)
		{
			Init(__dataArg, __sizeArg, __needsInitArg);
		}

		ArrayReference(ArrayReference&& __otherArg)
		{
			Array<__TArg>^* __pOther = reinterpret_cast<Array<__TArg>^*>(&__otherArg);
			Init((*__pOther)->__data, (*__pOther)->__size);
		}

		ArrayReference& operator=(ArrayReference&& __otherArg)
		{
			Array<__TArg>^* __pOther = reinterpret_cast<Array<__TArg>^*>(&__otherArg);
			Init((*pOther)->__data, (*pOther)->__size);
		}

		__declspec(no_refcount) __declspec(no_release_return)
			operator Array<__TArg>^()
		{
			ArrayReference* __pThis = this;
			Array<__TArg>^* __pArrayThis = reinterpret_cast<Array<__TArg>^*>(&__pThis);

			return *__pArrayThis;
		}
	private:
		ArrayReference(const ArrayReference&);
		ArrayReference& operator=(const ArrayReference&);
	};

#pragma region ::Platform::WriteOnlyArray
	template <typename __TArg>
	inline WriteOnlyArray<__TArg, 1>::WriteOnlyArray() : __size(0), __fastpassflag(false), __data(nullptr)
	{
	}

	template <typename __TArg>
	inline WriteOnlyArray<__TArg, 1>::WriteOnlyArray(unsigned int __sizeArg) : __size(0), __fastpassflag(false), __data(nullptr)
	{
		if (__sizeArg == 0)
		{
			return;
		}
		__data = AllocateAndZeroInitialize(__sizeArg);
		__size = __sizeArg;
	}

	template <typename __TArg>
	inline WriteOnlyArray<__TArg, 1>::WriteOnlyArray(__TArg* __dataArg, unsigned int __sizeArg) : __size(0), __fastpassflag(false), __data(nullptr)
	{
		if (__sizeArg == 0)
		{
			return;
		}
		__data = AllocateAndCopyElements(__dataArg, __sizeArg);
		__size = __sizeArg;
	}

	template <typename __TArg>
	inline WriteOnlyArray<__TArg, 1>::~WriteOnlyArray()
	{
		if ((__fastpassflag == false) && (__data != nullptr))
		{
			::Platform::Details::array_helper<__TArg>::DestructElementsAndFree(__data, __size);
		}
		Clear();
	}

	template <typename __TArg>
	inline __TArg& WriteOnlyArray<__TArg, 1>::set(unsigned int __positionArg, __TArg __valueArg)
	{
		if (__data == nullptr)
		{
			__abi_WinRTraiseNullReferenceException();
		}

		if (__positionArg >= 0 && __positionArg < __size)
		{
			__data[__positionArg] = __valueArg;
			return __data[__positionArg];
		}

		__abi_WinRTraiseOutOfBoundsException();
	}

	template <typename __TArg>
	inline __TArg& WriteOnlyArray<__TArg, 1>::get(unsigned int __positionArg) const
	{
		if (__data == nullptr)
		{
			__abi_WinRTraiseNullReferenceException();
		}

		if (__positionArg >= 0 && __positionArg < __size)
		{
			return __data[__positionArg];
		}

		__abi_WinRTraiseOutOfBoundsException();
	}

	template <typename __TArg>
	inline __TArg* WriteOnlyArray<__TArg, 1>::begin() const
	{
		if (__data == nullptr)
		{
			return nullptr;
		}

		return &(__data[0]);
	}

	template <typename __TArg>
	inline __TArg* WriteOnlyArray<__TArg, 1>::end() const
	{
		if (__data == nullptr)
		{
			return nullptr;
		}

		return &(__data[__size]);
	}

	template <typename __TArg>
	inline unsigned int  WriteOnlyArray<__TArg, 1>::Length::get() const
	{
		return __size;
	}

	template <typename __TArg>
	inline __TArg* WriteOnlyArray<__TArg, 1>::Data::get() const
	{
		return this->begin();
	}

	template <typename __TArg>
	inline bool WriteOnlyArray<__TArg, 1>::FastPass::get() const
	{
		return __fastpassflag;
	}

	template <typename __TArg>
	inline void WriteOnlyArray<__TArg, 1>::Clear()
	{
		__size = 0;
		__fastpassflag = false;
		__data = nullptr;
	}

	template <typename __TArg>
	inline __TArg* WriteOnlyArray<__TArg, 1>::AllocateAndZeroInitialize(unsigned int __countArg)
	{
		__TArg* __dest = nullptr;
		if (__countArg == 0)
		{
			return __dest;
		}

		if (static_cast<unsigned int>(-1) / sizeof(__TArg) < __countArg)
		{
			__abi_WinRTraiseInvalidCastException();
		}
		__dest = (__TArg*)__Platform_CoTaskMemAlloc(__countArg * sizeof(__TArg));
		if (__dest == nullptr)
		{
			__abi_WinRTraiseOutOfMemoryException();
		}

		__Platform_memset(__dest, 0, __countArg * sizeof(__TArg));
		return __dest;
	}

	template <typename __TArg>
	inline __TArg* WriteOnlyArray<__TArg, 1>::AllocateAndCopyElements(const __TArg* __srcArg, unsigned int __countArg)
	{
		__TArg* __dest = AllocateAndZeroInitialize(__countArg);
		for (unsigned int __i = 0; __i < __countArg; ++__i)
		{
			__dest[__i] = __srcArg[__i];
		}
		return __dest;
	}
#pragma endregion

#pragma region ::Platform::Array
	template <typename __TArg>
	inline Array<__TArg, 1>::Array() : WriteOnlyArray()
	{
		*reinterpret_cast<void**>(static_cast< ::Platform::IValueType^ >(this)) = 
			Details::GetIBoxArrayVtable(reinterpret_cast<void*>(static_cast< ::Platform::IBoxArray<__TArg>^ >(this)));                
	}

	template <typename __TArg>
	inline Array<__TArg, 1>::Array(const Array<__TArg, 1>^ __source) : WriteOnlyArray(__source ? __source->Data : nullptr, __source ? __source->Length : 0)
	{
		*reinterpret_cast<void**>(static_cast< ::Platform::IValueType^ >(this)) = 
			Details::GetIBoxArrayVtable(reinterpret_cast<void*>(static_cast< ::Platform::IBoxArray<__TArg>^ >(this)));
    }

	template <typename __TArg>
	inline Array<__TArg, 1>::Array(unsigned int __sizeArg) : WriteOnlyArray(__sizeArg)
	{
		*reinterpret_cast<void**>(static_cast< ::Platform::IValueType^ >(this)) = 
			Details::GetIBoxArrayVtable(reinterpret_cast<void*>(static_cast< ::Platform::IBoxArray<__TArg>^ >(this)));
	}

	template <typename __TArg>
	inline Array<__TArg, 1>::Array(__TArg* __dataArg, unsigned int __sizeArg) : WriteOnlyArray(__dataArg, __sizeArg)
	{
		*reinterpret_cast<void**>(static_cast< ::Platform::IValueType^ >(this)) = 
			Details::GetIBoxArrayVtable(reinterpret_cast<void*>(static_cast< ::Platform::IBoxArray<__TArg>^>(this)));
	}

	template <typename T>
	inline T& Array<T, 1>::get(unsigned int __positionArg) const
	{
		return WriteOnlyArray<T, 1>::get(__positionArg);
	}

	template <typename __TArg>
	inline Array<__TArg, 1>^ Array<__TArg, 1>::Value::get()
	{
		return this;
	}

	template <typename __TArg>
	inline void Array<__TArg, 1>::Attach(__TArg* __srcArg, unsigned int __sizeArg)
	{
		// Precondition:
		//    default constructed object
		// Postcondition:
		//    _data = src
		//    _size = size
		//    _fastpassflag = false
		//    _refcount = 1

		if (__size == 0 && __data == nullptr)
		{
			__size = __sizeArg;
			__fastpassflag = false;
			__data = __srcArg;
			return;
		}

		__abi_WinRTraiseFailureException();
	}

	template <typename __TArg>
	inline void Array<__TArg, 1>::AttachFastPass(__TArg* __srcArg, unsigned int __sizeArg)
	{
		// Precondition:
		//    default constructed object
		// Postcondition:
		//    _data = src
		//    _size = size
		//    _fastpassflag = true

		if (__size == 0 && __data == nullptr)
		{
			__size = __sizeArg;
			__fastpassflag = true;
			__data = __srcArg;
			return;
		}

		__abi_WinRTraiseFailureException();
	}

	template <typename __TArg>
	inline void Array<__TArg, 1>::CopyToOrDetach(__TArg** __destArg, unsigned int* __sizeArg)
	{
		// Postcondition:
		//    if (_refcount == 1 && !_fastpassflag)
		//        *dest = _data
		//        *size = _size
		//          Clear()
		//    if (_refcount > 1 || _fastpassflag)
		//        *dest = new buffer with contents of _data
		//        *size = _size

		if ((__destArg == nullptr) || (__sizeArg == nullptr))
		{
			__abi_WinRTraiseNullReferenceException();
		}

		if (__size == 0)
		{
			*__destArg = nullptr;
			*__sizeArg = 0;
			return;
		}

		if(__data == nullptr)
		{
			__abi_WinRTraiseFailureException();
		}

		if (!__fastpassflag && __abi_reference_count.Get() == 1)
		{
			*__destArg = __data;
			*__sizeArg = __size;
			Clear();
		}
		else if (__fastpassflag || __abi_reference_count.Get() > 1)
		{
			*__sizeArg = __size;
			*__destArg = AllocateAndCopyElements(__data, __size);
		}
		else
		{
			__abi_WinRTraiseFailureException();
		}
	}

#pragma endregion

#pragma region Array iterators
	template<class __TArg>
	__TArg * begin(const Array<__TArg, 1>^ __arrArg)
	{
		return __arrArg->begin();
	}

	template<class __TArg>
	__TArg * end(const Array<__TArg, 1>^ __arrArg)
	{
		return __arrArg->end();
	}
#pragma endregion
} // namespace Platform {
#pragma endregion

namespace Platform 
{
	namespace Details 
	{
#if !defined(VCWINRT_DLL)
		__declspec(dllimport) void __stdcall EventSourceInitialize(void**);
		__declspec(dllimport) void __stdcall EventSourceUninitialize(void**);
		__declspec(dllimport) void* __stdcall EventSourceGetTargetArray(void*, EventLock*);
		__declspec(dllimport) ::Windows::Foundation::EventRegistrationToken __stdcall EventSourceAdd(void**, EventLock*, ::Platform::Delegate^);
		__declspec(dllimport) void __stdcall EventSourceRemove(void**, EventLock*, ::Windows::Foundation::EventRegistrationToken);
		__declspec(dllimport) __abi_IUnknown* __stdcall GetWeakReference(const volatile ::Platform::Object^ const other);
		__declspec(dllimport) __declspec(no_refcount) ::Platform::Object^ __stdcall ResolveWeakReference(const ::_GUID& guid, __abi_IUnknown** weakRef);
#else
		__declspec(dllexport) void __stdcall EventSourceInitialize(void**);
		__declspec(dllexport) void __stdcall EventSourceUninitialize(void**);
		__declspec(dllexport) void* __stdcall EventSourceGetTargetArray(void*, EventLock*);
		__declspec(dllexport) ::Windows::Foundation::EventRegistrationToken __stdcall EventSourceAdd(void**, EventLock*, ::Platform::Delegate^);
		__declspec(dllexport) void __stdcall EventSourceRemove(void**, EventLock*, ::Windows::Foundation::EventRegistrationToken);
		__declspec(dllexport) __abi_IUnknown* __stdcall GetWeakReference(const volatile ::Platform::Object^ const other);
		__declspec(dllexport) __declspec(no_refcount) ::Platform::Object^ __stdcall ResolveWeakReference(const ::_GUID& guid, __abi_IUnknown** weakRef);
#endif
	} // Details

	class EventSource
	{
	public:
		EventSource()
		{
			Details::EventSourceInitialize(&__targets);
		}

		~EventSource()
		{
			Details::EventSourceUninitialize(&__targets);
		}

		::Windows::Foundation::EventRegistrationToken Add(Details::EventLock* __lockArg, ::Platform::Object^ __delegateInterfaceArg)
		{
			return Details::EventSourceAdd(&__targets, __lockArg, reinterpret_cast<::Platform::Delegate^>(__delegateInterfaceArg));
		}

		void Remove(Details::EventLock* __lockArg, ::Windows::Foundation::EventRegistrationToken __tokenArg)
		{
			Details::EventSourceRemove(&__targets, __lockArg, __tokenArg);
		}
	private:
		// __TInvokeMethod is a functor that performs the appropriate invoke, depending on the
		// number of arguments specified.
		template <typename __TDelegate, typename __TReturnType, typename __TInvokeMethod>
		typename __TReturnType DoInvoke(Details::EventLock* __lockArg, __TInvokeMethod __invokeOneArg)
		{
			// lock pointer exhange
			//        targets = _targets
			// unlock pointer exhange
			// iterate all targets and do invoke

			// The _targetsPointerLock protects the acquisition of an AddRef'd pointer to
			// "current list".  An Add/Remove operation may occur during the
			// firing of events (but occurs on a copy of the list).  i.e. both
			// DoInvoke/invoke and Add/Remove are readers of the "current list".
			// NOTE:  EventSource::Invoke(...) must never take the _addRemoveLock.
			::Platform::Array< ::Platform::Delegate^>^ __targetsLoc;
			// Attaching Array without AddRef'ing
			*reinterpret_cast<void**>(&__targetsLoc) = Details::EventSourceGetTargetArray(__targets, __lockArg);

			typename __TReturnType __returnVal = typename __TReturnType();
			// The list may not exist if nobody has registered
			if (__targetsLoc != nullptr)
			{
				const unsigned int __size = __targetsLoc->Length;

				for (unsigned int __index = 0; __index < __size; __index++)
				{
					__TDelegate^ __element = reinterpret_cast<__TDelegate^>(__targetsLoc[__index]);
					try
					{
						__returnVal = (__invokeOneArg)(__element);
					}
					catch(::Platform::DisconnectedException^)
					{
						::Windows::Foundation::EventRegistrationToken __token;
						void* __pv = reinterpret_cast<void*>(__element);
						__token.Value = reinterpret_cast<__int64>(__pv);
						Details::EventSourceRemove(&__targets, __lockArg, __token);
					}
				}
			}
			return __returnVal;
		}

		// __TInvokeMethod is a functor that performs the appropriate invoke, depending on the
		// number of arguments specified.
		template <typename __TDelegate, typename __TInvokeMethod>
		void DoInvokeVoid(Details::EventLock* __lockArg, __TInvokeMethod __invokeOneArg)
		{
			// lock pointer exhange
			//        targets = _targets
			// unlock pointer exhange
			// iterate all targets and do invoke

			// The _targetsPointerLock protects the acquisition of an AddRef'd pointer to
			// "current list".  An Add/Remove operation may occur during the
			// firing of events (but occurs on a copy of the list).  i.e. both
			// Invoke/invoke and Add/Remove are readers of the "current list".
			// NOTE:  EventSource::Invoke(...) must never take the _addRemoveLock.
			::Platform::Array< ::Platform::Delegate^>^ __targetsLoc;
			// Attaching Array without AddRef'ing
			*reinterpret_cast<void**>(&__targetsLoc) = Details::EventSourceGetTargetArray(__targets, __lockArg);

			// The list may not exist if nobody has registered
			if (__targetsLoc != nullptr)
			{
				const unsigned int __size = __targetsLoc->Length;

				for (unsigned int __index = 0; __index < __size; __index++)
				{
					__TDelegate^ __element = reinterpret_cast<__TDelegate^>(__targetsLoc[__index]);
					try
					{
						(__invokeOneArg)(__element);
					}
					catch(::Platform::DisconnectedException^)
					{
						::Windows::Foundation::EventRegistrationToken __token;
						void* __pv = reinterpret_cast<void*>(__element);
						__token.Value = reinterpret_cast<__int64>(__pv);
						Details::EventSourceRemove(&__targets, __lockArg, __token);
					}
				}
			}
		}

	public:
		template < typename __TLambda >
		void InvokeVoid(Details::EventLock* __lockArg)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [](__TLambda^ __lambda) -> void { __lambda(); });
		}

		template < typename __TLambda, typename __TArg0 > void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0](__TLambda^ __lambda) -> void {
				__lambda(__arg0);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1 >
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2 >
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3>
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4>
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3, ____arg4](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3, __arg4);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5>
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 ____arg4, __TArg5 __arg5)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3, ____arg4, ____arg5](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6>
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7>
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6, __TArg7 __arg7)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7, typename __TArg8>
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6, __TArg7 __arg7, __TArg8 __arg8)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8);
			});
		}

		template < typename __TLambda, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7, typename __TArg8, typename __TArg9 >
		void InvokeVoid(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6, __TArg7 __arg7, __TArg8 __arg8, __TArg9 __arg9)
		{
			DoInvokeVoid<__TLambda>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8, __arg9](__TLambda^ __lambda) -> void {
				__lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8, __arg9);
			});
		}

		template < typename __TLambda, typename __TReturnType >
		typename __TReturnType Invoke(Details::EventLock* __lockArg)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [](__TLambda^ __lambda) -> typename __TReturnType { return __lambda(); });
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0 >
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1 >
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2 >
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3>
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4>
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3, __arg4);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5>
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6>
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7>
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6, __TArg7 __arg7)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7, typename __TArg8>
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6, __TArg7 __arg7, __TArg8 __arg8)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8);
			});
		}

		template < typename __TLambda, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7, typename __TArg8, typename __TArg9 >
		__TReturnType Invoke(Details::EventLock* __lockArg, __TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 ____arg5, __TArg6 __arg6, __TArg7 __arg7, __TArg8 __arg8, __TArg9 __arg9)
		{
			return DoInvoke<__TLambda, __TReturnType>(__lockArg, [__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8, __arg9](__TLambda^ __lambda) -> __TReturnType{
				return __lambda(__arg0, __arg1, __arg2, __arg3, __arg4, ____arg5, __arg6, __arg7, __arg8, __arg9);
			});
		}
	protected:
		void* __targets;
	};

	class Module
	{
	public:
		static void __stdcall RunServer(const ::default::char16* __serverName = nullptr);
		static ::Platform::Details::IActivationFactory^ __stdcall GetActivationFactory(::Platform::String^);
		static bool __stdcall CanUnloadNow();
	};

	inline ::default::int32 Exception::HResult::get()
	{
		return __hresult;
	}

	class WeakReference
	{
	private:
		__abi_IUnknown* __weakPtr;

		void InternalAddRef()
		{
			if (__weakPtr != nullptr)
			{
				__weakPtr->__abi_AddRef();
			}
		}

		void InternalRelease()
		{
			__abi_IUnknown* __tmp = __weakPtr;
			if (__tmp != nullptr)
			{
				__weakPtr = nullptr;
				__tmp->__abi_Release();
			}
		}
	public:
		struct BoolStruct
		{
			int Member;
		};

		typedef int BoolStruct::* BoolType;

		WeakReference() throw() : __weakPtr(nullptr)
		{
		}

		WeakReference(decltype(__nullptr)) throw() : __weakPtr(nullptr)
		{
		}

		WeakReference(const WeakReference& __otherArg) throw() : __weakPtr(__otherArg.__weakPtr)
		{
			InternalAddRef();
		}

		WeakReference(WeakReference&& __otherArg) throw() : __weakPtr(__otherArg.__weakPtr)
		{
			__otherArg.__weakPtr = nullptr;
		}

		explicit WeakReference(const volatile ::Platform::Object^ const __otherArg) : __weakPtr(nullptr)
		{
			__weakPtr = Details::GetWeakReference(__otherArg);
		}

		~WeakReference() throw()
		{
			InternalRelease();
		}

		WeakReference& operator=(decltype(__nullptr)) throw()
		{
			InternalRelease();
			return *this;
		}

		WeakReference& operator=(const WeakReference& __otherArg) throw()
		{
			if (&__otherArg != this)
			{
				InternalRelease();
				__weakPtr = __otherArg.__weakPtr;
				InternalAddRef();
			}
			return *this;
		}

		WeakReference& operator=(WeakReference&& __otherArg) throw()
		{
			InternalRelease();
			__weakPtr = __otherArg.__weakPtr;
			__otherArg.__weakPtr = nullptr;
			return *this;
		}

		WeakReference& operator=(const volatile ::Platform::Object^ const __otherArg)
		{
			__abi_IUnknown* __weakPtrLoc = Details::GetWeakReference(__otherArg);
			InternalRelease();
			__weakPtr = __weakPtrLoc;
			return *this;
		}

		template<typename __TArg>
		__declspec(no_refcount)
			__TArg^ Resolve()
		{
			return reinterpret_cast<__TArg^>(Details::ResolveWeakReference(__uuidof(__TArg^), &__weakPtr));
		}

		operator BoolType() const throw()
		{
			return __weakPtr != nullptr ? &BoolStruct::Member : nullptr;
		}

		friend bool operator==(const WeakReference&, const WeakReference&) throw();
		friend bool operator==(const WeakReference&, decltype(__nullptr)) throw();
		friend bool operator==(decltype(__nullptr), const WeakReference&) throw();
		friend bool operator!=(const WeakReference&, const WeakReference&) throw();
		friend bool operator!=(const WeakReference&, decltype(__nullptr)) throw();
		friend bool operator!=(decltype(__nullptr), const WeakReference&) throw();
		friend bool operator<(const WeakReference&, const WeakReference&) throw();
	};

	inline bool operator==(const WeakReference& __aArg, const WeakReference& __bArg) throw()
	{
		return __aArg.__weakPtr == __bArg.__weakPtr;
	}

	inline bool operator==(const WeakReference& __aArg, decltype(__nullptr)) throw()
	{
		return __aArg.__weakPtr == nullptr;
	}

	inline bool operator==(decltype(__nullptr), const WeakReference& __bArg) throw()
	{
		return __bArg.__weakPtr == nullptr;
	}

	inline bool operator!=(const WeakReference& __aArg, const WeakReference& __bArg) throw()
	{
		return __aArg.__weakPtr != __bArg.__weakPtr;
	}

	inline bool operator!=(const WeakReference& __aArg, decltype(__nullptr)) throw()
	{
		return __aArg.__weakPtr != nullptr;
	}

	inline bool operator!=(decltype(__nullptr), const WeakReference& __bArg) throw()
	{
		return __bArg.__weakPtr != nullptr;
	}

	inline bool operator<(const WeakReference& __aArg, const WeakReference& __bArg) throw()
	{
		return __aArg.__weakPtr < __bArg.__weakPtr;
	}
} // namespace Platform

namespace Windows { namespace Foundation
{
	inline Point::Point(float __xArg, float __yArg) : X(__xArg), Y(__yArg)
	{
	}

	inline bool Point::operator ==(Point __point1Arg, Point __point2Arg)
	{
		return __point1Arg.X == __point2Arg.X && __point1Arg.Y == __point2Arg.Y;
	}

	inline bool Point::operator !=(Point __point1Arg, Point __point2Arg)
	{
		return !(__point1Arg == __point2Arg);
	}

	// Size
	inline Size::Size(float __widthArg, float __heightArg)
	{
		if (__widthArg < 0 || __heightArg < 0)
		{
			__abi_WinRTraiseInvalidArgumentException();
		}

		Width = __widthArg;
		Height = __heightArg;
	}

	inline bool Size::IsEmpty::get() 
	{ 
		return Width < 0; 
	}

	inline bool Size::operator ==(Size __size1Arg, Size __size2Arg)
	{
		return __size1Arg.Height == __size2Arg.Height && __size1Arg.Width == __size2Arg.Width;
	}

	inline bool Size::operator !=(Size __size1Arg, Size __size2Arg)
	{
		return !(__size1Arg == __size2Arg);
	}

	inline Rect::Rect(float __xArg, float __yArg, float __widthArg, float __heightArg)
	{
		if (__widthArg < 0 || __heightArg < 0)
		{
			__abi_WinRTraiseInvalidArgumentException();
		}

		X = __xArg;
		Y = __yArg;
		Width = __widthArg;
		Height = __heightArg;
	}

	inline bool Rect::IsEmpty::get() 
	{ 
		return Width < 0; 
	}

	inline float Rect::Left::get() 
	{ 
		return X; 
	}

	inline float Rect::Top::get() 
	{ 
		return Y; 
	}

	inline bool Rect::operator ==(Rect __rect1Arg, Rect __rect2Arg)
	{
		return __rect1Arg.X == __rect2Arg.X
			&& __rect1Arg.Y == __rect2Arg.Y
			&& __rect1Arg.Width == __rect2Arg.Width
			&& __rect1Arg.Height == __rect2Arg.Height;
	}

	inline bool Rect::operator !=(Rect __rect1Arg, Rect __rect2Arg)
	{
		return !(__rect1Arg == __rect2Arg);
	}
} } // namespace Windows::Foundation

namespace Windows { namespace UI { namespace Xaml 
{
	inline Thickness::Thickness(double __uniformLengthArg)
	{
		Left = Top = Right = Bottom = __uniformLengthArg;
	}

	inline Thickness::Thickness(double __leftArg, double __topArg, double __rightArg, double __bottomArg)
	{
		Left = __leftArg;
		Top = __topArg;
		Right = __rightArg;
		Bottom = __bottomArg;
	}

	inline bool Thickness::operator==(Thickness __thickness1Arg, Thickness __thickness2Arg)
	{
		return __thickness1Arg.Left == __thickness2Arg.Left &&
			__thickness1Arg.Top == __thickness2Arg.Top &&
			__thickness1Arg.Right == __thickness2Arg.Right &&
			__thickness1Arg.Bottom == __thickness2Arg.Bottom;
	}

	inline bool Thickness::operator!=(Thickness __thickness1Arg, Thickness __thickness2Arg)
	{
		return !(__thickness1Arg == __thickness2Arg);
	}

	inline CornerRadius::CornerRadius(double __uniformRadiusArg)
	{
		TopLeft = TopRight = BottomRight = BottomLeft = __uniformRadiusArg;
	}

	inline CornerRadius::CornerRadius(double __topLeftArg, double __topRightArg, double __bottomRightArg, double __bottomLeftArg)
	{
		TopLeft = __topLeftArg;
		TopRight = __topRightArg;
		BottomRight = __bottomRightArg;
		BottomLeft = __bottomLeftArg;
	}

	inline bool CornerRadius::operator==(CornerRadius __cornerRadius1Arg, CornerRadius __cornerRadius2Arg)
	{
		return __cornerRadius1Arg.TopLeft == __cornerRadius2Arg.TopLeft &&
			__cornerRadius1Arg.TopRight == __cornerRadius2Arg.TopRight &&
			__cornerRadius1Arg.BottomRight == __cornerRadius2Arg.BottomRight &&
			__cornerRadius1Arg.BottomLeft == __cornerRadius2Arg.BottomLeft;
	}

	inline bool CornerRadius::operator!=(CornerRadius __cornerRadius1Arg, CornerRadius __cornerRadius2Arg)
	{
		return !(__cornerRadius1Arg == __cornerRadius2Arg);
	}

	namespace Media
	{
		inline Matrix Matrix::Identity::get() 
		{ 
			return Matrix(1, 0, 0, 1, 0, 0); 
		}

		inline bool Matrix::IsIdentity::get() 
		{ 
			return M11 == 1 &&
				M12 == 0 &&
				M21 == 0 &&
				M22 == 1 &&
				OffsetX == 0 &&
				OffsetY == 0; 
		}

		inline Windows::Foundation::Point Matrix::Transform(Windows::Foundation::Point __pointArg)
		{
			float x = __pointArg.X;
			float y = __pointArg.Y;
			double num = (y * M21) + OffsetX;
			double num2 = (x * M12) + OffsetY;
			x *= (float)M11;
			x += (float)num;
			y *= (float)M22;
			y += (float)num2;
			return Windows::Foundation::Point(x, y);
		}

		inline Matrix::Matrix(double __m11Arg, double __m12Arg, double __m21Arg, double __m22Arg, double __offsetXArg, double __offsetYArg)
		{
			M11 = __m11Arg;
			M12 = __m12Arg;
			M21 = __m21Arg;
			M22 = __m22Arg;
			OffsetX = __offsetXArg;
			OffsetY = __offsetYArg;
		}

		inline bool Matrix::operator==(Matrix __matrix1Arg, Matrix __matrix2Arg)
		{
			return 
				__matrix1Arg.M11 == __matrix2Arg.M11 &&
				__matrix1Arg.M12 == __matrix2Arg.M12 &&
				__matrix1Arg.M21 == __matrix2Arg.M21 &&
				__matrix1Arg.M22 == __matrix2Arg.M22 &&
				__matrix1Arg.OffsetX == __matrix2Arg.OffsetX &&
				__matrix1Arg.OffsetY == __matrix2Arg.OffsetY; 
		}

		inline bool Matrix::operator!=(Matrix __matrix1Arg, Matrix __matrix2Arg)
		{
			return !(__matrix1Arg == __matrix2Arg);
		}

		namespace Media3D
		{
			inline Matrix3D Matrix3D::Identity::get() 
			{ 
				return Matrix3D(1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1); 
			}

			inline bool Matrix3D::IsIdentity::get() 
			{ 
				return M11 == 1 && M12 == 0 && M13 == 0 && M14 == 0 &&
					M21 == 0 && M22 == 1 && M23 == 0 && M24 == 0 &&
					M31 == 0 && M32 == 0 && M33 == 1 && M34 == 0 &&
					OffsetX == 0 && OffsetY == 0 && OffsetZ == 0 && M44 == 1; 
			}

			inline Matrix3D::Matrix3D(double __m11Arg, double __m12Arg, double __m13Arg, double __m14Arg, 
				double __m21Arg, double __m22Arg, double __m23Arg, double __m24Arg, 
				double __m31Arg, double __m32Arg, double __m33Arg, double __m34Arg, 
				double __offsetXArg, double __offsetYArg, double __offsetZArg, double __m44Arg)
			{
				M11 = __m11Arg;
				M12 = __m12Arg;
				M13 = __m13Arg;
				M14 = __m14Arg;
				M21 = __m21Arg;
				M22 = __m22Arg;
				M23 = __m23Arg;
				M24 = __m24Arg;
				M31 = __m31Arg;
				M32 = __m32Arg;
				M33 = __m33Arg;
				M34 = __m34Arg;
				OffsetX = __offsetXArg;
				OffsetY = __offsetYArg;
				OffsetZ = __offsetZArg;
				M44 = __m44Arg;
			}

			inline bool Matrix3D::operator==(Matrix3D __matrix1Arg, Matrix3D __matrix2Arg)
			{
				return 
					__matrix1Arg.M11 == __matrix2Arg.M11 &&
					__matrix1Arg.M12 == __matrix2Arg.M12 &&
					__matrix1Arg.M13 == __matrix2Arg.M13 &&
					__matrix1Arg.M14 == __matrix2Arg.M14 &&
					__matrix1Arg.M21 == __matrix2Arg.M21 &&
					__matrix1Arg.M22 == __matrix2Arg.M22 &&
					__matrix1Arg.M23 == __matrix2Arg.M23 &&
					__matrix1Arg.M24 == __matrix2Arg.M24 &&
					__matrix1Arg.M31 == __matrix2Arg.M31 &&
					__matrix1Arg.M32 == __matrix2Arg.M32 &&
					__matrix1Arg.M33 == __matrix2Arg.M33 &&
					__matrix1Arg.M34 == __matrix2Arg.M34 &&
					__matrix1Arg.OffsetX == __matrix2Arg.OffsetX &&
					__matrix1Arg.OffsetY == __matrix2Arg.OffsetY &&
					__matrix1Arg.OffsetZ == __matrix2Arg.OffsetZ &&
					__matrix1Arg.M44 == __matrix2Arg.M44;
			}

			inline bool Matrix3D::operator!=(Matrix3D __matrix1Arg, Matrix3D __matrix2Arg)
			{
				return !(__matrix1Arg == __matrix2Arg);
			}
		} // Media3D

		namespace Animation
		{
			inline KeyTime::KeyTime(Windows::Foundation::TimeSpan __timeSpanArg)
			{
				if (__timeSpanArg.Duration < 0 )
				{
					__abi_WinRTraiseInvalidArgumentException();
				}
				TimeSpan = __timeSpanArg;
			}

			inline bool KeyTime::operator==(KeyTime __keyTime1Arg, KeyTime __keyTime2Arg)
			{
				return __keyTime1Arg.TimeSpan.Duration == __keyTime2Arg.TimeSpan.Duration;
			}

			inline bool KeyTime::operator!=(KeyTime __keyTime1Arg, KeyTime __keyTime2Arg)
			{
				return !(__keyTime1Arg == __keyTime2Arg);
			}

			inline RepeatBehavior::RepeatBehavior(Windows::Foundation::TimeSpan __durationArg)
			{
				if (__durationArg.Duration < 0 )
				{
					__abi_WinRTraiseInvalidArgumentException();
				}

				__duration = __durationArg;
				__count = 0.0;
				__type = RepeatBehaviorType::Duration;
			}

			inline RepeatBehavior RepeatBehavior::Forever::get()
			{
				RepeatBehavior forever;
				forever.__type = RepeatBehaviorType::Forever;

				return forever;
			}

			inline bool RepeatBehavior::HasCount::get()
			{
				return __type == RepeatBehaviorType::Count;
			}

			inline bool RepeatBehavior::HasDuration::get()
			{
				return __type == RepeatBehaviorType::Duration;
			}


			inline bool RepeatBehavior::operator ==(RepeatBehavior __repeatBehavior1Arg, RepeatBehavior __repeatBehavior2Arg)
			{
				if (__repeatBehavior1Arg.__type == __repeatBehavior2Arg.__type)
				{
					switch (__repeatBehavior1Arg.__type)
					{
					case RepeatBehaviorType::Forever:
					case RepeatBehaviorType::Count:
						return true;

					case RepeatBehaviorType::Duration:

						return __repeatBehavior1Arg.__duration.Duration == __repeatBehavior2Arg.__duration.Duration;

					default:
						return false;
					}
				}
				else
				{
					return false;
				}
			}

			inline bool RepeatBehavior::operator !=(RepeatBehavior __repeatBehavior1Arg, RepeatBehavior __repeatBehavior2Arg)
			{
				return !(__repeatBehavior1Arg == __repeatBehavior2Arg);
			}

		} // Animation
	} // Media

	inline Duration::Duration(Windows::Foundation::TimeSpan __timeSpanArg)
	{
		__durationType = DurationType::TimeSpan;
		__timeSpan = __timeSpanArg;
	}

	inline bool Duration::operator ==(Duration __t1Arg, Duration __t2Arg)
	{
		if (__t1Arg.HasTimeSpan)
		{
			if (__t2Arg.HasTimeSpan)
			{
				return __t1Arg.__timeSpan.Duration == __t2Arg.__timeSpan.Duration;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return __t1Arg.__durationType == __t2Arg.__durationType;
		}
	}

	inline bool Duration::operator !=(Duration __t1Arg, Duration __t2Arg)
	{
		return !(__t1Arg == __t2Arg);
	}

	inline bool Duration::HasTimeSpan::get()
	{
		return (__durationType == DurationType::TimeSpan);
	}

	inline Duration Duration::Automatic::get()
	{
		Duration __duration;
		__duration.__durationType = DurationType::Automatic;

		return __duration;
	}

	inline Duration Duration::Forever::get()
	{
		Duration __duration;
		__duration.__durationType = DurationType::Forever;

		return __duration;
	}

	inline Windows::Foundation::TimeSpan Duration::TimeSpan::get()
	{
		if (HasTimeSpan)
		{
			return __timeSpan;
		}
		else
		{
			Windows::Foundation::TimeSpan __timeSpanLoc;
			__timeSpanLoc.Duration = 0;
			return __timeSpanLoc;
		}
	}

	inline GridLength::GridLength(double __pixelsArg)
	{
		*this = GridLength(__pixelsArg, Windows::UI::Xaml::GridUnitType::Pixel);
	}

	inline double GridLength::Value::get() 
	{ 
		return (__unitType == Windows::UI::Xaml::GridUnitType::Auto ) ? GridLength(1.0, Windows::UI::Xaml::GridUnitType::Auto).__unitValue : __unitValue; 
	}

	inline Windows::UI::Xaml::GridUnitType GridLength::GridUnitType::get()
	{ 
		return (__unitType); 
	} 

	inline bool GridLength::IsAbsolute::get()
	{ 
		return (__unitType == Windows::UI::Xaml::GridUnitType::Pixel); 
	} 

	inline bool GridLength::IsAuto::get() 
	{ 
		return (__unitType == Windows::UI::Xaml::GridUnitType::Auto); 
	} 

	inline bool GridLength::IsStar::get() 
	{ 
		return (__unitType == Windows::UI::Xaml::GridUnitType::Star) ; 
	} 

	inline GridLength GridLength::Auto::get()
	{ 
		return ( GridLength(1.0, Windows::UI::Xaml::GridUnitType::Auto)); 
	}

	inline bool GridLength::operator ==(GridLength __gridLength1Arg, GridLength __gridLength2Arg)
	{
		if (__gridLength1Arg.GridUnitType == __gridLength2Arg.GridUnitType)
		{
			if (__gridLength1Arg.IsAuto || __gridLength1Arg.__unitType == __gridLength2Arg.__unitType)
			{
				return true;
			}
		}
		return false;
	}

	inline bool GridLength::operator !=(GridLength __gridLength1Arg, GridLength __gridLength2Arg)
	{
		return !(__gridLength1Arg == __gridLength2Arg);
	}
} } } // Windows::UI::Xaml

// Don't pull in any symbols if it's vccorlib compiled
#ifndef VCWINRT_DLL
#ifdef _WINRT_DLL
// DLL
#ifdef _M_IX86
#pragma comment(linker, "/EXPORT:DllGetActivationFactory=_DllGetActivationFactory@8,PRIVATE")
#pragma comment(linker, "/EXPORT:DllCanUnloadNow=_DllCanUnloadNow@0,PRIVATE")
#else
#pragma comment(linker, "/EXPORT:DllGetActivationFactory=DllGetActivationFactory,PRIVATE")
#pragma comment(linker, "/EXPORT:DllCanUnloadNow,PRIVATE")
#endif
#endif //endif _WINRT_DLL
#endif //endif VCWINRT_DLL

#if !defined(VCWINRT_DLL)
#if defined(_DEBUG)
#pragma comment(lib, "vccorlibd.lib")
#else
#pragma comment(lib, "vccorlib.lib")
#endif  // DEBUG
#endif  // VCWINRT_DLL

#pragma comment(lib, "runtimeobject.lib")
#ifndef _CORESYS
#pragma comment(lib, "ole32.lib")
#endif

#pragma pack(pop)

#pragma initialize_winrt_types_done

// Restore warnings disabled for this file to their original settings
#pragma warning( pop )

#if defined(__VCCORLIB_H_ENABLE_ALL_WARNINGS)
#pragma warning(pop)
#endif

#endif // _VCCORLIB_H_
