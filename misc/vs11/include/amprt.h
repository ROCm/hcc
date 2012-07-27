/***
* ==++==
*
* Copyright (c) Microsoft Corporation.  All rights reserved.
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* amprt.h
*
* Define the C++ interfaces exported by the C++ AMP runtime
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/
#pragma once

#if !(defined (_M_X64) || defined (_M_IX86) || defined (_M_ARM))
    #error ERROR: C++ AMP runtime is supported only on X64, X86 and ARM architectures.
#endif  

#if defined (_M_CEE)
    #error ERROR: C++ AMP runtime is not supported when compiling /clr.
#endif  

#ifndef __cplusplus
    #error ERROR: C++ AMP runtime is supported only for C++.
#endif  

#if !defined(_CXXAMP)

#if defined(_DEBUG)
    #pragma comment(lib, "vcampd")
#else   // _DEBUG
    #pragma comment(lib, "vcamp")
#endif  // _DEBUG

#endif // _CXXAMP

#if !defined(_CXXAMP)

#define __GPU      restrict(amp,cpu)
#define __GPU_ONLY restrict(amp)
#define __CPU_ONLY

#else

#define __GPU
#define __GPU_ONLY
#define __CPU_ONLY

#endif // _CXXAMP

#include <exception>
#include <unknwn.h>
#include <crtdbg.h>
#include <string>
#include <vector>

#if defined(_CXXAMP)
#include <strsafe.h>
#endif // _CXXAMP

#include <future>
#include <functional>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <concrt.h>
#include <type_traits>
#include <evntprov.h>
#include <evntrace.h>

#if !defined(_AMPIMP)
#define _AMPIMP     __declspec(dllimport)
#endif

#pragma pack(push,8)

// Part of runtime-compiler interface
extern "C" 
{
    // Access mode of fields
    enum _Access_mode
    {
        _No_access = 0,
        _Read_access = (1 << 0),
        _Write_access = (1 << 1),
        _Is_array_mode = (1 << 30),
        _Read_write_access = _Read_access | _Write_access,
    };
}

namespace Concurrency
{

// Forward declarations
class accelerator_view;
class accelerator;

namespace details
{
    const size_t ERROR_MSG_BUFFER_SIZE = 1024;

    //  A reference counter to be used as the base class for all reference counted types.
    class _Reference_counter
    {
    public:

        //  Constructor.
        _Reference_counter()  : _M_rc(0) {}

        //  Destructor.
        virtual ~_Reference_counter() {}

        // Add a reference. 
        // Thread-safe.
        size_t _Add_reference()
        {
            return InterlockedIncrement(reinterpret_cast<LONG volatile*>(&_M_rc));
        }

        // Remove a reference. 
        // Thread-safe.
        size_t _Remove_reference()
        {
            _ASSERTE(_M_rc > 0);

            size_t refCount = InterlockedDecrement(reinterpret_cast<LONG volatile*>(&_M_rc));

            if (refCount == 0)
                this->_Release();

            return refCount;
        }

        // Release the counter
        virtual void _Release()
        {
            // We need to swallow any exceptions here since this method would be called
            // when reference counted buffer ptrs go out of scope and since it may happen 
            // when unwindind the stack while handling an exception, another exception
            // will cause the program to terminate
            try {
                delete this;
            }
            catch(...) {}
        }

        // Return the reference count value
        size_t _Get_reference_count()
        {
            return _M_rc;
        }

    private:
        size_t _M_rc;
    };

    // A smart pointer to a reference counted object
    // T must be a type derived from _Reference_counter
    template <class T>
    class _Reference_counted_obj_ptr
    {
    public:
    
        // Constructor
        _Reference_counted_obj_ptr(T* _Ptr = NULL) :  _M_obj_ptr(_Ptr)
        {
            _Init();
        }

        // Copy constructor
        _Reference_counted_obj_ptr(const _Reference_counted_obj_ptr &_Other) : _M_obj_ptr(_Other._M_obj_ptr)
        {
            _Init();
        }

        // Destructor
        ~_Reference_counted_obj_ptr()
        {
            if (_M_obj_ptr != NULL) {
                _UnInitialize(_M_obj_ptr);
            }
        }

        // Assignment operator
        _Reference_counted_obj_ptr& operator=(const _Reference_counted_obj_ptr &_Other)
        {
            if (_M_obj_ptr != _Other._M_obj_ptr)
            {
                T *oldPtr = _M_obj_ptr;
                _M_obj_ptr = _Other._M_obj_ptr;
                _Init();
        
                if (oldPtr != NULL) {
                    _UnInitialize(oldPtr);
                }
            }

            return *this;
        }

        _Ret_ T* operator->() const
        {
            return _M_obj_ptr;
        }

        T& operator*() const
        {
            return *_M_obj_ptr;
        }

        operator T*() const
        {
            return _M_obj_ptr;
        }

        _Ret_ T* _Get_ptr() const
        {
            return _M_obj_ptr;
        }

    private:
        T *_M_obj_ptr;

        void _Init()
        {
            if (_M_obj_ptr == NULL)
                return;

            reinterpret_cast<_Reference_counter*>(_M_obj_ptr)->_Add_reference();
        }

        static void _UnInitialize(_In_ T *_Obj_ptr)
        {
            reinterpret_cast<_Reference_counter*>(_Obj_ptr)->_Remove_reference();
        }
    };

    // Forward declarations
    class _Amp_runtime_trace;
    class _Buffer;
    class _Texture;
    class _Ubiquitous_buffer;
    class _D3D_interop;
    class _Accelerator_view_impl;
    class _CPU_accelerator_view_impl;
    class _D3D_accelerator_view_impl;
    class _Accelerator_impl;
    class _Event_impl;
    class _DPC_runtime_factory;
    class _View_shape;
    struct _Buffer_descriptor;
    class _Accelerator_view_hasher;

    // The enum specifies the base type for short vector type.
    enum _Short_vector_base_type_id : unsigned int
    {
        _Uint_type = 0,
        _Int_type = 1,
        _Float_type = 2,
        _Unorm_type = 3,
        _Norm_type = 4,
        _Double_type = 5,
        _Invalid_type = 0xFFFFFFFF
    };

    typedef enum _Short_vector_base_type_id _Texture_base_type_id;

} // namespace Concurrency::details

typedef details::_Reference_counted_obj_ptr<details::_Accelerator_view_impl> _Accelerator_view_impl_ptr;
typedef details::_Reference_counted_obj_ptr<details::_Accelerator_impl> _Accelerator_impl_ptr;
typedef details::_Reference_counted_obj_ptr<details::_Buffer> _Buffer_ptr;
typedef details::_Reference_counted_obj_ptr<details::_Texture> _Texture_ptr;
typedef details::_Reference_counted_obj_ptr<details::_Ubiquitous_buffer> _Ubiquitous_buffer_ptr;
typedef details::_Reference_counted_obj_ptr<details::_Event_impl> _Event_impl_ptr;
typedef details::_Reference_counted_obj_ptr<details::_View_shape> _View_shape_ptr;

namespace details
{
    // The _Event class.
    class _Event
    {
        friend class _Buffer;
        friend class _Texture;
        friend class accelerator_view;
        friend class _D3D_accelerator_view_impl;

    public:
        /// <summary>
        ///     Constructor of the _Event.
        /// </summary>
        _AMPIMP _Event();

        /// <summary>
        ///     Destructor of the _Event.
        /// </summary>
        _AMPIMP ~_Event();

        /// <summary>
        ///     Copy constructor
        /// </summary>
        _AMPIMP _Event(const _Event & _Other);

        /// <summary>
        ///     Assignment operator
        /// </summary>
        _AMPIMP _Event & operator=(const _Event & _Other);

        /// <summary>
        ///     Poll whether the _Event has completed or not. Swallows any exceptions
        /// </summary>
        /// <returns>
        ///     true, if the _Event has completed, false otherwise
        /// </returns>
        _AMPIMP bool _Is_finished_nothrow(); 

        /// <summary>
        ///     Poll whether the _Event has completed or not and throws any exceptions that occur
        /// </summary>
        /// <returns>
        ///     true, if the _Event has completed, false otherwise
        /// </returns>
        _AMPIMP bool _Is_finished(); 

        /// <summary>
        ///     Wait until the _Event completes and throw any exceptions that occur.
        /// </summary>
        _AMPIMP void _Get();

        /// <summary>
        ///     Tells if this is an empty event
        /// </summary>
        /// <returns>
        ///     true, if the _Event is empty
        ///     false, otherwise
        /// </returns>
        _AMPIMP bool _Is_empty() const;

        /// <summary>
        ///     Creates an event which is an ordered collection of this and _Ev
        /// </summary>
        /// <returns>
        ///     The composite event
        /// </returns>
        _AMPIMP _Event _Add_event(_Event _Ev);

        /// <summary>
        ///     Creates an event which is an ordered collection of this and a continuation task
        /// </summary>
        /// <returns>
        ///     The composite event
        /// </returns>
        _AMPIMP _Event _Add_continuation(const std::function<_Event __cdecl ()> &_Continuation_task);

        /// <summary>
        ///     Return true if the other _Event is same as this _Event; false otherwise
        /// </summary>
        _AMPIMP bool operator==(const _Event &_Other) const;

        /// <summary>
        ///     Return false if the other _Event is same as this _Event; true otherwise
        /// </summary>
        _AMPIMP bool operator!=(const _Event &_Other) const;

    private:

        // Private constructor
        _Event(_Event_impl_ptr _Impl);

        _Event_impl_ptr _M_ptr_event_impl;
    };

    typedef _Buffer_descriptor *_View_key;

    _Accelerator_view_impl_ptr _Get_accelerator_view_impl_ptr(const accelerator_view& _Accl_view);
    _Accelerator_impl_ptr _Get_accelerator_impl_ptr(const accelerator& _Accl);
    _Event _Get_access_async(const _View_key _Key, accelerator_view _Av, _Access_mode _Mode, _Buffer_ptr &_Buf_ptr);

    inline bool _Is_valid_access_mode(_Access_mode _Mode)
    {
        if ((_Mode != _Read_access) && 
            (_Mode != _Write_access) &&
            (_Mode != _Read_write_access))
        {
            return false;
        }

        return true;
    }

    // Caution: Do not change this structure defintion.
    // This struct is special and is processed by the FE to identify the buffers
    // used in a parallel_for_each and to setup the _M_data_ptr with the appropriate
    // buffer ptr value in the device code.
    typedef struct _Buffer_descriptor
    {
        friend _Event _Get_access_async(const _View_key _Key, accelerator_view _Av, _Access_mode _Mode, _Buffer_ptr &_Buf_ptr);

        // _M_data_ptr points to the raw data underlying the buffer for accessing on host
        mutable void *_M_data_ptr;

    private:
        // _M_buffer_ptr points to a _Ubiquitous_buffer that holds the data in an 1D array.
        // This is private to ensure that all assignments to this data member
        // only happen through public functions which properly manage the
        // ref count of the underlying buffer
        _Ubiquitous_buffer *_M_buffer_ptr;

    public:
        // _M_curr_cpu_access_mode specifies the current access mode of the data on the
        // cpu accelerator_view specified at the time of registration of this view
        _Access_mode _M_curr_cpu_access_mode;

        // _M_type_acess_mode specifies the access mode of the overlay type
        // array_views set it to the appropriate access mode and for arrays it is
        // always _Is_array_mode.
        _Access_mode _M_type_access_mode;

    public:
        // Public functions

        // Default constructor
        _Buffer_descriptor() __GPU
            : _M_data_ptr(NULL), _M_buffer_ptr(NULL),
            _M_curr_cpu_access_mode(_No_access), _M_type_access_mode(_Is_array_mode)
        {
        }

        _Buffer_descriptor(_In_ void *_Data_ptr, _In_ _Ubiquitous_buffer *_Buffer_ptr,
                           _Access_mode _Curr_cpu_access_mode, _Access_mode _Type_mode) __GPU
            : _M_data_ptr(_Data_ptr), _M_buffer_ptr(NULL),
            _M_curr_cpu_access_mode(_Curr_cpu_access_mode), _M_type_access_mode(_Type_mode)
        {
            _Set_buffer_ptr(_Buffer_ptr);
        }

        // Destructor
        ~_Buffer_descriptor() __GPU
        {
            _Set_buffer_ptr(NULL);
        }

        // Copy constructor
        _Buffer_descriptor(const _Buffer_descriptor &_Other) __GPU
            : _M_data_ptr(_Other._M_data_ptr), _M_buffer_ptr(NULL),
            _M_curr_cpu_access_mode(_Other._M_curr_cpu_access_mode), _M_type_access_mode(_Other._M_type_access_mode)
        {
            _Set_buffer_ptr(_Other._M_buffer_ptr);
        }

        // Assignment operator
        _Buffer_descriptor& operator=(const _Buffer_descriptor &_Other) __GPU
        {
            if (this != &_Other)
            {
                _M_data_ptr = _Other._M_data_ptr;
                _M_curr_cpu_access_mode = _Other._M_curr_cpu_access_mode;
                _M_type_access_mode = _Other._M_type_access_mode;
                _Set_buffer_ptr(_Other._M_buffer_ptr);
            }

            return *this;
        }

        _Ubiquitous_buffer_ptr _Get_buffer_ptr() const __CPU_ONLY
        {
            return _M_buffer_ptr;
        }

        void _Set_buffer_ptr(_In_opt_ _Ubiquitous_buffer *_Buffer_ptr) __CPU_ONLY
        {
            if (_M_buffer_ptr != _Buffer_ptr)
            {
                if (_M_buffer_ptr != NULL) {
                    reinterpret_cast<_Reference_counter*>(_M_buffer_ptr)->_Remove_reference();
                }

                _M_buffer_ptr = _Buffer_ptr;

                if (_M_buffer_ptr != NULL) {
                    reinterpret_cast<_Reference_counter*>(_M_buffer_ptr)->_Add_reference();
                }
            }
        }

#if !defined(_CXXAMP)
        void _Set_buffer_ptr(_In_opt_ _Ubiquitous_buffer *_Buffer_ptr) __GPU_ONLY
        {
            // No need to set the buffer ptr on the GPU
            UNREFERENCED_PARAMETER(_Buffer_ptr);
            _M_buffer_ptr = NULL;
        }
#endif // _CXXAMP

        bool _Is_array() const
        {
            return (_M_type_access_mode == _Is_array_mode);
        }

        _Ret_ _View_key _Get_view_key()
        {
            return this;
        }

        const _View_key _Get_view_key() const
        {
            return ((const _View_key)(this));
        }

        _AMPIMP void _Get_CPU_access(_Access_mode _Requested_mode) const;

    } _Buffer_descriptor;

    // Caution: Do not change this structure defintion.
    // This struct is special and is processed by the FE to identify the textures
    // used in a parallel_for_each and to setup the _M_data_ptr with the appropriate
    // texture ptr value in the device code.
    typedef struct _Texture_descriptor
    {
        // _M_data_ptr points to the raw data underlying the texture
        mutable IUnknown *_M_data_ptr;

    private:
        // _M_texture_ptr points to a _Texture that holds the data
        // This is private to ensure that all assignments to this data member
        // only happen through public functions which properly manage the
        // ref count of the underlying texture
        _Texture *_M_texture_ptr;

    public:
        // Public functions

        // Default constructor
        _Texture_descriptor() __GPU
            : _M_data_ptr(NULL), _M_texture_ptr(NULL)
        {
        }

        _Texture_descriptor(_In_ _Texture * _Texture_ptr) __GPU
            : _M_data_ptr(NULL), _M_texture_ptr(NULL)
        {
            _Set_texture_ptr(_Texture_ptr);
        }

        _Texture_descriptor(_In_ IUnknown *_Data_ptr, _In_ _Texture * _Texture_ptr) __CPU_ONLY
            : _M_data_ptr(_Data_ptr), _M_texture_ptr(NULL)
        {
            _Set_texture_ptr(_Texture_ptr);
        }

        // Destructor
        ~_Texture_descriptor() __GPU
        {
            _Set_texture_ptr(NULL);
        }

        // Copy constructor
        _Texture_descriptor(const _Texture_descriptor &_Other) __GPU
            : _M_data_ptr(_Other._M_data_ptr), _M_texture_ptr(NULL)
        {
            _Set_texture_ptr(_Other._M_texture_ptr);
        }

        // Assignment operator
        _Texture_descriptor& operator=(const _Texture_descriptor &_Other) __GPU
        {
            if (this != &_Other)
            {
                _M_data_ptr = _Other._M_data_ptr;
                _Set_texture_ptr(_Other._M_texture_ptr);
            }

            return *this;
        }

        // Move constructor
        _Texture_descriptor(_Texture_descriptor &&_Other) __CPU_ONLY
        {
            *this = std::move(_Other);
        }

        bool operator==(const _Texture_descriptor &_Other) const __GPU
        {
            return _M_texture_ptr == _Other._M_texture_ptr && _M_data_ptr == _Other._M_data_ptr;
        }

        _Texture_ptr _Get_texture_ptr() const __CPU_ONLY
        {
            return _M_texture_ptr;
        }

        void _Set_texture_ptr(_In_opt_ _Texture *_Texture_ptr) __CPU_ONLY
        {
            if (_M_texture_ptr != _Texture_ptr)
            {
                if (_M_texture_ptr != NULL) {
                    reinterpret_cast<_Reference_counter*>(_M_texture_ptr)->_Remove_reference();
                }

                _M_texture_ptr = _Texture_ptr;

                if (_M_texture_ptr != NULL) {
                    reinterpret_cast<_Reference_counter*>(_M_texture_ptr)->_Add_reference();
                }
            }
        }

#if !defined(_CXXAMP)
        void _Set_texture_ptr(_In_opt_ _Texture *_Texture_ptr) __GPU_ONLY
        {
            // No need to set the texture ptr on the GPU
            UNREFERENCED_PARAMETER(_Texture_ptr);
            _M_texture_ptr = NULL;
        }
#endif // _CXXAMP

    } _Texture_descriptor;


} // namespace Concurrency::details

// Forward declaration
class accelerator;

namespace details
{
    _AMPIMP size_t __cdecl _Get_num_devices();
    _AMPIMP _Ret_ _Accelerator_impl_ptr * __cdecl _Get_devices();
    _AMPIMP accelerator __cdecl _Select_default_accelerator();
    _AMPIMP bool __cdecl _Set_default_accelerator(_Accelerator_impl_ptr _Accl);
    _AMPIMP bool __cdecl _Is_D3D_accelerator_view(const accelerator_view& _Av);
    _AMPIMP void __cdecl _Register_async_event(const _Event &_Ev, const std::shared_future<void> &_Shared_future);
}

/// <summary>
///    Queuing modes supported for accelerator views
/// </summary>
enum queuing_mode {
    queuing_mode_immediate,
    queuing_mode_automatic
}; 

/// <summary>
///     Exception thrown due to a C++ AMP runtime_exception.
///     This is the base type for all C++ AMP exception types.
/// </summary>
class runtime_exception : public std::exception
{
public:
    /// <summary>
    ///     Construct a runtime_exception exception with a message and an error code
    /// </summary>
    /// <param name="_Message">
    ///     Descriptive message of error
    /// </param>
    /// <param name="_Hresult">
    ///     HRESULT of error that caused this exception
    /// </param>
    _AMPIMP runtime_exception(const char * _Message, HRESULT _Hresult) throw();

    /// <summary>
    ///     Construct a runtime_exception exception with an error code
    /// </summary>
    /// <param name="_Hresult">
    ///     HRESULT of error that caused this exception
    /// </param>
    _AMPIMP explicit runtime_exception(HRESULT _Hresult) throw();

    /// <summary>
    ///     Copy construct a runtime_exception exception
    /// </summary>
    /// <param name="_Other">
    ///     The runtime_exception object to be copied from
    /// </param>
    _AMPIMP runtime_exception(const runtime_exception &_Other) throw();

    /// <summary>
    ///     Assignment operator
    /// </summary>
    /// <param name="_Other">
    ///     The runtime_exception object to be assigned from
    /// </param>
    _AMPIMP runtime_exception &operator=(const runtime_exception &_Other) throw();

    /// <summary>
    ///     Destruct a runtime_exception exception object instance
    /// </summary>
    _AMPIMP virtual ~runtime_exception() throw();

    /// <summary>
    ///     Get the error code that caused this exception
    /// </summary>
    /// <returns>
    ///     HRESULT of error that caused the exception
    /// </returns>
    _AMPIMP HRESULT get_error_code() const throw();

private:
    HRESULT _M_error_code;
}; // class runtime_exception

/// <summary>
///     Exception thrown when an underlying OS/DirectX call fails
///     due to lack of system or device memory 
/// </summary>
class out_of_memory : public runtime_exception
{
public:
    /// <summary>
    ///     Construct an out_of_memory exception with a message
    /// </summary>
    /// <param name="_Message">
    ///     Descriptive message of error
    /// </param>
    _AMPIMP explicit out_of_memory(const char * _Message) throw();

    /// <summary>
    ///     Construct an out_of_memory exception
    /// </summary>
    _AMPIMP out_of_memory () throw();
}; // class out_of_memory

/// <summary>
///  Class represents a accelerator abstraction for C++ AMP data-parallel devices 
/// </summary>
class accelerator
{
    friend class accelerator_view;

    friend _AMPIMP accelerator details::_Select_default_accelerator();

    friend _Accelerator_impl_ptr details::_Get_accelerator_impl_ptr(const accelerator& _Accl);

public:

    /// <summary>
    ///     String constant for default accelerator
    /// </summary>
    _AMPIMP static const wchar_t default_accelerator[];

    /// <summary>
    ///     String constant for cpu accelerator
    /// </summary>
    _AMPIMP static const wchar_t cpu_accelerator[];

	/// <summary>
    ///     String constant for direct3d WARP accelerator
    /// </summary>
    _AMPIMP static const wchar_t direct3d_warp[];

    /// <summary>
    ///     String constant for direct3d reference accelerator
    /// </summary>
    _AMPIMP static const wchar_t direct3d_ref[];

    /// <summary>
    ///     Construct a accelerator representing the default accelerator
    /// </summary>
    _AMPIMP accelerator();

    /// <summary>
    ///     Construct a accelerator representing the accelerator with the 
    ///     specified device instance path
    /// </summary>
    explicit accelerator(const std::wstring &_Device_path) : _M_impl(NULL)
    {
        _Init(_Device_path.c_str());
    }

    /// <summary>
    ///     Destructor
    /// </summary>
    _AMPIMP ~accelerator();

    /// <summary>
    ///     Copy constructor
    /// </summary>
    _AMPIMP accelerator(const accelerator &_Other);

    /// <summary>
    ///     Assignment operator
    /// </summary>
    _AMPIMP accelerator &operator=(const accelerator &_Other);

    /// <summary>
    ///     Returns the vector of accelerator objects representing all available accelerators
    /// </summary>
    /// <returns>
    ///     The vector of available accelerators
    /// </returns> 
    static inline std::vector<accelerator> get_all()
    {
        std::vector<accelerator> _AcceleratorVector;
        size_t _NumDevices = details::_Get_num_devices();
        for (size_t _I = 0; (_I < _NumDevices); ++_I)
        {
            _AcceleratorVector.push_back(details::_Get_devices()[_I]);
        }

        return _AcceleratorVector;
    }

    /// <summary>
    ///     Sets the default accelerator to be used for any operation
    ///     that implicitly uses the default accelerator. This method 
    ///     only succeeds if the runtime selected default accelerator
    ///     has not already been used in an operation that implicitly
    ///     uses the default accelerator
    /// </summary>
    /// <returns>
    ///     A boolean value indicating if the call succeeds in setting 
    ///     the default accelerator
    /// </returns> 
    static inline bool set_default(const std::wstring& _Path)
    {
        accelerator _Accl(_Path);
        return details::_Set_default_accelerator(_Accl._M_impl);
    }

    /// <summary>
    ///     Returns the system-wide unique device instance path as a std::wstring
    /// </summary>
    std::wstring get_device_path() const 
    {
        return _Get_device_path();
    }

    __declspec(property(get=get_device_path)) std::wstring device_path;

    /// <summary>
    ///     Get the version for this accelerator
    /// </summary>
    _AMPIMP unsigned int get_version() const;
    __declspec(property(get=get_version)) unsigned int version; // hiword=major, loword=minor

    /// <summary>
    ///     Returns the device description as a std::wstring
    /// </summary>
    std::wstring get_description() const
    {
        return _Get_description();
    }

    __declspec(property(get=get_description)) std::wstring description;

    /// <summary>
    ///     Returns a boolean value indicating whether the accelerator
    ///     was created with DEBUG layer enabled for extensive error reporting
    /// </summary>
    _AMPIMP bool get_is_debug() const;
    __declspec(property(get=get_is_debug)) bool is_debug;

    /// <summary>
    ///     Returns a boolean value indicating whether the accelerator is emulated. 
    ///     This is true, for example, with the direct3d reference and WARP accelerators.
    /// </summary>
    _AMPIMP bool get_is_emulated() const;
    __declspec(property(get=get_is_emulated)) bool is_emulated;

    /// <summary>
    ///     Returns a boolean value indicating whether the accelerator
    ///     is attached to a display
    /// </summary>
    _AMPIMP bool get_has_display() const;
    __declspec(property(get=get_has_display)) bool has_display;

    /// <summary>
    ///     Returns a boolean value indicating whether the accelerator
    ///     supports full double precision (including double division,
    ///     precise_math functions, int to double, double to int conversions)
    ///     in a parallel_for_each kernel.
    /// </summary>
    _AMPIMP bool get_supports_double_precision() const;
    __declspec(property(get=get_supports_double_precision)) bool supports_double_precision;

    /// <summary>
    ///     Returns a boolean value indicating whether the accelerator
    ///     has limited double precision support (excludes double division,
    ///     precise_math functions, int to double, double to int conversions)
    ///     for a parallel_for_each kernel.
    /// </summary>
    _AMPIMP bool get_supports_limited_double_precision() const;
    __declspec(property(get=get_supports_limited_double_precision)) bool supports_limited_double_precision;

    /// <summary>
    ///     Return the default accelerator view associated with this accelerator
    /// </summary>
    _AMPIMP accelerator_view get_default_view() const;
    __declspec(property(get=get_default_view)) accelerator_view default_view;

    /// <summary>
    ///     Get the dedicated memory for this accelerator in KB
    /// </summary>
    _AMPIMP size_t get_dedicated_memory() const;
    __declspec(property(get=get_dedicated_memory)) size_t dedicated_memory;

    /// <summary>
    ///     Create and return a new accelerator view on this accelerator
    ///     with the specified queuing mode. When unspecified the accelerator_view 
    ///     is created with queuing_mode_automatic queuing mode.
    /// </summary>
    _AMPIMP accelerator_view create_view(queuing_mode qmode = queuing_mode_automatic);

    /// <summary>
    ///     Return true if the other accelerator is same as this accelerator; false otherwise
    /// </summary>
    _AMPIMP bool operator==(const accelerator &_Other) const;

    /// <summary>
    ///     Return false if the other accelerator is same as this accelerator; true otherwise
    /// </summary>
    _AMPIMP bool operator!=(const accelerator &_Other) const;

private:

    // Private constructor
    _AMPIMP accelerator(_Accelerator_impl_ptr _Impl);

    // Private helper methods
    _AMPIMP const wchar_t *_Get_device_path() const;
    _AMPIMP const wchar_t *_Get_description() const;

    _AMPIMP void _Init(const wchar_t *_Path);

private:

    _Accelerator_impl_ptr _M_impl;
};

namespace direct3d
{
    /// <summary>
    ///     Get the D3D device interface underlying a accelerator_view.
    /// </summary>
    /// <param name="_Av">
    ///     The D3D accelerator_view for which the underlying D3D device interface is returned.
    /// </param>
    /// <returns>
    ///     The IUnknown interface pointer of the D3D device underlying the accelerator_view.
    /// </returns>
    _AMPIMP _Ret_ IUnknown * __cdecl get_device(const accelerator_view &_Av);

    /// <summary>
    ///     Create a accelerator_view from a D3D device interface pointer.
    /// </summary>
    /// <param name="_D3D_device">
    ///     The D3D device interface pointer to create the accelerator_view from.
    /// </param>
    /// <returns>
    ///     The accelerator_view created from the passed D3D device interface.
    /// </returns>
    _AMPIMP accelerator_view __cdecl create_accelerator_view(_In_ IUnknown *_D3D_device, queuing_mode qmode = queuing_mode_automatic);

} // namespace direct3d

/// <summary>
///  Class represents a future corresponding to a C++ AMP asynchronous operation
/// </summary>
class completion_future
{
    friend class details::_Amp_runtime_trace;
public:

    /// <summary>
    ///     Default constructor
    /// </summary>
    completion_future()
    {
    }

    /// <summary>
    ///     Copy constructor
    /// </summary>
    completion_future(const completion_future& _Other)
        : _M_shared_future(_Other._M_shared_future),
        _M_task(_Other._M_task)
    {
    }

    /// <summary>
    ///     Move constructor
    /// </summary>
    completion_future(completion_future&& _Other)
        : _M_shared_future(std::move(_Other._M_shared_future)),
        _M_task(std::move(_Other._M_task))
    {
    }

    /// <summary>
    ///     Destructor
    /// </summary>
    ~completion_future()
    {
    }

    /// <summary>
    ///     Copy assignment operator
    /// </summary>
    completion_future& operator=(const completion_future& _Other)
    {
        if (this != &_Other) {
            _M_shared_future = _Other._M_shared_future;
            _M_task = _Other._M_task;
        }

        return (*this);
    }

    /// <summary>
    ///     Move assignment operator
    /// </summary>
    completion_future& operator=(completion_future&& _Other)
    {
        if (this != &_Other) {
            _M_shared_future = std::move(_Other._M_shared_future);
            _M_task = std::move(_Other._M_task);
        }

        return (*this);
    }
    
    /// <summary>
    ///     Waits until the associated asynchronous operation completes
    ///     Throws the stored exception if one was encountered during the 
    ///     asynchronous operation
    /// </summary>
    void get() const
    {
        _M_shared_future.get();
    }
    
    /// <summary>
    ///     Returns true if the object is associated with an asynchronous
    ///     operation
    /// </summary>
    /// <returns>
    ///     true if the object is associated with an asynchronous operation
    ///     and false otherwise
    /// </returns>
    bool valid() const
    {
        return _M_shared_future.valid();
    }
    
    /// <summary>
    ///     Blocks until the associated asynchronous operation completes
    /// </summary>
    void wait() const
    {
        _M_shared_future.wait();
    }

    /// <summary>
    ///     Blocks until the associated asynchronous operation completes or
    ///     _Rel_time has elapsed
    /// </summary>
    /// <returns>
    ///     - future_status::deferred if the associated asynchronous operation is not running
    ///     - future_status::ready if the associated asynchronous operation is finished
    ///     - future_status::timeout if the time period specified has elapsed
    /// </returns>
    template <class _Rep, class _Period>
    std::future_status::future_status wait_for(const std::chrono::duration<_Rep, _Period>& _Rel_time) const
    {
        return _M_shared_future.wait_for(_Rel_time);
    }

    /// <summary>
    ///     Blocks until the associated asynchronous operation completes or
    ///     until the current time exceeds _Abs_time
    /// </summary>
    /// <returns>
    ///     - future_status::deferred if the associated asynchronous operation is not running
    ///     - future_status::ready if the associated asynchronous operation is finished
    ///     - future_status::timeout if the time point specified has been reached
    /// </returns>
    template <class _Clock, class _Duration>
    std::future_status::future_status wait_until(const std::chrono::time_point<_Clock, _Duration>& _Abs_time) const
    {
        return _M_shared_future.wait_until(_Abs_time);
    }

    /// <summary>
    ///     Returns a std::shared_future&lt;void&gt; object corresponding to the 
    ///     associated asynchronous operation
    /// </summary>
    /// <returns>
    ///     A std::shared_future&lt;void&gt; object corresponding to the associated
    ///     asynchronous operation
    /// </returns>
    operator std::shared_future<void>() const
    {
        return _M_shared_future;
    }
    
    /// <summary>
    ///     Chains a callback Functor to the completion_future to be executed
    ///     when the associated asynchronous operation finishes execution
    /// </summary>
    template <typename _Functor>
    void then(const _Functor &_Func) const
    {
        this->to_task().then(_Func);
    }

    /// <summary>
    ///     Returns a concurrency::task&lt;void&gt; object corresponding to the 
    ///     associated asynchronous operation
    /// </summary>
    /// <returns>
    ///     A concurrency::task&lt;void&gt; object corresponding to the associated
    ///     asynchronous operation
    /// </returns>
    concurrency::task<void> to_task() const
    {
        return _M_task;
    }

private:

    // Private constructor
    completion_future(const std::shared_future<void> &_Shared_future,
                      const concurrency::task<void>& _Task)
                      : _M_shared_future(_Shared_future), _M_task(_Task)
    {
    }

    std::shared_future<void> _M_shared_future;
    concurrency::task<void> _M_task;
}; 

/// <summary>
///  Class represents a virtual device abstraction on a C++ AMP data-parallel accelerator
/// </summary>
class accelerator_view
{
    friend class accelerator;
    friend class details::_Buffer;
    friend class details::_Texture;
    friend class details::_Ubiquitous_buffer;
    friend class details::_D3D_interop;
    friend class details::_D3D_accelerator_view_impl;
    friend class details::_CPU_accelerator_view_impl;
    friend class details::_Accelerator_view_hasher;

    _AMPIMP friend _Ret_ IUnknown * __cdecl direct3d::get_device(const accelerator_view &_Av);

    _AMPIMP friend accelerator_view __cdecl direct3d::create_accelerator_view(_In_ IUnknown *_D3D_device, queuing_mode qmode /* = queuing_mode_automatic */);

    friend _Accelerator_view_impl_ptr details::_Get_accelerator_view_impl_ptr(const accelerator_view& _Accl_view);

public:

    /// <summary>
    ///     Destructor
    /// </summary>
    _AMPIMP ~accelerator_view();

    /// <summary>
    ///     Copy constructor
    /// </summary>
    _AMPIMP accelerator_view(const accelerator_view &_Other);

    /// <summary>
    ///     Assignment operator
    /// </summary>
    _AMPIMP accelerator_view &operator=(const accelerator_view &_Other);

    /// <summary>
    ///     Get the accelerator for this accelerator view
    /// </summary>
    _AMPIMP accelerator get_accelerator() const;
    __declspec(property(get=get_accelerator)) Concurrency::accelerator accelerator;

    /// <summary>
    ///     Returns a boolean value indicating whether the accelerator view
    ///     was created with DEBUG layer enabled for extensive error reporting
    /// </summary>
    _AMPIMP bool get_is_debug() const;
    __declspec(property(get=get_is_debug)) bool is_debug;

    /// <summary>
    ///     Get the version for this accelerator view
    /// </summary>
    _AMPIMP unsigned int get_version() const;
    __declspec(property(get=get_version)) unsigned int version; // hiword=major, loword=minor

    /// <summary>
    ///     Get the queuing mode for this accelerator view
    /// </summary>
    _AMPIMP queuing_mode get_queuing_mode() const;
    __declspec(property(get=get_queuing_mode)) Concurrency::queuing_mode queuing_mode;

    /// <summary>
    ///     Return true if the other accelerator view is same as this accelerator view; false otherwise
    /// </summary>
    _AMPIMP bool operator==(const accelerator_view &_Other) const;

    /// <summary>
    ///     Return false if the other accelerator view is same as this accelerator view; true otherwise
    /// </summary>
    _AMPIMP bool operator!=(const accelerator_view &_Other) const;

    /// <summary>
    ///     Waits for completion of all commands submitted so far to this accelerator_view
    /// </summary>
    _AMPIMP void wait();

    /// <summary>
    ///     Submit all pending commands queued to this accelerator_view to the accelerator
    ///     for execution.
    /// </summary>
    _AMPIMP void flush();

    /// <summary>
    ///     Return a future to track the completion of all commands submitted so far to this accelerator_view
    /// </summary>
    _AMPIMP concurrency::completion_future create_marker();

private:

    // No default constructor
    accelerator_view();

    // Private constructor
    _AMPIMP accelerator_view(_Accelerator_view_impl_ptr _Impl);

private:

    _Accelerator_view_impl_ptr _M_impl;
};

namespace details
{
    inline _Accelerator_view_impl_ptr _Get_accelerator_view_impl_ptr(const accelerator_view& _Accl_view)
    {
        return _Accl_view._M_impl;
    }

    inline _Accelerator_impl_ptr _Get_accelerator_impl_ptr(const accelerator& _Accl)
    {
         return _Accl._M_impl;
    }

    // Type defining a hasher for accelerator_view objects
    // for use with std::unordered_set and std::unordered_map
    class _Accelerator_view_hasher
    {
    public:
        size_t operator()(const accelerator_view &_Accl_view) const
        {
            std::hash<_Accelerator_view_impl*> _HashFunctor;
            return _HashFunctor(_Accl_view._M_impl._Get_ptr());
        }
    };

    typedef std::unordered_set<accelerator_view, _Accelerator_view_hasher> _Accelerator_view_unordered_set;

    // Describes the N dimensional shape of a view in a buffer
    class _View_shape : public _Reference_counter
    {
    public:

        _AMPIMP static _Ret_ _View_shape* __cdecl _Create_view_shape(unsigned int _Rank, unsigned int _Linear_offset,
                                                                  const unsigned int *_Base_extent, const unsigned int *_View_offset,
                                                                  const unsigned int *_View_extent, const bool *_Projection_info = NULL);

        _AMPIMP void _Release();

        inline unsigned int _Get_rank() const
        {
            return _M_rank;
        }

        inline unsigned int _Get_linear_offset() const
        {
            return _M_linear_offset;
        }

        inline const unsigned int *_Get_base_extent() const
        {
            return _M_base_extent;
        }

        inline const unsigned int *_Get_view_offset() const
        {
            return _M_view_offset;
        }
        inline const unsigned int *_Get_view_extent() const
        {
            return _M_view_extent;
        }

        inline const bool *_Get_projection_info() const
        {
            return _M_projection_info;
        }
        
        inline bool _Is_valid(size_t _Buffer_size) const
        {
            // The end point of the base shape should not be greater than the size of the buffer
            size_t endLinearOffset = _M_linear_offset + _Get_extent_size(_M_rank, _M_base_extent);
            if (endLinearOffset > _Buffer_size) {
                return false;
            }

            return _Is_valid();
        }

        inline unsigned int _Get_view_size() const
        {
            return _Get_extent_size(_M_rank, _M_view_extent);
        }

        inline unsigned int _Get_view_linear_offset() const
        {
            unsigned int currMultiplier = 1;
            unsigned int linearOffset = _M_linear_offset;
            for (int _I = static_cast<int>(_M_rank - 1); _I >= 0; _I--)
            {
                linearOffset += (currMultiplier * _M_view_offset[_I]);
                currMultiplier *= _M_base_extent[_I];
            }

            return linearOffset;
        }

        static inline bool
        _Compare_extent_with_elem_size(unsigned int _Rank, const unsigned int *_Extent1, size_t _Elem_size1, const unsigned int *_Extent2, size_t _Elem_size2)
        {
            _ASSERTE((_Rank >= 1) && (_Extent1 != NULL)&& (_Extent2 != NULL));

            // The extents should match accounting for the element sizes of the respective buffers
            if ((_Extent1[_Rank - 1] * _Elem_size1) != (_Extent2[_Rank - 1] * _Elem_size2)) 
            {
                return false;
            }

            // Now compare the extent in all but the least significant dimension
            if ((_Rank > 1) && !_Compare_extent(_Rank - 1, _Extent1, _Extent2))
            {
                return false;
            }

            return true;
        }


        static inline bool
        _Compare_extent(unsigned int _Rank, const unsigned int *_Extent1, const unsigned int *_Extent2)
        {
            for (size_t _I = 0; _I < _Rank; ++_I) {
                if (_Extent1[_I] != _Extent2[_I]) {
                    return false;
                }
            }

            return true;
        }

        inline bool _Is_view_linear(unsigned int &_Linear_offset, unsigned int &_Linear_size) const
        {
            // It is linear if the rank is 1
            if (_M_rank == 1) {
                _Linear_offset = _M_linear_offset + _M_view_offset[0];
                _Linear_size = _M_view_extent[0];
                return true;
            }
            else if (_Compare_extent(_M_rank - 1, &_M_base_extent[1], &_M_view_extent[1]))
            {
                // Also linear if the base extent and view extent are same 
                // in all dimensions but the highest
                _Linear_offset = _Get_view_linear_offset();
                _Linear_size = _Get_view_size();
                return true;
            }

            return false;
        }

        inline bool _Overlaps(const _View_shape_ptr _Other) const
        {
            if (_Compare_base_shape(_Other))
            {
                // If the base shapes are identical we will do the N-dimensional
                // bounding box overlap test
                
                for (size_t _I = 0; _I < _M_rank; ++_I)
                {
                    if (!_Intervals_overlap(_M_view_offset[_I], _M_view_offset[_I] + _M_view_extent[_I] - 1,
                                            _Other->_M_view_offset[_I], _Other->_M_view_offset[_I] + _Other->_M_view_extent[_I] - 1)) 
                    {
                        return false;
                    }
                }

                return true;
            }
            else
            {
                // The base shapes are different. Check based on linear intervals
                size_t firstStart = _Get_view_linear_offset();
                size_t firstEnd = firstStart + _Get_view_size() - 1;

                size_t secondStart = _Other->_Get_view_linear_offset();
                size_t secondEnd = secondStart + _Other->_Get_view_size() - 1;

                return _Intervals_overlap(firstStart, firstEnd, secondStart, secondEnd);
            }
        }

        inline bool _Subsumes(const _View_shape_ptr _Other) const
        {
            // Subsumption test can only be done for shapes that have the same base shape or 
            // when both have a rank of 1
            if ((_M_rank == 1) && (_Other->_Get_rank() == 1)) 
            {
                size_t thisStart = _Get_view_linear_offset();
                size_t thisEnd = thisStart + _Get_view_size() - 1;

                size_t otherStart = _Other->_Get_view_linear_offset();
                size_t otherEnd = otherStart + _Other->_Get_view_size() - 1;

                return ((otherStart >= thisStart) && (otherEnd <= thisEnd));
            }

            if (!_Compare_base_shape(_Other)) {
                return false;
            }

            if (!_Contains(_Other->_Get_view_offset())) {
                return false;
            }

            std::vector<unsigned int> otherEndPointIndex(_M_rank);
            for (size_t _I = 0; _I < _M_rank; ++_I) {
                otherEndPointIndex[_I] = _Other->_Get_view_offset()[_I] + _Other->_Get_view_extent()[_I] - 1;
            }

            return _Contains(otherEndPointIndex.data());
        }

    private:
        // Private constructor to force construction through the _Create_view_shape method
        _View_shape(unsigned int _Rank, unsigned int _Linear_offset,
                    const unsigned int *_Base_extent, const unsigned int *_View_offset,
                    const unsigned int *_View_extent, const bool *_Projection_info);
        
        virtual ~_View_shape();

        // No default constructor or copy/assignment
        _View_shape();
        _View_shape(const _View_shape &_Other);
        _View_shape(_View_shape &&_Other);
        _View_shape& operator=(const _View_shape &_Other);
        _View_shape& operator=(_View_shape &&_Other);

        // Helper methods
        static bool _Intervals_overlap(size_t _First_start, size_t _First_end,
                                       size_t _Second_start, size_t _Second_end)
        {
            // Order the intervals by their start points
            if (_First_start > _Second_start) {
                size_t temp = _First_start;
                _First_start = _Second_start;
                _Second_start = temp;

                temp = _First_end;
                _First_end = _Second_end;
                _Second_end = temp;
            }

            // The start of the second one must be within the bounds of the first one
            return (_Second_start <= _First_end);
        }

        static unsigned int _Get_extent_size(unsigned int _Rank, const unsigned int *_Extent)
        {
            unsigned int totalExtent = 1;
            for (size_t _I = 0; _I < _Rank; ++_I) {
                totalExtent *= _Extent[_I];
            }

            return totalExtent;
        }

        inline bool _Is_valid() const
        {
            if (_M_rank == 0) {
                return false;
            }

            // Ensure the _M_view_offset + _M_view_extent is within the bounds of _M_base_extent
            size_t viewSize = 1;
            
            for (size_t _I = 0; _I < _M_rank; ++_I)
            {
                viewSize *= _M_view_extent[_I];
                if ((_M_view_offset[_I] + _M_view_extent[_I]) > _M_base_extent[_I]) {
                    return false;
                }
            }

            if (viewSize == 0) {
                return false;
            }

            return true;
        }

        inline bool _Compare_base_shape(const _View_shape_ptr _Other) const
        {
            return ((_M_rank == _Other->_M_rank) &&
                    (_M_linear_offset == _Other->_M_linear_offset) &&
                    _Compare_extent(_M_rank, _M_base_extent, _Other->_M_base_extent));
        }

        // Checks if the element at the specified index
        // is contained within this view shape
        // Assumes the rank of the index is same as the 
        // rank of this view's shape
        inline bool _Contains(const unsigned int* _Element_index) const
        {
            for (size_t _I = 0; _I < _M_rank; ++_I)
            {
                if ((_Element_index[_I] < _M_view_offset[_I]) ||
                    (_Element_index[_I] >= (_M_view_offset[_I] + _M_view_extent[_I]))) 
                {
                    return false;
                }
            }

            return true;
        }

    private:       

        unsigned int _M_rank;
        unsigned int _M_linear_offset;
        unsigned int *_M_base_extent;
        unsigned int *_M_view_offset;
        unsigned int *_M_view_extent;
        bool *_M_projection_info;
    };

    // This function creates a new _View_shape object from an existing _View_shape object when the data underlying the view
    // needs to be reinterpreted to use a different element size than the one used by the original view.
    inline
    _Ret_ _View_shape *_Create_reinterpreted_shape(const _View_shape_ptr _Source_shape, size_t _Curr_elem_size, size_t _New_elem_size)
    {
        unsigned int _Rank = _Source_shape->_Get_rank();
        size_t _LinearOffsetInBytes = _Source_shape->_Get_linear_offset() * _Curr_elem_size;
        size_t _BaseLSDExtentInBytes = (_Source_shape->_Get_base_extent())[_Rank - 1] * _Curr_elem_size;
        size_t _ViewLSDOffsetInBytes = (_Source_shape->_Get_view_offset())[_Rank - 1] * _Curr_elem_size;
        size_t _ViewLSDExtentInBytes = (_Source_shape->_Get_view_extent())[_Rank - 1] * _Curr_elem_size;

        _ASSERTE((_LinearOffsetInBytes % _New_elem_size) == 0);
        _ASSERTE((_BaseLSDExtentInBytes % _New_elem_size) == 0);
        _ASSERTE((_ViewLSDOffsetInBytes % _New_elem_size) == 0);
        _ASSERTE((_ViewLSDExtentInBytes % _New_elem_size) == 0);

        size_t _Temp_val = _LinearOffsetInBytes / _New_elem_size;
        _ASSERTE(_Temp_val <= UINT_MAX);
        unsigned int _New_linear_offset = static_cast<unsigned int>(_Temp_val);

        std::vector<unsigned int> _New_base_extent(_Rank);
        std::vector<unsigned int> _New_view_offset(_Rank);
        std::vector<unsigned int> _New_view_extent(_Rank);
        for (unsigned int i = 0; i < _Rank - 1; ++i) {
            _New_base_extent[i] = (_Source_shape->_Get_base_extent())[i];
            _New_view_offset[i] = (_Source_shape->_Get_view_offset())[i];
            _New_view_extent[i] = (_Source_shape->_Get_view_extent())[i];
        }

        // The extent in the least significant dimension needs to be adjusted
        _Temp_val = _BaseLSDExtentInBytes / _New_elem_size;
        _ASSERTE(_Temp_val <= UINT_MAX);
        _New_base_extent[_Rank - 1] = static_cast<unsigned int>(_Temp_val);

        _Temp_val = _ViewLSDOffsetInBytes / _New_elem_size;
        _ASSERTE(_Temp_val <= UINT_MAX);
        _New_view_offset[_Rank - 1] = static_cast<unsigned int>(_Temp_val);

        _Temp_val = _ViewLSDExtentInBytes / _New_elem_size;
        _ASSERTE(_Temp_val <= UINT_MAX);
        _New_view_extent[_Rank - 1] = static_cast<unsigned int>(_Temp_val);

        return _View_shape::_Create_view_shape(_Rank, _New_linear_offset, _New_base_extent.data(), _New_view_offset.data(), _New_view_extent.data());
    }

    //  Class manages a raw buffer in a accelerator view
    class _Buffer : public _Reference_counter
    {
        friend class _CPU_accelerator_view_impl;
        friend class _D3D_accelerator_view_impl;
        friend class _D3D_temp_staging_buffer_cache;

    public:

        // Force construction through these static public method to ensure that _Buffer
        // objects are allocated in the runtime

        // Allocate a new buffer on the specified accelerator_view
        _AMPIMP static _Ret_ _Buffer * __cdecl _Create_buffer(accelerator_view _Accelerator_view, size_t _Num_elems,
                                                        size_t _Elem_size, bool _Is_temp = false);

        // Create a buffer object from a pre-allocated storage on the specified accelerator_view. This can be thought
        // of as the accelerator_view "adopting" the passed data buffer.
        _AMPIMP static _Ret_ _Buffer * __cdecl _Create_buffer(_In_ void *_Data_ptr, accelerator_view _Accelerator_view, size_t _Num_elems,
                                                        size_t _Elem_size);

        // Create a staging buffer on the specified accelerator_view which can be accesed on the cpu upon mapping.
        _AMPIMP static _Ret_ _Buffer * __cdecl _Create_stage_buffer(accelerator_view _Accelerator_view, accelerator_view _Access_on_accelerator_view,
                                                              size_t _Num_elems, size_t _Elem_size, bool _Is_temp = false);

        // Creates a temp staging buffer of the requested size. This function may create 
        // a staging buffer smaller than the requested size.
        _AMPIMP static size_t __cdecl _Get_temp_staging_buffer(accelerator_view _Av, size_t _Requested_num_elems,
                                                               size_t _Elem_size, _Inout_ _Buffer **_Ret_buf);

        _AMPIMP void _Release();

        // Map a staging buffer for access on the CPU.
        _AMPIMP _Ret_ void * _Map_stage_buffer(_Access_mode _Map_type, bool _Wait);

        // Unmap a staging buffer denying CPU access
        _AMPIMP void _Unmap_stage_buffer();

        //  Copy data to _Dest asynchronously.
        _AMPIMP _Event _Copy_to_async(_Out_ _Buffer * _Dest, size_t _Num_elems, size_t _Src_offset = 0, size_t _Dest_offset = 0);

        //  Copy data to _Dest asynchronously.
        _AMPIMP _Event _Copy_to_async(_Out_ _Buffer * _Dest, _View_shape_ptr _Src_shape, _View_shape_ptr _Dest_shape);

        _AMPIMP accelerator_view _Get_accelerator_view() const;
        _AMPIMP accelerator_view _Get_access_on_accelerator_view() const;

        _AMPIMP void _Register_view(_In_ _View_key _Key);
        _AMPIMP void _Unregister_view(_In_ _View_key _Key);

        // Return the raw data ptr - only a accelerator view implementation can interpret
        // this raw pointer. This method should usually not be used in the AMP header files
        // The _Get_host_ptr is the right way for accessing the host accesible ptr for a buffer
        _Ret_ void * _Get_data_ptr() const
        {
            return _M_data_ptr;
        }

        // Returns the host accessible ptr corresponding to the buffer. This would
        // return NULL when the buffer is inaccesible on the CPU
        _Ret_ void * _Get_host_ptr() const
        {
            return _M_host_ptr;
        }

        size_t _Get_elem_size() const
        {
            return _M_elem_size;
        }

        size_t _Get_num_elems() const
        {
            return _M_num_elems;
        }

        _Accelerator_view_impl_ptr _Get_accelerator_view_impl() const
        {
            return _M_accelerator_view;
        }

        _Accelerator_view_impl_ptr _Get_access_on_accelerator_view_impl() const
        {
            return _M_access_on_accelerator_view;
        }

        bool _Owns_data() const 
        {
            return _M_owns_data;
        }

        bool _Is_staging() const
        {
            return _M_is_staging;
        }

        bool _Is_temp() const
        {
            return _M_is_temp;
        }

        bool _Is_adopted() const
        {
            // Is it adopted from interop?
            return _M_is_adopted;
        }

        virtual bool _Is_buffer()
        {
            return true;
        }

    protected:

        // The _Buffer constructor is protected to force construction through the static 
        // _Create_buffer method to ensure the object is allocated in the runtime
        _Buffer(_Accelerator_view_impl_ptr _Av, _In_ void *_Buffer_data_ptr, _In_ void * _Host_ptr, size_t _Num_elems,
                size_t _Elem_size, bool _Owns_data = true, bool _Is_staging = false, bool _Is_temp = false, bool _Is_adopted = false);

        // protected destructor to force deletion through _Release
        virtual ~_Buffer();

        // No default consturctor, copy constructor and assignment operator
        _Buffer();
        _Buffer(const _Buffer &rhs);
        _Buffer &operator=(const _Buffer &rhs);

        void _Set_host_ptr(_In_ void *_Host_ptr)
        {
            _M_host_ptr = _Host_ptr;
        }

        void _Set_data_ptr(_In_ IUnknown *_Data_ptr)
        {
            _M_data_ptr = _Data_ptr;
        }

    protected:
        _Accelerator_view_impl_ptr _M_accelerator_view;
        _Accelerator_view_impl_ptr _M_access_on_accelerator_view;
        void * _M_data_ptr;
        void * _M_host_ptr;
        size_t _M_elem_size;
        size_t _M_num_elems;
        bool   _M_owns_data;
        bool   _M_is_staging;

        // Used to determine how to map the staging buffer after its involved in a copy
        bool   _M_is_temp;

        bool   _M_is_adopted;
    private:
        // A set of view_keys to invalidate whenever the host ptr of a staging buffer is invalidated
        std::unordered_set<_View_key> _M_view_keys;
        Concurrency::critical_section _M_critical_section;
    };

    //  Class manages a texture in a accelerator view
    class _Texture : public _Buffer
    {
        friend class _CPU_accelerator_view_impl;
        friend class _D3D_accelerator_view_impl;

    public:

        // Allocate a new texture on the specified accelerator_view
        _AMPIMP static _Ret_ _Texture * __cdecl _Create_texture(accelerator_view _Accelerator_view,
                                                          unsigned int _Rank,
                                                          size_t _Width, size_t _Height, size_t _Depth,
                                                          unsigned int _Mip_levels,
                                                          _Short_vector_base_type_id _Type_id,
                                                          unsigned int _Num_channels,
                                                          unsigned int _Bits_per_channel,
                                                          bool _Is_temp = false);

        // Create a texture object from a pre-allocated storage on the specified accelerator_view. This can be thought
        // of as the accelerator_view "adopting" the passed data buffer.
        _AMPIMP static _Ret_ _Texture * __cdecl _Create_texture(unsigned int _Rank, _Texture_base_type_id _Id, _In_ IUnknown *_Data_ptr, accelerator_view _Accelerator_view);

        // Create a staging texture on the specified accelerator_view which can be accesed on the cpu upon mapping.
        _AMPIMP static _Ret_ _Texture * __cdecl _Create_stage_texture(accelerator_view _Accelerator_view, accelerator_view _Access_on_accelerator_view,
                                                       unsigned int _Rank,
                                                       size_t _Width, size_t _Height, size_t _Depth,
                                                       unsigned int _Mip_levels,
                                                       unsigned int _Format,
                                                       bool _Is_temp = false);

        //  Copy data to _Dest asynchronously for textures. The two textures must have been created with 
        //  exactly the same extent and with compatible physical formats.
        _AMPIMP _Event _Copy_to_async(_Out_ _Texture * _Dest);

        size_t _Get_width() const
        {
            return _M_width;
        }

        size_t _Get_height() const
        {
            return _M_height;
        }

        size_t _Get_depth() const
        {
            return _M_depth;
        }

        unsigned int _Get_rank() const
        {
            return _M_rank;
        }

        unsigned int _Get_format() const
        {
            return _M_format;
        }

        unsigned int _Get_num_channels() const
        {
            return _M_num_channels;
        }

        unsigned int _Get_bits_per_channel() const
        {
            // For texture adopted from interop, return 0.
            return _Is_adopted() ? 0 : _M_bits_per_channel;
        }

        unsigned int _Get_bits_per_element() const
        {
            return _M_bits_per_channel * _M_num_channels;
        }

        unsigned int _Get_data_length() const  // in bytes
        {
            unsigned int _Bits_per_byte = 8;
            return (_Get_bits_per_element() * static_cast<unsigned int>(_M_num_elems))/ _Bits_per_byte;
        }

        unsigned int _Get_mip_levels() const
        {
            return _M_mip_levels;
        }

        virtual bool _Is_buffer()
        {
            return false;
        }

        size_t _Get_row_pitch() const
        {
            return _M_row_pitch;
        }

        void _Set_row_pitch(size_t _Val)
        {
            _M_row_pitch = _Val;
        }

        size_t _Get_depth_pitch() const
        {
            return _M_depth_pitch;
        }

        void _Set_depth_pitch(size_t _Val)
        {
            _M_depth_pitch = _Val;
        }

    private:

        // The _Texture constructor is private to force construction through the static 
        // _Create_texture method to ensure the object is allocated in the runtime
        _Texture(_Accelerator_view_impl_ptr _Av, _In_ void *_Texture_data_ptr, _In_ void * _Host_ptr, 
                 unsigned int _Rank,
                 size_t _Width, size_t _Height, size_t _Depth,
                 unsigned int _Mip_levels,
                 unsigned int _Format,
                 unsigned int _Num_channels,
                 unsigned int _Bits_per_channel,
                 bool _Owns_data = true, bool _Is_staging = false, bool _Is_temp = false, bool _Is_adopted = false);

        // Private destructor to force deletion through _Release
        ~_Texture();

        // No default consturctor, copy constructor and assignment operator
        _Texture();
        _Texture(const _Texture &rhs);
        _Texture &operator=(const _Texture &rhs);

        // Texture only
        unsigned int _M_rank;
        size_t _M_width;
        size_t _M_height;
        size_t _M_depth;
        unsigned int _M_format;
        unsigned int _M_bits_per_channel;
        unsigned int _M_num_channels;
        unsigned int _M_mip_levels;

        size_t _M_row_pitch;
        size_t _M_depth_pitch;
    };

    // Finds the greatest common divisor of 2 unsigned integral numbers using Euclid's algorithm
    template <typename _T>
    inline _T _Greatest_common_divisor(_T _M, _T _N)
    {
        static_assert(std::is_unsigned<_T>::value, "This GCD function only supports unsigned integral types");

        _ASSERTE((_M > 0) && (_N > 0));

        if (_N > _M) {
            std::swap(_N , _M);
        }

        _T _Temp;
        while (_N > 0)
        {
            _Temp = _N;
            _N = _M % _N;
            _M = _Temp;
        }

        return _M;
    }

    // Finds the least common multiple of 2 unsigned integral numbers using their greatest_common_divisor
    template <typename _T>
    inline _T _Least_common_multiple(_T _M, _T _N)
    {
        static_assert(std::is_unsigned<_T>::value, "This LCM function only supports unsigned integral types");

        _ASSERTE((_M > 0) && (_N > 0));

        _T _Gcd = _Greatest_common_divisor(_M, _N);
        return ((_M / _Gcd) * _N);
    }

    template <typename InputIterator, typename _Value_type>
    inline _Event _Copy_impl(InputIterator _SrcFirst, InputIterator _SrcLast, size_t _NumElemsToCopy, _Out_ _Buffer * _Dst, size_t _Dest_offset)
    {
        if (_NumElemsToCopy == 0) {
            return _Event();
        }

        if (_Dst == NULL) {
            throw runtime_exception("Failed to copy to buffer.", E_INVALIDARG);
        }

#pragma warning ( push )
#pragma warning ( disable : 6001 ) // Using uninitialized memory '*_Dst'
		if (((_NumElemsToCopy * sizeof(_Value_type)) + (_Dest_offset * _Dst->_Get_elem_size())) > (_Dst->_Get_num_elems() * _Dst->_Get_elem_size()))
        {
            throw runtime_exception("Invalid _Src argument(s). _Src size exceeds total size of the _Dest.", E_INVALIDARG);
        }
#pragma warning ( pop )

        _ASSERTE(_NumElemsToCopy == (size_t)(std::distance(_SrcFirst, _SrcLast)));

        // If the dest has CPU ptr then we do the copy on
        // accelerator(accelerator::cpu_accelerator).default_view
        if (_Dst->_Get_host_ptr() != NULL)
        {
            // The _Dest is accessible on host. We just need to do a std::copy using a raw pointer as OutputIterator
            _Value_type *_DestPtr = reinterpret_cast<_Value_type*>(reinterpret_cast<char*>(_Dst->_Get_host_ptr()) + (_Dest_offset * _Dst->_Get_elem_size()));
            std::copy(_SrcFirst, _SrcLast, stdext::make_unchecked_array_iterator(_DestPtr));

            return _Event();
        }
        else
        {
            // _Dest is on a device. Lets create a temp staging buffer on the _Dest accelerator_view and copy the input over
            // We may create a staging buffer of size smaller than the copy size and in that case we will perform the copy
            // as a series of smaller copies
            _Buffer_ptr _PDestBuf = _Dst;
            size_t _NumElemsToCopyRemaining = _NumElemsToCopy;
            size_t _CurrDstOffset = _Dest_offset;
            InputIterator _CurrStartIter = _SrcFirst;
            _Event _Ev;

            do
            {
                _Buffer *_PTempStagingBuf = NULL;
                size_t _StagingBufNumElems = _Buffer::_Get_temp_staging_buffer(_Dst->_Get_accelerator_view(),
                                                                               _NumElemsToCopyRemaining, sizeof(_Value_type),
                                                                               &_PTempStagingBuf);

                _ASSERTE(_PTempStagingBuf != NULL);
                _Buffer_ptr _PDestStagingBuf = _PTempStagingBuf;

                // The total byte size of a copy chunk must be an integral multiple of both the
                // destination buffer's element size and sizeof(_Value_type).
                size_t _Lcm = _Least_common_multiple(_Dst->_Get_elem_size(), sizeof(_Value_type));
                size_t _AdjustmentRatio = _Lcm / sizeof(_Value_type);

                InputIterator _CurrEndIter = _CurrStartIter;
                size_t _CurrNumElemsToCopy;
                if (_NumElemsToCopyRemaining <= _StagingBufNumElems) {
                    _CurrNumElemsToCopy = _NumElemsToCopyRemaining;
                    _CurrEndIter = _SrcLast;
                }
                else
                {
                    // We need to adjust the _StagingBufNumElems to be a multiple of the 
                    // least common multiple of the destination buffer's element size and sizeof(_Value_type).
                    _CurrNumElemsToCopy = (_StagingBufNumElems / _AdjustmentRatio) * _AdjustmentRatio;
                    std::advance(_CurrEndIter, _CurrNumElemsToCopy);
                }

                _ASSERTE((_CurrNumElemsToCopy % _AdjustmentRatio) == 0);

                // This would not actually never block since we just created this staging buffer or are using
                // a cached one that is not in use
                _PDestStagingBuf->_Map_stage_buffer(_Write_access, true);

                // Copy from input to the staging using a raw pointer as OutputIterator
                std::copy(_CurrStartIter, _CurrEndIter, stdext::make_unchecked_array_iterator(reinterpret_cast<_Value_type*>(_PDestStagingBuf->_Get_host_ptr())));

                _Ev = _Ev._Add_event(details::_Copy_impl(_PDestStagingBuf, 0, _PDestBuf, _CurrDstOffset, _CurrNumElemsToCopy));

                // Adjust the iterators and offsets
                _NumElemsToCopyRemaining -= _CurrNumElemsToCopy;
                _CurrDstOffset += (_CurrNumElemsToCopy * sizeof(_Value_type)) / _Dst->_Get_elem_size();
                _CurrStartIter = _CurrEndIter;

            } while (_NumElemsToCopyRemaining != 0);

            return _Ev;
        }
    }

    // The std::advance method is only supported for InputIterators and hence we have a custom implementation
    // which forwards to the std::advance if the iterator is an input iterator and uses a loop based advance
    // implementation otherwise
    template<typename _InputIterator, typename _Distance>
    typename std::enable_if<std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<_InputIterator>::iterator_category>::value>::type
    _Advance_output_iterator(_InputIterator &_Iter, _Distance _N)
    {
        std::advance(_Iter, _N);
    }

    template<typename _OutputIterator, typename _Distance>
    typename std::enable_if<!std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<_OutputIterator>::iterator_category>::value>::type
    _Advance_output_iterator(_OutputIterator &_Iter, size_t _N)
    {
        for (size_t i = 0; i < _N; ++i)
        {
            _Iter++;
        }
    }

    template <typename OutputIterator, typename _Value_type>
    inline _Event _Copy_impl(_In_ _Buffer *_Src, size_t _Src_offset, size_t _Num_elems, OutputIterator _DestIter)
    {
        if ((_Src == NULL) || ((_Src_offset + _Num_elems) > _Src->_Get_num_elems())) {
            throw runtime_exception("Failed to copy to buffer.", E_INVALIDARG);
        }

        if (_Num_elems == 0) {
            return _Event();
        }

        _Event _Ev;

        size_t _NumElemsToCopy = (_Num_elems * _Src->_Get_elem_size()) / sizeof(_Value_type);

        // If the src has CPU ptr then we do the copy on
        // accelerator(accelerator::cpu_accelerator).default_view
        if (_Src->_Get_host_ptr() != NULL)
        {
            // The _Src is accessible on host. We just need to do a std::copy
            const _Value_type *_PFirst = reinterpret_cast<const _Value_type*>(reinterpret_cast<char*>(_Src->_Get_host_ptr()) + (_Src_offset * _Src->_Get_elem_size()));
            std::copy(_PFirst, _PFirst + _NumElemsToCopy, _DestIter);
        }
        else
        {
            // The _Src is on the device. We need to copy it out to a temporary staging array
            // We may create a staging buffer of size smaller than the copy size and in that case we will
            // perform the copy as a series of smaller copies
            _Buffer_ptr _PSrcBuf = _Src;
            _Buffer *_PTempStagingBuf = NULL;
            size_t _StagingBufNumElems = _Buffer::_Get_temp_staging_buffer(_Src->_Get_accelerator_view(), _NumElemsToCopy,
                                                                           sizeof(_Value_type), &_PTempStagingBuf);

            // The total byte size of a copy chunk must be an integral multiple of both the
            // source buffer's element size and sizeof(_Value_type).
            size_t _Lcm = _Least_common_multiple(_Src->_Get_elem_size(), sizeof(_Value_type));
            size_t _AdjustmentRatio = _Lcm / sizeof(_Value_type);

            _ASSERTE(_PTempStagingBuf != NULL);
            _Buffer_ptr _PSrcStagingBuf = _PTempStagingBuf;

            size_t _CurrNumElemsToCopy;
            if (_NumElemsToCopy <= _StagingBufNumElems)
            {
                _CurrNumElemsToCopy = _NumElemsToCopy;
            }
            else
            {
                // We need to adjust the _StagingBufNumElems to be a multiple of the 
                // least common multiple of the source buffer's element size and sizeof(_Value_type).
                _CurrNumElemsToCopy = (_StagingBufNumElems / _AdjustmentRatio) * _AdjustmentRatio;
            }

            _ASSERTE((_CurrNumElemsToCopy % _AdjustmentRatio) == 0);

            size_t _NumElemsToCopyRemaining = _NumElemsToCopy - _CurrNumElemsToCopy;

            _Ev = _PSrcBuf->_Copy_to_async(_PSrcStagingBuf, (_CurrNumElemsToCopy * sizeof(_Value_type)) / _PSrcBuf->_Get_elem_size(), _Src_offset, 0);

            if (_NumElemsToCopyRemaining != 0)
            {
                _Ev = _Ev._Add_continuation(std::function<_Event()>([_DestIter, _PSrcBuf, _PSrcStagingBuf,
                                                                     _CurrNumElemsToCopy, _NumElemsToCopyRemaining,
                                                                     _Src_offset]() mutable -> _Event 
                {
                    // Initiate an asynchronous copy of the remaining part so that this part of the copy
                    // makes progress while we consummate the copying of the first part
                    size_t _CurrSrcOffset = _Src_offset + ((_CurrNumElemsToCopy * sizeof(_Value_type)) / _PSrcBuf->_Get_elem_size());
                    OutputIterator _CurrDestIter = _DestIter;
                    _Advance_output_iterator<decltype(_CurrDestIter), size_t>(_CurrDestIter, _CurrNumElemsToCopy);
                    _Event _Ret_ev = _Copy_impl<OutputIterator, _Value_type>(_PSrcBuf._Get_ptr(), _CurrSrcOffset,
                                                                             (_NumElemsToCopyRemaining * sizeof(_Value_type)) / _PSrcBuf->_Get_elem_size(),
                                                                             _CurrDestIter);

                    // Now copy the data from staging buffer to the destination
                    _Value_type *_PFirst = reinterpret_cast<_Value_type*>(_PSrcStagingBuf->_Get_host_ptr());
                    std::copy(_PFirst, _PFirst + _CurrNumElemsToCopy, _DestIter);
                    return _Ret_ev;
                }));
            }
            else
            {
                _Ev = _Ev._Add_continuation(std::function<_Event()>([_DestIter, _PSrcStagingBuf, _CurrNumElemsToCopy]() mutable -> _Event 
                {
                    _Value_type *_PFirst = reinterpret_cast<_Value_type*>(_PSrcStagingBuf->_Get_host_ptr());
                    std::copy(_PFirst, _PFirst + _CurrNumElemsToCopy, _DestIter);
                    return _Event();
                }));
            }
        }

        return _Ev;
    }

    // Linear copy between buffers across AVs
    inline _Event _Copy_impl(_In_ _Buffer *_Src, size_t _Src_offset,
                             _Out_ _Buffer * _Dst, size_t _Dest_offset, size_t _Num_elems)
    {        
        if ((_Src == NULL) || (_Dst == NULL)) {
            throw runtime_exception("Failed to copy between buffers.", E_INVALIDARG);
        }

        if (_Num_elems == 0) {
            return _Event();
        }

#pragma warning ( push )
#pragma warning ( disable : 6001 ) // Using uninitialized memory '*_Dst'
        // If both the src and dest have CPU ptrs then we do the copy on
        // accelerator(accelerator::cpu_accelerator).default_view by calling 
        // the cpu_buffer_copy_async helper method
        if ((_Src->_Get_host_ptr() != NULL) && (_Dst->_Get_host_ptr() != NULL))
        {
            // This covers the case of copying between arrays on cpu_accelerator and device staging arrays.

            Concurrency::accelerator_view cpuAcceleratorView = accelerator(accelerator::cpu_accelerator).default_view;

            _Buffer_ptr pSrcBuffer = _Buffer::_Create_buffer(_Src->_Get_host_ptr(), cpuAcceleratorView,
                                                             _Src->_Get_num_elems(), _Src->_Get_elem_size());
            _Buffer_ptr pDestBuffer = _Buffer::_Create_buffer(_Dst->_Get_host_ptr(), cpuAcceleratorView,
                                                              _Dst->_Get_num_elems(), _Dst->_Get_elem_size());
            return pSrcBuffer->_Copy_to_async(pDestBuffer, _Num_elems, _Src_offset, _Dest_offset);
        }
#pragma warning ( pop )
        else
        {
            // Either the src or dest is a non-staging device array

            // If both the buffers are on the same accelerator_view call the direct copy method
            // Note: Must compare the accelerator_view of the buffers and not the arrays since
            // the array.accelerator_view is the one where the array is accesbile and for the copy
            // purppose we need to determine where the buffer is allocated which corresponds to the
            // arrays underlying buffer's accelerator_view.
            if (_Src->_Get_accelerator_view() == _Dst->_Get_accelerator_view())
            {
                // This covers the cases of copying between 2 buffers (staging and non-staging) on the
                // same accelerator_view

                return _Src->_Copy_to_async(_Dst, _Num_elems, _Src_offset, _Dest_offset);
            }
            else
            {
                // The buffers are on different accelerator_views

                // If the src is accessible on the cpu
                if (_Src->_Get_host_ptr() != NULL)
                {
                    // The source array is accessible on the host but the destination array is not.
                    auto _SrcFirst = stdext::make_unchecked_array_iterator(reinterpret_cast<const unsigned int*>(_Src->_Get_host_ptr()) + ((_Src_offset * _Src->_Get_elem_size())/sizeof(unsigned int)));
                    auto _SrcLast = _SrcFirst;
                    std::advance(_SrcLast, (_Num_elems * _Src->_Get_elem_size())/sizeof(unsigned int));

                    return _Copy_impl<decltype(_SrcFirst), unsigned int>(_SrcFirst, _SrcLast, (_Num_elems * _Src->_Get_elem_size()) / sizeof(unsigned int), _Dst, _Dest_offset);
                }
                else
                {
                    // The src is on the device. See if the dest is accesible on the host
                    if (_Dst->_Get_host_ptr() != NULL)
                    {
                        // The src array is not accessible on the cpu but the dest array is accessible on cpu
                        auto _DestIter = stdext::make_unchecked_array_iterator(reinterpret_cast<unsigned int*>(_Dst->_Get_host_ptr()) + ((_Dest_offset * _Dst->_Get_elem_size()) / sizeof(unsigned int)));
                        return _Copy_impl<decltype(_DestIter), unsigned int>(_Src, _Src_offset, _Num_elems, _DestIter);
                    }
                    else
                    {
                        // Neither the source and destination are accessible on the CPU and they are on different
                        // accelerator_views. Now we will create a temporary staging buffer on the dest accelerator_view
                        // and copy over the src contents to it.
                        // We may create a staging buffer of size smaller that the copy size and in that case we will
                        // perform the copy as a series of smaller copies
                        _Buffer_ptr _PSrcBuf = _Src;
                        _Buffer_ptr _PDestBuf = _Dst;

                        _Buffer *_PTempStagingBuf = NULL;
                        size_t _StagingBufNumElems = _Buffer::_Get_temp_staging_buffer(_Src->_Get_accelerator_view(), _Num_elems,
                                                                                       _Src->_Get_elem_size(), &_PTempStagingBuf);

                        // The total byte size of a copy chunk must be an integral multiple of both the 
                        // source buffer's element size and destination buffer's element size. 
                        size_t _Lcm = _Least_common_multiple(_Src->_Get_elem_size(), _Dst->_Get_elem_size());
                        size_t _AdjustmentRatio = _Lcm / _Src->_Get_elem_size();

                        _ASSERTE(_PTempStagingBuf != NULL);
                        _Buffer_ptr _PDestStagingBuf = _PTempStagingBuf;

                        size_t _CurrNumElemsToCopy;
                        if (_Num_elems <= _StagingBufNumElems)
                        {
                            _CurrNumElemsToCopy = _Num_elems;
                        }
                        else
                        {
                            // We need to adjust the _StagingBufNumElems to be a multiple of the least common
                            // multiple of the source buffer's element size and destination buffer's element size.
                            _CurrNumElemsToCopy = (_StagingBufNumElems / _AdjustmentRatio) * _AdjustmentRatio;
                        }
                        
                        _ASSERTE((_CurrNumElemsToCopy % _AdjustmentRatio) == 0);

                        _Event _Ev = _Copy_impl(_PSrcBuf, _Src_offset, _PDestStagingBuf, 0, _CurrNumElemsToCopy);
                        size_t _NumElemsToCopyRemaining = _Num_elems - _CurrNumElemsToCopy;

                        if (_NumElemsToCopyRemaining != 0)
                        {
                            _Ev = _Ev._Add_continuation(std::function<_Event()>([_PDestStagingBuf, _PSrcBuf, _Src_offset, _PDestBuf,
                                                                                 _Dest_offset, _CurrNumElemsToCopy,
                                                                                 _NumElemsToCopyRemaining]() mutable -> _Event 
                            {
                                // Lets initiate the copy of the remaining part asynchronously so that this makes progress
                                // while we consummate the first portion of the copy
                                size_t _CurrSrcOffset = _Src_offset + _CurrNumElemsToCopy;
                                size_t _CurrDstOffset = _Dest_offset + ((_CurrNumElemsToCopy * _PSrcBuf->_Get_elem_size()) / _PDestBuf->_Get_elem_size());
                                _Event _Temp_ev = _Copy_impl(_PSrcBuf, _CurrSrcOffset, _PDestBuf, _CurrDstOffset, _NumElemsToCopyRemaining);
                                _Event _Ret_ev = _Copy_impl(_PDestStagingBuf, 0, _PDestBuf, _Dest_offset, _CurrNumElemsToCopy);
                                return _Ret_ev._Add_event(_Temp_ev);
                            }));
                        }
                        else
                        {
                            _Ev = _Ev._Add_continuation(std::function<_Event()>([_PDestStagingBuf, _PDestBuf, _Dest_offset,
                                                                                 _CurrNumElemsToCopy]() mutable -> _Event 
                            {
                                return _Copy_impl(_PDestStagingBuf, 0, _PDestBuf, _Dest_offset, _CurrNumElemsToCopy);
                            }));
                        }

                        return _Ev;
                    }
                }
            }
        }
    }

    // Structured copy between buffers across AVs
    inline _Event _Copy_impl(_In_ _Buffer *_Src, _View_shape_ptr _Src_shape,
                             _Out_ _Buffer * _Dst, _View_shape_ptr _Dst_shape)
    {
        if ((_Src == NULL) || (_Dst == NULL) || (_Src_shape == NULL) || (_Dst_shape == NULL))
        {
            throw runtime_exception("Failed to copy between buffers.", E_INVALIDARG);
        }
        if (_Src_shape->_Get_rank() != _Dst_shape->_Get_rank())
        {
            throw runtime_exception("Failed to copy because ranks do not match.", E_INVALIDARG);
        }

#pragma warning ( push )
#pragma warning ( disable : 6001 ) // Using uninitialized memory '*_Dst'
        // The extents should match accounting for the element sizes of the respective buffers
        if (!_View_shape::_Compare_extent_with_elem_size(_Src_shape->_Get_rank(), _Src_shape->_Get_view_extent(),
                                                        _Src->_Get_elem_size(), _Dst_shape->_Get_view_extent(), _Dst->_Get_elem_size())) 
        {
            throw runtime_exception("Failed to copy because extents do not match.", E_INVALIDARG);
        }
#pragma warning ( pop )

        // If both the _Src and _Dst shapes are linear then perform a linear copy
        unsigned int _Src_linear_offset, _Src_linear_size;
        unsigned int _Dst_linear_offset, _Dst_linear_size;
        if (_Src_shape->_Is_view_linear(_Src_linear_offset, _Src_linear_size) && 
            _Dst_shape->_Is_view_linear(_Dst_linear_offset, _Dst_linear_size))
        {
            return _Copy_impl(_Src, _Src_linear_offset, _Dst, _Dst_linear_offset, _Src_linear_size);
        }
        
        size_t numElementsToCopy = _Src_shape->_Get_view_size();

        // If both the src and dest have CPU ptrs then we do the copy on
        // accelerator(accelerator::cpu_accelerator).default_view by calling 
        // the cpu_buffer_copy_async helper method
        if ((_Src->_Get_host_ptr() != NULL) && (_Dst->_Get_host_ptr() != NULL))
        {
            // This covers the case of copying between arrays on cpu_accelerator and device staging arrays.

            Concurrency::accelerator_view cpuAcceleratorView = accelerator(accelerator::cpu_accelerator).default_view;

            _Buffer_ptr pSrcBuffer = _Buffer::_Create_buffer(_Src->_Get_host_ptr(), cpuAcceleratorView,
                                                             _Src->_Get_num_elems(), _Src->_Get_elem_size());
            _Buffer_ptr pDestBuffer = _Buffer::_Create_buffer(_Dst->_Get_host_ptr(), cpuAcceleratorView,
                                                              _Dst->_Get_num_elems(), _Dst->_Get_elem_size());
            return pSrcBuffer->_Copy_to_async(pDestBuffer, _Src_shape, _Dst_shape);
        }
        else
        {
            // Either the src or dest is a non-staging device array

            // If both the buffers are on the same accelerator_view call the direct copy method
            // Note: Must compare the accelerator_view of the buffers and not the arrays since
            // the array.accelerator_view is the one where the array is accesbile and for the copy
            // purppose we need to determine where the buffer is allocated which corresponds to the
            // arrays underlying buffer's accelerator_view.
            if (_Src->_Get_accelerator_view() == _Dst->_Get_accelerator_view())
            {
                // This covers the cases of copying between 2 buffers (staging and non-staging) on the
                // same accelerator_view

                return _Src->_Copy_to_async(_Dst, _Src_shape, _Dst_shape);
            }
            else
            {
                // The buffers are on different accelerator_views and at least one of them is not
                // host accesible

                // If the src is accessible on the cpu
                if (_Src->_Get_host_ptr() != NULL)
                {
                    // The source array is accessible on the host but the destination array is not.

                    // Create a temporary staging buffer on the destination buffer's acclerator_view
                    _Buffer_ptr pTempStagingBuf = _Buffer::_Create_stage_buffer(_Dst->_Get_accelerator_view(), accelerator(accelerator::cpu_accelerator).default_view,
                                                                                (numElementsToCopy * _Src->_Get_elem_size()) / _Dst->_Get_elem_size(), _Dst->_Get_elem_size(), true);

                    pTempStagingBuf->_Map_stage_buffer(_Write_access, true);

                    std::vector<unsigned int> _ZeroOffset(_Src_shape->_Get_rank(), 0);
                    _View_shape_ptr pTempStagingShape = _View_shape::_Create_view_shape(_Src_shape->_Get_rank(), 0, _Dst_shape->_Get_view_extent(), _ZeroOffset.data(), _Dst_shape->_Get_view_extent());

                    _Copy_impl(_Src, _Src_shape, pTempStagingBuf, pTempStagingShape)._Get();

                    // Now copy from the temp staging to the device buffer
                    return pTempStagingBuf->_Copy_to_async(_Dst, pTempStagingShape, _Dst_shape);
                }
                else
                {
                    // The src is on the device. We need to copy it out to a temporary staging array
                    _Buffer_ptr pTempStagingBuf = _Buffer::_Create_stage_buffer(_Src->_Get_accelerator_view(), accelerator(accelerator::cpu_accelerator).default_view,
                                                                                numElementsToCopy, _Src->_Get_elem_size(), true);

                    std::vector<unsigned int> _ZeroOffset(_Src_shape->_Get_rank(), 0);
                    _View_shape_ptr pTempStagingShape = _View_shape::_Create_view_shape(_Src_shape->_Get_rank(), 0, _Src_shape->_Get_view_extent(), _ZeroOffset.data(), _Src_shape->_Get_view_extent());

                    _Event _Ev = _Src->_Copy_to_async(pTempStagingBuf, _Src_shape, pTempStagingShape);

                    if (_Dst->_Get_host_ptr() != NULL)
                    {
                        // The src array is not accessible on the cpu but the dest array is accessible on cpu

                        _Buffer_ptr pDestBuffer = _Dst;
                        return _Ev._Add_continuation(std::function<_Event()>([pTempStagingBuf, pTempStagingShape, pDestBuffer, _Dst_shape]() mutable -> _Event {
                            return _Copy_impl(pTempStagingBuf, pTempStagingShape, pDestBuffer, _Dst_shape);
                        }));
                    }
                    else
                    {
                        // Neither the source and destination are accessible on the CPU and they are on different
                        // accelerator_views

                        // Now we will create a temporary staging buffer on the dest accelerator_view
                        // and copy over the src contents to it
                        _Buffer_ptr pTempDestStagingBuf = _Buffer::_Create_stage_buffer(_Dst->_Get_accelerator_view(), accelerator(accelerator::cpu_accelerator).default_view,
                                                                                        numElementsToCopy, _Src->_Get_elem_size(), true);

                        pTempDestStagingBuf->_Map_stage_buffer(_Write_access, true);

                        _Buffer_ptr pDestBuf = _Dst;
                        _Ev = _Ev._Add_continuation(std::function<_Event()>([pTempStagingBuf, pTempDestStagingBuf, numElementsToCopy]() mutable -> _Event {
                            return _Copy_impl(pTempStagingBuf, 0, pTempDestStagingBuf, 0, numElementsToCopy);
                        }));

                        return _Ev._Add_continuation(std::function<_Event()>([pTempDestStagingBuf, pDestBuf, pTempStagingShape, _Dst_shape]() mutable -> _Event {
                            return pTempDestStagingBuf->_Copy_to_async(pDestBuf, pTempStagingShape, _Dst_shape);
                        }));
                    }
                }
            }
        }
    }

    struct _Array_copy_desc
    {
        _Array_copy_desc(
            const unsigned int _Rank,
            const unsigned int _Src_linear_offset,
            const unsigned int * _Src_extents, 
            const unsigned int * _Src_copy_offset,
            const unsigned int _Dst_linear_offset,
            const unsigned int * _Dst_extents, 
            const unsigned int * _Dst_copy_offset, 
            const unsigned int * _Copy_extents)
        {
            this->_Rank = _Rank;

            this->_Src_linear_offset = _Src_linear_offset;
            this->_Src_extents.assign( _Src_extents, _Src_extents + _Rank);
            this->_Src_copy_offset.assign( _Src_copy_offset, _Src_copy_offset + _Rank);
    
            this->_Dst_linear_offset = _Dst_linear_offset;
            this->_Dst_extents.assign( _Dst_extents, _Dst_extents + _Rank);
            this->_Dst_copy_offset.assign( _Dst_copy_offset, _Dst_copy_offset + _Rank);
    
            this->_Copy_extents.assign( _Copy_extents, _Copy_extents + _Rank);
        }

        _Array_copy_desc() {}
    
        unsigned int _Rank;

        // Shape of source
        unsigned int  _Src_linear_offset;
        std::vector<unsigned int> _Src_extents;
        std::vector<unsigned int> _Src_copy_offset;
        
        // Shape of destination
        unsigned int  _Dst_linear_offset;
        std::vector<unsigned int> _Dst_extents;
        std::vector<unsigned int> _Dst_copy_offset;

        // Shape of copy region
        std::vector<unsigned int> _Copy_extents;
    };

    // Declaration
    _AMPIMP HRESULT __cdecl _Recursive_array_copy(const _Array_copy_desc& _Desc, 
                                                  unsigned int _Native_copy_rank,
                                                  std::function<HRESULT(const _Array_copy_desc &_Reduced)> _Native_copy_func);

    // Iterator based copy function
    template<typename _InputInterator, typename _OutputIterator>
    inline _Event _Copy_impl_iter(_InputInterator _SrcFirst, _InputInterator _SrcLast, _OutputIterator _DstFirst)
    {
        std::copy(_SrcFirst, _SrcLast, _DstFirst);
        return _Event();
    }

    // Iterator based copy function
    template <typename InputIterator, typename _Value_type>
    inline _Event _Copy_impl(InputIterator _SrcFirst, _View_shape_ptr _Src_shape, _Inout_ _Buffer * _Dst, _View_shape_ptr _Dst_shape)
    {
        _ASSERTE(_Dst != NULL);
        _ASSERTE(_Src_shape != NULL);
        _ASSERTE(_Dst_shape != NULL);
        _ASSERTE(_Src_shape->_Get_rank() == _Dst_shape->_Get_rank());

        _ASSERTE(_View_shape::_Compare_extent_with_elem_size(_Src_shape->_Get_rank(), _Src_shape->_Get_view_extent(),
                                                             sizeof(_Value_type), _Dst_shape->_Get_view_extent(), _Dst->_Get_elem_size()));

        if (_Dst->_Get_host_ptr() != NULL)
        {
            // The destination buffer is accesible on the host.
            return _Copy_impl_iter(_SrcFirst, _Src_shape, stdext::make_unchecked_array_iterator(reinterpret_cast<_Value_type*>(_Dst->_Get_host_ptr())),
                                   _Create_reinterpreted_shape(_Dst_shape, _Dst->_Get_elem_size(), sizeof(_Value_type)));
        }
        else
        {
            // The dest buffer is not accesible on host. Lets create a temporary 
            // staging buffer on the destination buffer's accelerator_view
            _Buffer_ptr _PTempStagingBuf = _Buffer::_Create_stage_buffer(_Dst->_Get_accelerator_view(), accelerator(accelerator::cpu_accelerator).default_view,
                                                                         _Src_shape->_Get_view_size(), sizeof(_Value_type), true /* _Is_temp */);

            _PTempStagingBuf->_Map_stage_buffer(_Write_access, true /* _Wait */);
            _Value_type *_Dst_ptr = reinterpret_cast<_Value_type*>(_PTempStagingBuf->_Get_host_ptr());
            _Event _Ev = _Copy_impl_iter(_SrcFirst, _Src_shape, stdext::make_unchecked_array_iterator(_Dst_ptr), _Src_shape);

            // Now copy from the staging buffer to the destination buffer
            _Buffer_ptr _PDestBuf = _Dst;
            _Ev = _Ev._Add_continuation(std::function<_Event()>([_PTempStagingBuf, _Src_shape, _PDestBuf, _Dst_shape]() mutable -> _Event {
                return _Copy_impl(_PTempStagingBuf, _Src_shape, _PDestBuf, _Dst_shape);
            }));

            return _Ev;
        }
    }

    template <typename OutputIterator, typename _Value_type>
    inline _Event _Copy_impl(_In_ _Buffer *_Src, _View_shape_ptr _Src_shape, OutputIterator _DestIter, _View_shape_ptr _Dst_shape)
    {
        _ASSERTE(_Src != NULL);
        _ASSERTE(_Src_shape != NULL);
        _ASSERTE(_Dst_shape != NULL);
        _ASSERTE(_Src_shape->_Get_rank() == _Dst_shape->_Get_rank());

        _ASSERTE(_View_shape::_Compare_extent_with_elem_size(_Src_shape->_Get_rank(), _Src_shape->_Get_view_extent(),
                                                             _Src->_Get_elem_size(), _Dst_shape->_Get_view_extent(), sizeof(_Value_type)));

        if (_Src->_Get_host_ptr() != NULL)
        {
            // The source buffer is accessible on the host.
            return _Copy_impl_iter(reinterpret_cast<_Value_type*>(_Src->_Get_host_ptr()),
                                   _Create_reinterpreted_shape(_Src_shape, _Src->_Get_elem_size(), sizeof(_Value_type)),
                                   _DestIter, _Dst_shape);
        }
        else
        {
            // The source buffer is not accessible on host. Lets create a temporary 
            // staging buffer on the source buffer's accelerator_view and initiate a copy
            // from the source buffer to the temporary staging buffer
            _Buffer_ptr _PTempStagingBuf = _Buffer::_Create_stage_buffer(_Src->_Get_accelerator_view(), accelerator(accelerator::cpu_accelerator).default_view,
                                                                         _Dst_shape->_Get_view_size(), sizeof(_Value_type), true);

            _Event _Ev = _Src->_Copy_to_async(_PTempStagingBuf, _Src_shape, _Dst_shape);
            return _Ev._Add_continuation(std::function<_Event()>([_PTempStagingBuf, _Dst_shape, _DestIter]() mutable -> _Event {
                return _Copy_impl_iter(reinterpret_cast<_Value_type*>(_PTempStagingBuf->_Get_host_ptr()),
                                       _Dst_shape, _DestIter, _Dst_shape);
            }));
        }
    }

    // Iterator based structured copy function
    template<typename _InputInterator, typename _OutputIterator>
    inline _Event _Copy_impl_iter(_InputInterator _SrcIter, _View_shape_ptr _Src_shape,
                                  _OutputIterator _DstIter, _View_shape_ptr _Dst_shape)
    {
        _ASSERTE(_Src_shape->_Get_rank() == _Dst_shape->_Get_rank());
        _ASSERTE(_View_shape::_Compare_extent(_Src_shape->_Get_rank(), _Src_shape->_Get_view_extent(), _Dst_shape->_Get_view_extent()));

        // If both the _Src_shape and _Dst_shape are linear we can be more efficient
        unsigned int _Src_linear_offset, _Src_linear_size, _Dst_linear_offset, _Dst_linear_size;
        if (_Src_shape->_Is_view_linear(_Src_linear_offset, _Src_linear_size) &&
            _Dst_shape->_Is_view_linear(_Dst_linear_offset, _Dst_linear_size))
        {
            _ASSERTE(_Src_linear_size == _Dst_linear_size);

            // These iterators might be not contiguous, therefore we use std::advance
            std::advance(_SrcIter, _Src_linear_offset);
            auto _SrcLast = _SrcIter;
            std::advance(_SrcLast, _Src_linear_size);
            std::advance(_DstIter, _Dst_linear_offset);

            return _Copy_impl_iter(_SrcIter, _SrcLast, _DstIter);
        }

        std::vector<unsigned int> _Src_extent(_Src_shape->_Get_rank());
        std::vector<unsigned int> _Src_offset(_Src_shape->_Get_rank());
        std::vector<unsigned int> _Dst_extent(_Dst_shape->_Get_rank());
        std::vector<unsigned int> _Dst_offset(_Dst_shape->_Get_rank());
        std::vector<unsigned int> _Copy_extent(_Src_shape->_Get_rank());

        for (size_t i = 0; i < _Src_shape->_Get_rank(); ++i) {
            _Src_extent[i] = _Src_shape->_Get_base_extent()[i];
            _Src_offset[i] = _Src_shape->_Get_view_offset()[i];
            _Dst_extent[i] = _Dst_shape->_Get_base_extent()[i];
            _Dst_offset[i] = _Dst_shape->_Get_view_offset()[i];
            _Copy_extent[i] = _Src_shape->_Get_view_extent()[i];
        }

        _Array_copy_desc _Desc(
            _Src_shape->_Get_rank(),
            _Src_shape->_Get_linear_offset(),
            _Src_extent.data(),
            _Src_offset.data(),
            _Dst_shape->_Get_linear_offset(),
            _Dst_extent.data(),
            _Dst_offset.data(),
            _Copy_extent.data());

        // Note: Capturing shape pointers would be incorrect, they are valid for setting up the call.
        // They might be deleted right after this call completes. 
        HRESULT hr = _Recursive_array_copy(_Desc, 1, [_SrcIter, _DstIter](const _Array_copy_desc &_Reduced) -> HRESULT {

            auto _SrcFirst = _SrcIter;
            auto _DstFirst = _DstIter;
    
            std::advance(_DstFirst, _Reduced._Dst_linear_offset + _Reduced._Dst_copy_offset[0]);
            std::advance(_SrcFirst, _Reduced._Src_linear_offset + _Reduced._Src_copy_offset[0]);
            auto _SrcLast = _SrcFirst;
            std::advance(_SrcLast, _Reduced._Copy_extents[0]);

            std::copy(_SrcFirst, _SrcLast, _DstFirst);

            return S_OK;
        });

        if (FAILED(hr)) {
            throw Concurrency::runtime_exception("Failed to copy between buffers", E_FAIL);
        }

        return _Event();
    }

    // Enumeration of states with regards to a pending (in progress) read/commit operation
    // This is used to prevent redundant reads/commits of data when a read/commit of the
    // data has already been triggered by a previous _Get_access_async operation and is 
    // currently in flight.
    enum _Pending_operation_status { _Pending_read, _Pending_commit, _Pending_none };

    // This type captures information corresponding to a distinct view of data
    // currently active atop a parent _Ubiquitous_buffer
    struct _View_info
    {
        // This field tracks the current access mode of this view
        // Note that it is different from the cpu access mode indicated
        // by the view key which only tells the access mode of the cpu av
        // specified at the time of registration of the view. This field 
        // on the other hand is more general and tells the current access
        // mode of this view for any acclerator view
        _Access_mode _M_curr_view_access_mode;

        // A set of all live views (keys) that correspond to this View info
        // This essentially is the set of array_views that are shallow 
        // copies of the same view and hence share the same view metadata
        // and is used to update the CPU access mode whenever the access
        // mode of the view is updated by a _Get_access_async operation
        std::unordered_set<_View_key> _M_live_view_keys;

        // The CPU accelerator view that the views corresponding to this
        // view info were registered with. This is used to determine
        // which cpu accelerator_view do the live keys correspond to, so 
        // that when access is granted on this accelerator_view, the live keys
        // are properly updated
        _Accelerator_view_impl_ptr _M_registered_cpu_av;

        // The shape of the view
        _View_shape_ptr _M_shape;

        // Indicates the accelerator_view which has the most recent copy
        // of the data corresponding to this view
        _Accelerator_view_impl_ptr _M_av_with_exclusive_copy;

        // A set of accelerator_views where valid copies of the data 
        // corresponding to this view are available
        std::set<_Accelerator_view_impl_ptr> _M_avs_with_shared_copies;

        // Indicates if the the contents of the view been dicarded. This is 
        // used to avoid copying of data when the user has indicated the 
        // current contents of the view to be garbage by "discarding"
        bool _M_is_content_discarded;

        // Are there any pending operations in progress on this view
        // This is used to prevent data corresponding to this view being
        // read/commit multiple times when the read/commit operation has already 
        // been trigerred by a _Get_access_async call and another request 
        // comes in before the previous request is finished. For example 
        // 2 threads concurrently calling "synchronize()" on a view.
        _Pending_operation_status _M_pending_operation_status;

        // An event corresponding to an async commit for this view already in progress 
        _Event _M_pending_commit_event;

        // A map of pending reads corresponding this view that are already in progress
        // Note that since multiple reads with different accelerator_view targets may 
        // be concurrently in flight, we maintain all such in-flight reads in this map
        // with the target accelerator_view as key with the corresponding async operation's 
        // event as the value.
        std::map<_Accelerator_view_impl_ptr, _Event> _M_pending_read_targets;

        _View_info(_Accelerator_view_impl_ptr _Registered_cpu_av, _View_shape_ptr _Shape) 
            : _M_registered_cpu_av(_Registered_cpu_av), _M_shape(_Shape),
            _M_curr_view_access_mode(_No_access), _M_av_with_exclusive_copy(NULL),
            _M_is_content_discarded(false), _M_pending_operation_status(_Pending_none)
        {
            _ASSERTE(_M_registered_cpu_av != NULL);
            _ASSERTE(_Shape != NULL);
        }

        void _Add_view_key(_In_ _View_key _Key)
        {
            bool inserted;
            inserted = _M_live_view_keys.insert(_Key).second;
            _ASSERTE(inserted);
        }

        size_t _Remove_view_key(_In_ _View_key _Key)
        {
            size_t numErased;
            numErased = _M_live_view_keys.erase(_Key);
            _ASSERTE(numErased != 0);

            return _M_live_view_keys.size();
        }

        // Updates the CPU access mode of all live views currently registered with this 
        // view. It also updates the _M_data_ptr field (cached host pointer) of the registered
        // views when the parameter _Host_ptr is non-null
        void _Set_live_keys_access_mode(_Access_mode _Mode, _In_ void *_Host_ptr = NULL)
        {
            for (std::unordered_set<_View_key>::iterator iter = _M_live_view_keys.begin(); iter != _M_live_view_keys.end(); ++iter) 
            {
                // If this is an upgrade of the cpu access mode, also update the host ptr
                if ((_Mode != _No_access) && (((*iter)->_M_curr_cpu_access_mode & _Mode) != _Mode) && (_Host_ptr != NULL))
                {
                    (*iter)->_M_data_ptr = _Host_ptr;
                }

                (*iter)->_M_curr_cpu_access_mode = _Mode;
            }
        }

        // Does this view have a valid copy of data on the specified 
        // accelerator_view
        bool _Has_copy_on_av(_Accelerator_view_impl_ptr _Av) const
        {
            return (_M_av_with_exclusive_copy == _Av) ||
                   (_M_avs_with_shared_copies.find(_Av) != _M_avs_with_shared_copies.end());
        }

        // Gets the CPU access mode corresponding to this view.
        _Access_mode _Cpu_access_mode() const
        {
            if (_M_live_view_keys.empty()) {
                return _No_access;
            }
            else {
                // Since the CPU access modes of all live views registered for
                // this data view are kept in sync, we can just return the access 
                // mode of any one of them
                return ((*(_M_live_view_keys.begin()))->_M_curr_cpu_access_mode);
            }
        }

    private:

        // No default constructor, copy constructor and assignment operator
        _View_info();
        _View_info(const _View_info& _Other);
        _View_info &operator=(const _View_info& _Other);
    };

    // A ubiquitous buffer that provides access to the underlying data 
    // on any accelerator_view
    class _Ubiquitous_buffer : public _Reference_counter
    {
        friend _Event _Get_access_async(const _View_key _Key, accelerator_view _Av, _Access_mode _Mode, _Buffer_ptr &_Buf_ptr);

    public:

        _AMPIMP static _Ret_ _Ubiquitous_buffer * __cdecl _Create_ubiquitous_buffer(_Buffer_ptr _Master_buffer);

        _AMPIMP void _Release();

        // Register a new view on top of this _Ubiquitous_buffer
        _AMPIMP void _Register_view(_In_ _View_key _Key, accelerator_view _Cpu_av, _View_shape_ptr _Shape);

        // Register a copy of an existing view registered with this _Ubiquitous_buffer
        _AMPIMP void _Register_view_copy(_In_ _View_key _New_view_key, _In_ _View_key _Existing_view_key);

        // Unregister a view currently registered with this _Ubiquitous_buffer
        _AMPIMP void _Unregister_view(_In_ _View_key _Key);

        // Obtain a specified mode of access to the specified view on the specified target
        // accelerator_view. This method also serves the purpose of determining the 
        // amount of data copy expected to happen as part of this _Get_access request
        // without actually performing the copies or state updates in the _Ubiquitous_buffer. This
        // is used for reporting the implicit data copies that happen when accessing array_views
        // in C++ AMP ETW events
        _AMPIMP _Event _Get_access_async(_In_ _View_key _Key, _Accelerator_view_impl_ptr _Av_view_impl_ptr,
                                         _Access_mode _Mode, _Buffer_ptr &_Buf_ptr,
                                         _Inout_opt_ ULONGLONG *_Sync_size = nullptr);

        // Discard the content underlying this view
        _AMPIMP void _Discard(_In_ _View_key _Key);

        // This method does not synchonize the copies. Should not be used for getting 
        // data access but only to get the underlying buffer's properties
        _AMPIMP _Buffer_ptr _Get_master_buffer() const;

        _AMPIMP accelerator_view _Get_master_accelerator_view() const;

        _AMPIMP _View_shape_ptr _Get_view_shape(_In_ _View_key _Key);

        _Accelerator_view_impl_ptr _Get_master_accelerator_view_impl() const
        {
            return _M_master_av;
        }

    private:

        // The _Ubiquitous_buffer constructor is private to force construction through the static 
        // _Create_ubiquitous_buffer method to ensure the object is allocated in the runtime
        _Ubiquitous_buffer(_Buffer_ptr _Master_buffer);

        // Private destructor to force deletion through _Release
        ~_Ubiquitous_buffer();

        // No default consturctor, copy constructor and assignment operator
        _Ubiquitous_buffer();
        _Ubiquitous_buffer(const _Ubiquitous_buffer &rhs);
        _Ubiquitous_buffer &operator=(const _Ubiquitous_buffer &rhs);
        
        // Helper methods

        // Get access to a buffer on a specified accelerator for a specified pre-registered view.
        // If _Sync_size parameter is not null, then function calculates number of bytes that we
        // need to synchronize to get desired access.
        _AMPIMP _Event _Get_access_async(_In_ _View_key _Key, accelerator_view _Av, _Access_mode _Mode,
                                         _Buffer_ptr &_Buf_ptr, _Inout_opt_ ULONGLONG *_Sync_size = NULL);

        // Commit a view to the master buffer if needed. When the _Sync_size parameter is non-null
        // this method just returns the amount of data to be copied as part of the commit, without 
        // actually performing the commit
        _Event _Commit_view_async(_In_ _View_info *_Info, _Inout_ ULONGLONG *_Sync_size = nullptr);

        // Get the _Buffer_ptr corresponding to a specified accelerator_view. When the 
        // _Create parameter is true, it creates a new _Buffer if one does not already exist
        // for that accelerator_view
        _Buffer_ptr _Get_buffer(_Accelerator_view_impl_ptr _Av, bool _Create = true);

        // Sets a new access mode for the specified view
        void _Set_new_access_mode(_Inout_ _View_info *_Info, _Access_mode _New_mode);

        // Unsets the discard flag from the specified view and all other 
        // overlapping views
        void _Unset_discard_flag(_Inout_ _View_info *_Info);

        // Determines whether the data underlying the specified view has been discarded
        // based on whether a subsuming view has the discard flag set.
        bool _Should_discard(const _View_info *_Info) const;

        // Does this view have exclusive data which is not discarded, 
        // not on the master accelerator_view and also there is not other view 
        // that subsumes this view and is marked dirty
        bool _Has_exclusive_data(const _View_info *_Info) const;

        // Based on the current state of overlapping views in the _Ubiquitous_buffer
        // does the specified view require a data update on the target accelerator_view
        // to fulfil an access request
        bool _Requires_update_on_target_accelerator_view(const _View_info *_Info,
                                                         _Access_mode _Requested_mode,
                                                         _Accelerator_view_impl_ptr _Target_acclerator_view) const;

        // This method iterates over all views in the specified commit list
        // and flags them as "commit not needed" if that view is subsumed by another view present in the
        // commit list
        static void _Flag_redundant_commits(std::vector<std::pair<_View_info*, bool>> &_Commit_list);

    private:

        // Private data

        // The master accelerator_view for this _Ubiquitous_buffer
        // which is specified at construction time
        _Accelerator_view_impl_ptr _M_master_av;

        // The master _Buffer corresponding to this _Ubiquitous_buffer
        // which is specified at construction time
        _Buffer_ptr _M_master_buffer;

        // A map of pre-created _Buffers corresponding to different 
        // accelerator_views where the _Ubiquitous_buffer has already been
        // accessed
        std::map<_Accelerator_view_impl_ptr, _Buffer_ptr> _M_buffer_map;
        
        // A mapping between all registered view keys in this _Ubiquitous_buffer
        // to their corresponding _View_info
        std::unordered_map<_View_key, _View_info*> _M_view_map;

        // Set of distinct views of this buffer. As multiple copies of the same
        // view may have been registered for this _Ubiquitous_buffer, this set 
        // maintains the set of distinct views which really matter for the 
        // caching protocol. Also, note that some view_info may not have any live registered 
        // and hence does not exist in the _M_view_map but may exist here since
        // it has uncomiitted data which needs to be considered as part of the cache
        // coherence protocol to prevent modifications underlying this view from being lost
        std::unordered_set<_View_info*> _M_view_info_set;

        // Critical section object to protect the cache directory
        Concurrency::critical_section _M_critical_section;

    public:
        // Note that this function must be implemented in the header to prevent the STL unordered_set
        // return object from passing across the DLL boundary which causes issues when the application
        // uses a different CRT than the C++ AMP runtime.

        // This method returns the list of accelerator_views where the specified view already has
        // a valid cached copy of the data and getting read access  would not incur any data movement.
        // The _Can_access_anywhere parameter is an output parameter used to indicate to the 
        // caller that the specified view can be accessed on any accelerator_view without incurring
        // any data movement. This is true when there are no modified overlapping views that require 
        // synchronization and the specified view has the discard_data flag set.
        // This method is used for determining the source accelerator_view for copy and p_f_e operations
        // involving array_views
        inline _Accelerator_view_unordered_set _Get_caching_info(_In_ _View_key _Key, _Out_ bool *_Can_access_anywhere = NULL)
        {
            _Accelerator_view_unordered_set _ResultAcclViews;
            if (_Can_access_anywhere != NULL) {
                *_Can_access_anywhere = false;
            }

            if (_Key != NULL)
            {
                Concurrency::critical_section::scoped_lock _Lock(_M_critical_section);

                auto _Iter = _M_view_map.find(_Key);
                if (_Iter != _M_view_map.end()) 
                {
                    _View_info *_PViewInfo = _Iter->second;
                    _View_shape_ptr _PShape = _PViewInfo->_M_shape;

                    // Read access was requested
                    bool _OverlappingDirtyViewOnNonMasterAv = false;
                    for (auto _Iter1 = _M_view_info_set.begin(); _Iter1 != _M_view_info_set.end(); ++_Iter1)
                    {
                        // Skip the ones that have no access or do not overlap with the view we are trying to get access to
                        if (((*_Iter1)->_M_curr_view_access_mode == _No_access) || !(*_Iter1)->_M_shape->_Overlaps(_PShape)) {
                            continue;
                        }

                        // This view has read access. All its locations with valid copies are good candidates if the
                        // view subsumes the view we are getting info for.
                        if ((*_Iter1)->_M_shape->_Subsumes(_PShape)) 
                        {
                            if ((*_Iter1)->_M_av_with_exclusive_copy != NULL) 
                            {
                                _ResultAcclViews.insert(accelerator_view((*_Iter1)->_M_av_with_exclusive_copy));
                            }

                            for (auto _TempIter = (*_Iter1)->_M_avs_with_shared_copies.begin(); _TempIter != (*_Iter1)->_M_avs_with_shared_copies.end(); ++_TempIter)
                            {
                                _ResultAcclViews.insert(accelerator_view(*_TempIter));
                            }
                        }

                        if (((*_Iter1)->_M_curr_view_access_mode & _Write_access) &&
                            !(*_Iter1)->_M_is_content_discarded &&
                            ((*_Iter1)->_M_av_with_exclusive_copy != _M_master_av))
                        {
                            _OverlappingDirtyViewOnNonMasterAv = true;
                        }
                    }

                    // If there are no dirty views on non-master locations, the master accelerator_view
                    // has valid data for the queried view
                    if (!_OverlappingDirtyViewOnNonMasterAv)
                    {
                        _ResultAcclViews.insert(this->_Get_master_accelerator_view());

                        // Also, if the contents of the query view are discarded, then every accelerator_view where a buffer
                        // has been created is a good candidate for getting read access to without requiring any buffer allocation
                        // or data movement
                        if (_PViewInfo->_M_is_content_discarded)
                        {
                            // This view can be accessed on any accelerator_view without incurring any data movement.
                            if (_Can_access_anywhere != NULL) {
                                *_Can_access_anywhere = true;
                            }

                            for (auto _Iter = _M_buffer_map.begin(); _Iter != _M_buffer_map.end(); ++_Iter) {
                                _ResultAcclViews.insert(accelerator_view(_Iter->first));
                            }
                        }
                    }
                }
            }

            return _ResultAcclViews;
        }
    };

    // This method peeks into the current caching state of the specified src view to determine the best source
    // accelerator_view choice for copying from the src view to a specified destination accelerator_view. 
    // This method is used for determining the optimal source location for an array_view in a copy operation
    static inline accelerator_view _Select_copy_src_accelerator_view(_View_key _Src_view_key, const accelerator_view &_Dest_accelerator_view)
    {
        // Get the set of accelerator_views with the required access for the view
        auto _Candidate_src_accelerator_views = _Src_view_key->_Get_buffer_ptr()->_Get_caching_info(_Src_view_key);
        if (_Candidate_src_accelerator_views.empty()) 
        {
            // If the view is not cached anywhere we will use the view's master location
            // as the copy location
            return _Src_view_key->_Get_buffer_ptr()->_Get_master_accelerator_view();
        }
        else if (_Candidate_src_accelerator_views.size() == 1)
        {
            // If there is just one candidate we will use that one
            return (*(_Candidate_src_accelerator_views.begin()));
        }
        else
        {
            // We have multiple candidates. Our first choice would be the destination accelerator_view
            // itself since the copying would be fastest across buffers on the same accelerator_view (including CPU)
            auto _Iter = _Candidate_src_accelerator_views.find(_Dest_accelerator_view);
            if (_Iter != _Candidate_src_accelerator_views.end()) 
            {
                return _Dest_accelerator_view;
            }
            else
            {
                // Our next choice would be an accelerator_view on the CPU accelerator since
                // copying from CPU to accelerator_view memory is faster compared to copy
                // across different accelerator_views
                auto _Iter = std::find_if(_Candidate_src_accelerator_views.begin(), _Candidate_src_accelerator_views.end(), [](const accelerator_view &_Curr_accl_view) {
                    return (_Curr_accl_view.accelerator.device_path == accelerator::cpu_accelerator);
                });

                if (_Iter != _Candidate_src_accelerator_views.end())
                {
                    return *_Iter;
                }
                else
                {
                    // We do not have a valid cached copy on the destination accelerator_view or the CPU
                    // We will now prefer an accelerator_view from an accelerator different than the
                    // destination accelerator_view's accelerator. This is to avoid requiring both the source
                    // and destination buffers to be resident in the same accelerator's memory and instead 
                    // use the memory from different accelerators
                    auto _Iter = std::find_if(_Candidate_src_accelerator_views.begin(), _Candidate_src_accelerator_views.end(),
                                              [&_Dest_accelerator_view](const accelerator_view &_Curr_accl_view) 
                    {
                        return (_Curr_accl_view.accelerator != _Dest_accelerator_view.accelerator);
                    });

                    if (_Iter != _Candidate_src_accelerator_views.end())
                    {
                        return *_Iter;
                    }
                    else
                    {
                        // Lets just pick any accelerator_view from the candidate list
                        return (*(_Candidate_src_accelerator_views.begin()));
                    }
                }
            }
        }
    }

    // Class defines functions for interoperability with D3D
    class _D3D_interop
    {
    public:
        _AMPIMP static _Ret_ IUnknown * __cdecl _Get_D3D_buffer(_In_ _Buffer *_Buffer_ptr);
        _AMPIMP static _Ret_ IUnknown * __cdecl _Get_D3D_texture(_In_ _Texture *_Texture_ptr);
    };

    inline
    _Event _Get_access_async(const _View_key _Key, accelerator_view _Av, _Access_mode _Mode, _Buffer_ptr &_Buf_ptr)
    {
        return _Key->_Get_buffer_ptr()->_Get_access_async(_Key->_Get_view_key(), _Av, _Mode, _Buf_ptr);
    }

    inline
    _View_shape_ptr _Get_buffer_view_shape(const _Buffer_descriptor& _Descriptor)
    {
        return _Descriptor._Get_buffer_ptr()->_Get_view_shape(_Descriptor._Get_view_key());
    }

} // namespace Concurrency::details

/// <summary>
/// Exception thrown when an underlying DirectX call fails
/// due to the Windows timeout detection and recovery mechanism
/// </summary>
class accelerator_view_removed : public runtime_exception
{
public:
    /// <summary>
    ///     Construct an accelerator_view_removed exception with a message and
    ///     a view removed reason code
    /// </summary>
    /// <param name="_Message">
    ///     Descriptive message of error
    /// </param>
    /// <param name="_View_removed_reason">
    ///     HRESULT error code indicating the cause of removal of the accelerator_view
    /// </param>
    _AMPIMP explicit accelerator_view_removed(const char * _Message, HRESULT _View_removed_reason) throw();

    /// <summary>
    ///     Construct an accelerator_view_removed exception
    /// </summary>
    /// <param name="_View_removed_reason">
    ///     HRESULT error code indicating the cause of removal of the accelerator_view
    /// </param>
    _AMPIMP explicit accelerator_view_removed(HRESULT _View_removed_reason) throw();

    /// <summary>
    ///     Returns an HRESULT error code indicating the cause of the accelerator_view's removal
    /// </summary>
    /// <returns>
    ///     The HRESULT error code that indicates the cause of accelerator_view's removal
    /// </returns>
    _AMPIMP HRESULT get_view_removed_reason() const throw();

private:

    HRESULT _M_view_removed_reason_code;
}; // class accelerator_view_removed

/// <summary>
///     Exception thrown when the runtime fails to launch a kernel
///     using the compute domain specified at the parallel_for_each call site.
/// </summary>
class invalid_compute_domain : public runtime_exception
{
public:
    /// <summary>
    ///     Construct an invalid_compute_domain exception with a message
    /// </summary>
    /// <param name="_Message">
    ///     Descriptive message of error
    /// </param>
    _AMPIMP explicit invalid_compute_domain(const char * _Message) throw();

    /// <summary>
    ///     Construct an invalid_compute_domain exception
    /// </summary>
    _AMPIMP invalid_compute_domain() throw();
}; // class invalid_compute_domain

/// <summary>
///     Exception thrown when an unsupported feature is used
/// </summary>
class unsupported_feature  : public runtime_exception
{
public:
    /// <summary>
    ///     Construct an unsupported_feature exception with a message
    /// </summary>
    /// <param name="_Message">
    ///     Descriptive message of error
    /// </param>
    _AMPIMP explicit unsupported_feature(const char * _Message) throw();

    /// <summary>
    ///     Construct an unsupported_feature exception
    /// </summary>
    _AMPIMP unsupported_feature() throw();
}; // class unsupported_feature

} // namespace Concurrency

// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//
// Compiler/Runtime Interface
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#define HELPERAPI __cdecl

using namespace Concurrency::details;

extern "C" {

    // This structure is used for storing information about the const buffer array created
    // on the host for passing to the kernel and also for creating temporary buffers used
    // for passing non-primitive type scalars into the kernel functions
    struct _Device_buffer_info
    {
        bool  _M_is_buffer;  // buffer or texture

        void * _M_desc;        // Pointer to the _Buffer_descriptor/_Texture_descriptor instance which underlies all
                               // device storage

        _Access_mode _M_formal_access_mode;         // scalar: read-only
                                                    // const scalar ref: read-only
                                                    // scalar ref: ReadWrite
                                                    // array: ReadWrite
                                                    // const array: ReadOnly
        size_t _M_actual_arg_num;

        _Ret_ _Buffer_descriptor * _Get_buffer_desc() const
        {
            _ASSERTE(_M_is_buffer);
            return reinterpret_cast<_Buffer_descriptor *>(_M_desc);
        }

        _Ret_ _Texture_descriptor * _Get_texture_desc() const
        {
            _ASSERTE(!_M_is_buffer);
            return reinterpret_cast<_Texture_descriptor *>(_M_desc);
        }

        _Ret_ void * _Get_resource_ptr() const
        {
            if (_M_is_buffer) 
            {
                _Ubiquitous_buffer * _Tmp = _Get_buffer_desc()->_Get_buffer_ptr();
                return reinterpret_cast<void *>(_Tmp);
            }
            else
            {
                _Texture * _Tmp = _Get_texture_desc()->_Get_texture_ptr();
                return reinterpret_cast<void *>(_Tmp);
            }
        }
    };

    // This structure is used for storing information about the const buffers
    struct _Device_const_buffer_info
    {
        void * _M_data;                             // Pointer to the host data to intialize the
                                                    // constant buffer with

        size_t _M_const_buf_size;                   // Size of the const buffer in bytes

        unsigned int _M_is_debug_data;              // Is this debug data which will be 
                                                    // intialized by the runtime. 0 (false), 1 (true)
    };

    // This structure is used for passing the scheduling
    // info to the parallel_for_each which is handed back
    // to the compiler-runtime interface methods by the front-end
    struct _Host_Scheduling_info
    {
        // The accelerator view to invoke a parallel_for_each on
        _Accelerator_view_impl *_M_accelerator_view; 
    };
}

namespace Concurrency
{
// These consts MUST match the corresponding definitions in the back-end
//
// The maximum # of buffers would be 64 (UAV) + 128 (SRV) for DX11/11.1
static const UINT HLSL_MAX_NUM_BUFFERS = 64 + 128;
static const UINT MODULENAME_MAX_LENGTH = 1024;
namespace details
{
    // Important: runtime and backend compiler have to agree on the layout

    // Format:
    //    UINT  version  // the version number for the shader descriptor
    //    GUID  guid
    //    UINT  aliased?
    //    UINT  size // byte code size
    //    UINT  caps // record the caps of the shader
    //    UINT  numBuffers // number of buffers, runtime has this information, we keep it just for completeness and allow double checking in runtime
    //    UINT  bufferRW[numBuffers] // record the RW-ness of buffers
    //    UINT  instanceSlots[numBuffers] // record slot info for each instance. The total # used is the same as # of buffers used
    typedef struct _DPC_shader_blob
    {
        static const unsigned int CAPS_USE_DOUBLE = 0x1;
        static const unsigned int CAPS_USE_DOUBLE_EXT = 0x2;
        static const unsigned int CAPS_USE_DEBUGGING_INTRINSICS = 0x4;
        static const unsigned int CAPS_USE_EXTENDED_UAVS = 0x8;

        unsigned int _M_version;
        GUID _M_guid;
        unsigned int _M_aliased;
        unsigned int _M_bytecode_length;
        unsigned int _M_caps;
        unsigned int _M_num_buffers;

        static const unsigned int INVALID_SLOT = 0xFFFFFFFF;

        void GetGUIDIndentifierString(LPWSTR guidString, int size)
        {
            UINT length = StringFromGUID2(_M_guid, guidString, size);

            for (UINT _I = 0; _I < length; _I ++) {
                if ((guidString[_I] == L'{') || (guidString[_I] == L'}') || (guidString[_I] == L'-')) { 
                        guidString[_I] = L'_';
                }
            }
        }

        _Ret_ unsigned int * GetBufferRwPropertyTable() {
            return reinterpret_cast<unsigned int *>((reinterpret_cast<char *>(this) + sizeof(_DPC_shader_blob)));
        }

        _Ret_ unsigned int * GetInstanceSlotsTable() {
            return GetBufferRwPropertyTable() + _M_num_buffers;
        }

        _Ret_ char * GetByteCode(bool aliased) {
            if (aliased) {
                _ASSERTE(_M_aliased);
                return reinterpret_cast<char *>(GetInstanceSlotsTable() + _M_num_buffers);
            } else {
                _ASSERTE(!_M_aliased);
                return reinterpret_cast<char *>(GetBufferRwPropertyTable() + _M_num_buffers);
            }
        }

    } _DPC_shader_blob;

    enum _DPC_kernel_func_kind
    {
        NON_ALIASED_SHADER  = 0, // slot 0
        ALIASED_SHADER      = 1,  // slot 1
        NUM_SHADER_VERSIONS = 2
    };

    struct _DPC_call_handle
    {
        _Accelerator_view_impl *_M_rv;
        bool _M_is_explicit_target_acclview;

        // Info about the kernel function arguments
        _Device_buffer_info * _M_device_buffer_info;
        size_t _M_num_buffers;
        size_t _M_num_writable_buffers;

        // Info about the host buffer created corresponding to the const buffer
        _Device_const_buffer_info * _M_const_buffer_info;
        size_t _M_num_const_buffers;

        // Info about read-write aliasing 
        std::vector<int> _M_Redirect_indices;
        bool _M_RW_aliasing;
        std::unordered_set<void*> _M_aliased_buffer_set;

        // Kernel funcs
        _DPC_shader_blob * _M_shader_blobs[NUM_SHADER_VERSIONS];

        // Compute domain info
        int _M_is_flat_model;
        unsigned int _M_compute_rank;
        unsigned int * _M_grid_extents;

        // Kernel dispatch info
        unsigned int _M_groupCountX;
        unsigned int _M_groupCountY;
        unsigned int _M_groupCountZ;

        // The shape of the group
        unsigned int _M_groupExtentX;
        unsigned int _M_groupExtentY;
        unsigned int _M_groupExtentZ;

        _DPC_call_handle(_In_ _Accelerator_view_impl *_Accelerator_view)
        {
            _M_rv = _Accelerator_view;

            _M_is_explicit_target_acclview = false;
            if (_M_rv != NULL) {
                _M_is_explicit_target_acclview = true;
            }

            _M_device_buffer_info = NULL;        
            _M_num_buffers = 0;
            _M_num_writable_buffers = 0;

            _M_const_buffer_info = NULL;
            _M_num_const_buffers = 0;

            _M_RW_aliasing = false;

            for (size_t _I = 0; _I < NUM_SHADER_VERSIONS; _I++) 
            {
                _M_shader_blobs[_I] = NULL;
            }

            _M_is_flat_model = 0;
            _M_compute_rank = 0;
            _M_grid_extents = NULL;

            _M_groupCountX = 0;
            _M_groupCountY = 0;
            _M_groupCountZ = 0;

            _M_groupExtentX = 0;
            _M_groupExtentY = 0;
            _M_groupExtentZ = 0;            
        }

        ~_DPC_call_handle()
        {
            if (_M_grid_extents) {
                delete [] _M_grid_extents;
            }
        }

        bool _Is_buffer_aliased(_In_ void *_Buffer_ptr)
        {
            return (_M_aliased_buffer_set.find(_Buffer_ptr) != _M_aliased_buffer_set.end());
        }

        void _Check_buffer_aliasing();
        void _Update_buffer_rw_property();
        void _Setup_aliasing_redirection_indices();
        void _Select_accelerator_view();
        void _Verify_buffers_against_accelerator_view();
    };

} // namespace Concurrency::details
} // namespace Concurrency

extern "C" {

    // Return a compiler helper handle.
    _AMPIMP _Ret_ _DPC_call_handle * HELPERAPI __dpc_create_call_handle(_In_ _Host_Scheduling_info *_Sch_info) throw(...);

    // Destroy the call handle
    _AMPIMP void HELPERAPI __dpc_release_call_handle(_In_ _DPC_call_handle * _Handle) throw(...);

    _AMPIMP void HELPERAPI __dpc_set_device_buffer_info(_In_ _DPC_call_handle * _Handle, _In_ _Device_buffer_info * _DeviceBufferInfo, size_t _NumBuffers) throw(...);

    // Set const buffer info.
    _AMPIMP void HELPERAPI __dpc_set_const_buffer_info(_In_ _DPC_call_handle * _Handle, _In_ _Device_const_buffer_info * _DeviceConstBufferInfo, size_t _NumConstBuffers) throw(...);

    // Set kernel dispatch info
    _AMPIMP void HELPERAPI __dpc_set_kernel_dispatch_info(_In_ _DPC_call_handle * _Handle,
                                                         unsigned int _ComputeRank,
                                                         _In_ int * _Extents,
                                                         unsigned int _GroupRank,
                                                         const unsigned int * _GroupExtents,
                                                         unsigned int & _GroupCountX,
                                                         unsigned int & _GroupCountY,
                                                         unsigned int & _GroupCountZ) throw(...);

    // Dispatch the kernel
    _AMPIMP void HELPERAPI __dpc_dispatch_kernel(_In_ _DPC_call_handle * _Handle, _Inout_ void ** _ShaderBlobs) throw(...);
    
#ifdef _DEBUG
    // Dispatch the kernel passed as a HLSL source level shader 
    // This function is to be used only for testing and debugging purposes
    _AMPIMP void HELPERAPI __dpc_dispatch_kernel_test(_In_ _DPC_call_handle * _Handle, _In_ WCHAR* szFileName, LPCSTR szEntryPoint) throw(...);
#endif 
}

// =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//
// C++ AMP ETW Provider
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

namespace Concurrency
{
namespace details
{

// Thread-safe factory method for _Amp_runtime_trace object
_AMPIMP _Ret_ _Amp_runtime_trace* __cdecl _Get_amp_trace();

// Abstract base class for provider functionality
class _ITrace
{
public:
    virtual ~_ITrace() {};
    
    // Registers provider in system, callback function is called when new session is enabled/disabled
    virtual DWORD _Register_provider(LPCGUID _Provider_id, PENABLECALLBACK _Callback_function, PVOID _Callback_context) = 0;
    
    // Write event 
    virtual DWORD _Write_event(PCEVENT_DESCRIPTOR _Event_descriptor, PEVENT_DATA_DESCRIPTOR _Event_data, ULONG _Event_data_count) = 0;

    // Checks if provider is enabled for a given event
    virtual BOOL _Is_enabled(PCEVENT_DESCRIPTOR _Event_descriptor) = 0;
};

// Class that gathers C++ AMP diagnostic information and triggers events
class _Amp_runtime_trace
{

// Called by factory to create single instance of _Amp_runtime_trace type
friend BOOL CALLBACK _Init_amp_runtime_trace(PINIT_ONCE _Init_once, PVOID _Param, _Inout_ PVOID *_Context);

public:
    // Destructor for _Amp_runtime_trace, called at program termination
    _AMPIMP ~_Amp_runtime_trace();

    // End event is triggered by multiple other events such us StartComputeEvent to show exactly when given activity completed
    _AMPIMP void _Write_end_event(ULONG _Span_id);

    // Add accelerator configuration information
    // Note: This member function does not have to be exported, it is used by C++ AMP runtime factory
    void _Add_accelerator_config_event(PVOID _Accelerator_id, LPCWSTR _Device_path, LPCWSTR _Device_description);

    // Used by callback function, to write all configuration data when new session is detected
    // Note: This member function does not have to be exported, it is used by C++ AMP runtime factory
    void _Write_all_accelerator_config_events();

    // Started accelerator_view::wait operation
    // Note: This member function does not have to be exported, it is used by C++ AMP runtime factory
    ULONG _Start_accelerator_view_wait_event(PVOID _Accelerator_id, PVOID _Accelerator_view_id);

    // Launched accelerator_view::flush operation
    // Note: This member function does not have to be exported, it is used by C++ AMP runtime factory
    void _Launch_flush_event(PVOID _Accelerator_id, PVOID _Accelerator_view_id);

    // Launched accelerator_view::create_marker operation
    // Note: This member function does not have to be exported, it is used by C++ AMP runtime factory
    ULONG _Launch_marker(PVOID _Accelerator_id, PVOID _Accelerator_view_id);

    // Below are set of helpers that take various types that were available at event injection point and extract all necessary data
    _AMPIMP ULONG _Start_parallel_for_each_event_helper(_In_ _DPC_call_handle *_Handle);

    // This helper wraps functor with wait start and wait end events
    inline concurrency::completion_future _Start_async_op_wait_event_helper(ULONG _Async_op_id, _Event _Ev)
    {
        std::shared_future<void> retFuture;
        concurrency::task_completion_event<void> retTaskCompletionEvent;

        // Create a std::shared_future by creating a deferred task through std::async that waits for the
        // event _Ev to finish. Wrap functor with start and end events
        retFuture = std::async(std::launch::sync, [=]() mutable {
            try 
            {
                if (_Async_op_id == _Amp_runtime_trace::_M_event_disabled)
                {
                    _Ev._Get();
                }
                else
                {
                    auto _Span_id = details::_Get_amp_trace()->_Start_async_op_wait_event(_Async_op_id);
                    _Ev._Get();
                    details::_Get_amp_trace()->_Write_end_event(_Span_id);
                }
            }
            catch(...) 
            {
                // If an exception is encountered when executing the asynchronous operation
                // we should set the exception on the retTaskCompletionEvent so that it is
                // appropriately cancelled and the exception is propagated to continuations
                retTaskCompletionEvent.set_exception(std::current_exception());
                throw;
            }

            retTaskCompletionEvent.set();
        });

        // Register the async event with the runtime asynchronous events manager
        _Register_async_event(_Ev, retFuture);

        // Lets issue a continuation just to swallow any exceptions that are encountered during the
        // async operation and are never observed by the user or are just observed through the
        // shared_future and not through the task
        concurrency::task<void> retTask(retTaskCompletionEvent);
        retTask.then([](concurrency::task<void> _Task) {
            try {
                _Task.get();
            }
            catch(...) {
            }
        });

        return Concurrency::completion_future(retFuture, retTask);
    }

    _AMPIMP ULONG _Start_array_view_synchronize_event_helper(const _Buffer_descriptor &_Buff_desc);
    _AMPIMP ULONG _Launch_array_view_synchronize_event_helper(const _Buffer_descriptor &_Buff_desc);
    
    // Helpers for buffers (array, array_view)
    _AMPIMP ULONG _Start_copy_event_helper(const _Buffer_descriptor &_Src, const _Buffer_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Start_copy_event_helper(nullptr_t, const _Buffer_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Start_copy_event_helper(const _Buffer_descriptor &_Src, nullptr_t, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Launch_async_copy_event_helper(const _Buffer_descriptor &_Src, const _Buffer_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Launch_async_copy_event_helper(nullptr_t, const _Buffer_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Launch_async_copy_event_helper(const _Buffer_descriptor &_Src, nullptr_t, ULONGLONG _Num_bytes_for_copy);
    
    // Helper for textures
    _AMPIMP ULONG _Start_copy_event_helper(const _Texture_descriptor &_Src, nullptr_t, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Start_copy_event_helper(nullptr_t, const _Texture_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Start_copy_event_helper(const _Texture_descriptor &_Src, const _Texture_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Launch_async_copy_event_helper(const _Texture_descriptor &_Src, nullptr_t, ULONGLONG _Num_bytes_for_copy);
    _AMPIMP ULONG _Launch_async_copy_event_helper(nullptr_t, const _Texture_descriptor &_Dest, ULONGLONG _Num_bytes_for_copy);
private:
    // Private constructor. This type is created by factory method
    _Amp_runtime_trace(PENABLECALLBACK _Callback_function, _In_ _ITrace *_Trace);

    // Disallow copy construction 
    _Amp_runtime_trace(const _Amp_runtime_trace&);

    // Disallow assignment operator
    _Amp_runtime_trace& operator=(const _Amp_runtime_trace&);
    
    // Used internally to write configuation events 
    void _Write_accelerator_config_event(const std::tuple<PVOID, LPCWSTR, LPCWSTR> &_ConfigTuple);

    // Event triggered when computation is scheduled
    ULONG _Start_parallel_for_each_event(
        PVOID _Accelerator_id, 
        PVOID _Accelerator_view_id, 
        BOOL _Is_tiled_explicitly, 
        ULONGLONG _Num_of_tiles, 
        ULONG _Num_of_threads_per_tile, 
        BOOL _Is_aliased, 
        ULONG _Num_read_only_resources, 
        ULONG _Num_read_write_resources, 
        ULONGLONG _Size_of_all_resouces, 
        ULONG _Size_of_const_data, 
        ULONGLONG _Size_of_data_for_copy);

    // Synchronous copy operation has started
    ULONG _Start_copy_event(
        PVOID _Src_accelerator_id, 
        PVOID _Src_accelerator_view_id,
        PVOID _Dst_accelerator_id, 
        PVOID _Dst_accelerator_view_id,
        ULONGLONG _Num_bytes_for_copy,
        BOOL _Is_src_staging,
        BOOL _Is_dst_staging);

    // Asynchronous copy operation has been launched
    ULONG _Launch_async_copy_event(
        PVOID _Src_accelerator_id, 
        PVOID _Src_accelerator_view_id,
        PVOID _Dst_accelerator_id, 
        PVOID _Dst_accelerator_view_id,
        ULONGLONG _Num_bytes_for_copy,
        BOOL _Is_src_staging,
        BOOL _Is_dst_staging);

    // Started waiting for asynchronous operation to complete
    _AMPIMP ULONG _Start_async_op_wait_event(ULONG _Async_op_id);

    // Started array_view::synchronize operation
    ULONG _Start_array_view_synchronize_event(ULONGLONG _Num_bytes_to_synchronize);

    // Async array_view::synchronize operation has been launched
    ULONG _Launch_array_view_synchronize_event(ULONGLONG _Num_bytes_to_synchronize);

    // Helper function that extracts information from buffer descriptor
    std::tuple<PVOID, PVOID, BOOL> _Get_resource_diagnostic_info(const _Buffer_descriptor &_Buff_desc) const;

    // Helper function that extracts information from texture descriptor
    std::tuple<PVOID, PVOID, BOOL> _Get_resource_diagnostic_info(const _Texture_descriptor &_Tex_desc) const;

    // Generates unique identifiers for span_id and async_op_id
    ULONG _Get_unique_identifier();

    // Critical section object used by callback function to synchronize following situations:
    // a) multiple sessions have started at the same time
    // b) C++ AMP Runtime factory adds new accelerator config event to the collection 
    Concurrency::critical_section _M_critical_section;

    // Collection of all configuration events at the time of C++ AMP Runtime initialization
    std::vector<std::tuple<PVOID, LPCWSTR, LPCWSTR>> _M_accelerator_configs;

    // Unique counter for span id and async operation id
    volatile ULONG _M_counter;

    // Type that implements ITrace interface and writes events e.g. ETW events 
    _ITrace* _M_trace_ptr;

    // Special value that we return to chain events if provider is disabled
    static const ULONG _M_event_disabled = 0;
};

} // namespace Concurrency::details
} // namespace Concurrency

namespace concurrency = Concurrency;

#pragma pack(pop)
