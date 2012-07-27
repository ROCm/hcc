/***
* collection.h - Windows Runtime Collection/Iterator Wrappers
*
* Copyright (c) Microsoft Corporation. All rights reserved.
****/

#pragma once

#ifndef _COLLECTION_H_
#define _COLLECTION_H_

#ifndef RC_INVOKED

#ifndef __cplusplus_winrt
    #error collection.h requires the /ZW compiler option.
#endif

#include <stddef.h>
#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#define _COLLECTION_ATTRIBUTES [::Platform::Metadata::RuntimeClassName] [::Windows::Foundation::Metadata::Default]

#define _COLLECTION_TRANSLATE             \
} catch (const ::std::bad_alloc&) {       \
    throw ref new OutOfMemoryException(); \
} catch (const ::std::exception&) {       \
    throw ref new FailureException();     \
}

#ifndef _COLLECTION_WUXI
    #ifdef WINAPI_FAMILY
        #include <winapifamily.h>

        #if WINAPI_FAMILY == WINAPI_FAMILY_APP
            #define _COLLECTION_WUXI 1
        #elif WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP
            #define _COLLECTION_WUXI 1
        #else
            #define _COLLECTION_WUXI 0
        #endif
    #else
        #define _COLLECTION_WUXI 1
    #endif
#endif

#ifdef _WIN64
    #pragma pack(push, 16)
#else
    #pragma pack(push, 8)
#endif

#pragma warning(push, 4)

namespace Platform {
  namespace Collections {
    namespace Details {
        namespace WFC = ::Windows::Foundation::Collections;

#if _COLLECTION_WUXI
        namespace WUXI = ::Windows::UI::Xaml::Interop;
#endif // _COLLECTION_WUXI

        typedef ::Windows::Foundation::EventRegistrationToken Token;

        inline void ValidateBounds(bool b) {
            if (!b) {
                throw ref new OutOfBoundsException();
            }
        }

        inline void ValidateCounter(const ::std::shared_ptr<unsigned int>& ctr, unsigned int good_ctr) {
            if (*ctr != good_ctr) {
                throw ref new ChangedStateException();
            }
        }

        inline void ValidateSize(size_t n) {
            if (n > 0x7FFFFFFFUL) {
                throw ref new OutOfMemoryException();
            }
        }

        template <typename X> inline void Init(::std::shared_ptr<unsigned int>& ctr, ::std::shared_ptr<X>& sp) {
            try {
                ctr = ::std::make_shared<unsigned int>(0);
                sp = ::std::make_shared<X>();
            _COLLECTION_TRANSLATE
        }

        template <typename X, typename A> inline void Init(::std::shared_ptr<unsigned int>& ctr, ::std::shared_ptr<X>& sp, A&& a) {
            try {
                ctr = ::std::make_shared<unsigned int>(0);
                sp = ::std::make_shared<X>(::std::forward<A>(a));

                ValidateSize(sp->size());
            _COLLECTION_TRANSLATE
        }

        template <typename X, typename A, typename B> inline void Init(::std::shared_ptr<unsigned int>& ctr, ::std::shared_ptr<X>& sp, A&& a, B&& b) {
            try {
                ctr = ::std::make_shared<unsigned int>(0);
                sp = ::std::make_shared<X>(::std::forward<A>(a), ::std::forward<B>(b));

                ValidateSize(sp->size());
            _COLLECTION_TRANSLATE
        }

        template <typename X, typename A, typename B, typename C> inline void Init(::std::shared_ptr<unsigned int>& ctr, ::std::shared_ptr<X>& sp, A&& a, B&& b, C&& c) {
            try {
                ctr = ::std::make_shared<unsigned int>(0);
                sp = ::std::make_shared<X>(::std::forward<A>(a), ::std::forward<B>(b), ::std::forward<C>(c));

                ValidateSize(sp->size());
            _COLLECTION_TRANSLATE
        }

        inline void IncrementCounter(::std::shared_ptr<unsigned int>& ctr) {
            if (++*ctr == static_cast<unsigned int>(-1)) {
                // Wraparound is imminent! Create a fresh counter.
                ctr = ::std::make_shared<unsigned int>(0);
            }
        }

        ref class VectorChangedEventArgs sealed : public _COLLECTION_ATTRIBUTES WFC::IVectorChangedEventArgs {
        internal:
            VectorChangedEventArgs(WFC::CollectionChange change, unsigned int index)
                : m_change(change), m_index(index) { }

        public:
            virtual property WFC::CollectionChange CollectionChange {
                virtual WFC::CollectionChange get() {
                    return m_change;
                }
            }

            virtual property unsigned int Index {
                virtual unsigned int get() {
                    if (m_change == WFC::CollectionChange::Reset) {
                        throw ref new FailureException();
                    }

                    return m_index;
                }
            }

        private:
            WFC::CollectionChange m_change;
            unsigned int m_index;
        };

        template <typename K> ref class MapChangedEventArgs sealed : public _COLLECTION_ATTRIBUTES WFC::IMapChangedEventArgs<K> {
        internal:
            MapChangedEventArgs(WFC::CollectionChange change, K key)
                : m_change(change), m_key(key) { }

        public:
            virtual property WFC::CollectionChange CollectionChange {
                virtual WFC::CollectionChange get() {
                    return m_change;
                }
            }

            virtual property K Key {
                virtual K get() {
                    if (m_change == WFC::CollectionChange::Reset) {
                        throw ref new FailureException();
                    }

                    return m_key;
                }
            }

        private:
            WFC::CollectionChange m_change;
            K m_key;
        };

        template <typename E, typename T> inline bool VectorIndexOf(const ::std::vector<T>& v, T value, unsigned int * index) {
            auto pred = [&](const T& elem) { return E()(elem, value); };

            *index = static_cast<unsigned int>(::std::find_if(v.begin(), v.end(), pred) - v.begin());

            return *index < v.size();
        }

#if _COLLECTION_WUXI
        template <typename T> struct is_hat : public ::std::false_type { };

        template <typename U> struct is_hat<U^> : public ::std::true_type { };

        template <typename E, typename T> inline bool VectorBindableIndexOf(::std::false_type, const ::std::vector<T>& v, Object^ o, unsigned int * index) {
            IBox<T>^ ib = dynamic_cast<IBox<T>^>(o);

            if (ib) {
                return VectorIndexOf<E>(v, ib->Value, index);
            } else {
                *index = static_cast<unsigned int>(v.size());
                return false;
            }
        }

        template <typename E, typename T> inline bool VectorBindableIndexOf(::std::true_type, const ::std::vector<T>& v, Object^ o, unsigned int * index) {
            T t = dynamic_cast<T>(o);

            if (!o || t) {
                return VectorIndexOf<E>(v, t, index);
            } else {
                *index = static_cast<unsigned int>(v.size());
                return false;
            }
        }
#endif // _COLLECTION_WUXI

        template <typename T> inline unsigned int VectorGetMany(const ::std::vector<T>& v, unsigned int startIndex, WriteOnlyArray<T>^ dest) {
            unsigned int capacity = dest->Length;

            unsigned int actual = static_cast<unsigned int>(v.size()) - startIndex;

            if (actual > capacity) {
                actual = capacity;
            }

            for (unsigned int i = 0; i < actual; ++i) {
                dest->set(i, v[startIndex + i]);
            }

            return actual;
        }

        template <typename T> ref class IteratorForVectorView sealed
            : public _COLLECTION_ATTRIBUTES WFC::IIterator<T>
#if _COLLECTION_WUXI
            , public WUXI::IBindableIterator
#endif // _COLLECTION_WUXI
        {
        private:
            typedef WFC::IIterator<T> WFC_Base;

#if _COLLECTION_WUXI
            typedef WUXI::IBindableIterator WUXI_Base;
#endif // _COLLECTION_WUXI

        internal:
            IteratorForVectorView(const ::std::shared_ptr<unsigned int>& ctr, const ::std::shared_ptr< ::std::vector<T>>& vec)
                : m_ctr(ctr), m_vec(vec), m_good_ctr(*ctr), m_index(0) { }

        public:
            virtual property T Current {
                virtual T get() = WFC_Base::Current::get {
                    ValidateCounter(m_ctr, m_good_ctr);

                    ValidateBounds(m_index < m_vec->size());

                    return (*m_vec)[m_index];
                }
            }

            virtual property bool HasCurrent {
                virtual bool get() {
                    ValidateCounter(m_ctr, m_good_ctr);

                    return m_index < m_vec->size();
                }
            }

            virtual bool MoveNext() {
                ValidateCounter(m_ctr, m_good_ctr);

                ValidateBounds(m_index < m_vec->size());

                ++m_index;
                return m_index < m_vec->size();
            }

            virtual unsigned int GetMany(WriteOnlyArray<T>^ dest) {
                ValidateCounter(m_ctr, m_good_ctr);

                unsigned int actual = VectorGetMany(*m_vec, m_index, dest);

                m_index += actual;

                return actual;
            }

        private:

#if _COLLECTION_WUXI
            virtual Object^ BindableCurrent() = WUXI_Base::Current::get {
                return Current;
            }
#endif // _COLLECTION_WUXI

            ::std::shared_ptr<unsigned int> m_ctr;
            ::std::shared_ptr< ::std::vector<T>> m_vec;
            unsigned int m_good_ctr;
            unsigned int m_index;
        };
    } // namespace Details

    template <typename T, typename E = ::std::equal_to<T>> ref class Vector;
    template <typename T, typename E = ::std::equal_to<T>> ref class VectorView;

    template <typename T, typename E> ref class VectorView sealed
        : public _COLLECTION_ATTRIBUTES Details::WFC::IVectorView<T>
#if _COLLECTION_WUXI
        , public Details::WUXI::IBindableVectorView
#endif // _COLLECTION_WUXI
    {
    private:
        typedef Details::WFC::IVectorView<T> WFC_Base;

#if _COLLECTION_WUXI
        typedef Details::WUXI::IBindableVectorView WUXI_Base;
#endif // _COLLECTION_WUXI

    internal:
        VectorView() {
            Details::Init(m_ctr, m_vec);

            m_good_ctr = 0;
        }

        explicit VectorView(unsigned int size) {
            Details::Init(m_ctr, m_vec, size);

            m_good_ctr = 0;
        }

        VectorView(unsigned int size, T value) {
            Details::Init(m_ctr, m_vec, size, value);

            m_good_ctr = 0;
        }

        explicit VectorView(const ::std::vector<T>& v) {
            Details::Init(m_ctr, m_vec, v);

            m_good_ctr = 0;
        }

        explicit VectorView(::std::vector<T>&& v) {
            Details::Init(m_ctr, m_vec, ::std::move(v));

            m_good_ctr = 0;
        }

        VectorView(const T * ptr, unsigned int size) {
            Details::Init(m_ctr, m_vec, ptr, ptr + size);

            m_good_ctr = 0;
        }

        template <size_t N> explicit VectorView(const T (&arr)[N]) {
            Details::Init(m_ctr, m_vec, arr, arr + N);

            m_good_ctr = 0;
        }

        template <size_t N> explicit VectorView(const ::std::array<T, N>& a) {
            Details::Init(m_ctr, m_vec, a.begin(), a.end());

            m_good_ctr = 0;
        }

        explicit VectorView(const Array<T>^ arr) {
            Details::Init(m_ctr, m_vec, arr->begin(), arr->end());

            m_good_ctr = 0;
        }

        // SFINAE is unnecessary here.
        template <typename InIt> VectorView(InIt first, InIt last) {
            Details::Init(m_ctr, m_vec, first, last);

            m_good_ctr = 0;
        }

    public:
        virtual Details::WFC::IIterator<T>^ First() = WFC_Base::First {
            Details::ValidateCounter(m_ctr, m_good_ctr);

            return ref new Details::IteratorForVectorView<T>(m_ctr, m_vec);
        }

        virtual T GetAt(unsigned int index) = WFC_Base::GetAt {
            Details::ValidateCounter(m_ctr, m_good_ctr);

            Details::ValidateBounds(index < m_vec->size());

            return (*m_vec)[index];
        }

        virtual property unsigned int Size {
            virtual unsigned int get() {
                Details::ValidateCounter(m_ctr, m_good_ctr);

                return static_cast<unsigned int>(m_vec->size());
            }
        }

        virtual bool IndexOf(T value, unsigned int * index) = WFC_Base::IndexOf {
            *index = 0;

            Details::ValidateCounter(m_ctr, m_good_ctr);

            return Details::VectorIndexOf<E>(*m_vec, value, index);
        }

        virtual unsigned int GetMany(unsigned int startIndex, WriteOnlyArray<T>^ dest) {
            Details::ValidateCounter(m_ctr, m_good_ctr);

            Details::ValidateBounds(startIndex <= m_vec->size());

            return Details::VectorGetMany(*m_vec, startIndex, dest);
        }

    private:
        friend ref class Vector<T, E>;

        VectorView(const ::std::shared_ptr<unsigned int>& ctr, const ::std::shared_ptr< ::std::vector<T>>& vec)
            : m_ctr(ctr), m_vec(vec), m_good_ctr(*ctr) { }

#if _COLLECTION_WUXI
        virtual Details::WUXI::IBindableIterator^ BindableFirst() = WUXI_Base::First {
            return safe_cast<Details::WUXI::IBindableIterator^>(First());
        }

        virtual Object^ BindableGetAt(unsigned int index) = WUXI_Base::GetAt {
            return GetAt(index);
        }

        virtual bool BindableIndexOf(Object^ value, unsigned int * index) = WUXI_Base::IndexOf {
            *index = 0;

            Details::ValidateCounter(m_ctr, m_good_ctr);

            return Details::VectorBindableIndexOf<E>(Details::is_hat<T>(), *m_vec, value, index);
        }
#endif // _COLLECTION_WUXI

        ::std::shared_ptr<unsigned int> m_ctr;
        ::std::shared_ptr< ::std::vector<T>> m_vec;
        unsigned int m_good_ctr;
    };

    template <typename T, typename E> ref class Vector sealed
        : public _COLLECTION_ATTRIBUTES Details::WFC::IObservableVector<T>
#if _COLLECTION_WUXI
        , public Details::WUXI::IBindableObservableVector
#endif // _COLLECTION_WUXI
    {
    private:
        typedef Details::WFC::IObservableVector<T> WFC_Base;
        typedef Details::WFC::VectorChangedEventHandler<T> WFC_Handler;

#if _COLLECTION_WUXI
        typedef Details::WUXI::IBindableObservableVector WUXI_Base;
        typedef Details::WUXI::BindableVectorChangedEventHandler WUXI_Handler;
#endif // _COLLECTION_WUXI

    internal:
        Vector() {
            Details::Init(m_ctr, m_vec);

            m_observed = false;
        }

        explicit Vector(unsigned int size) {
            Details::Init(m_ctr, m_vec, size);

            m_observed = false;
        }

        Vector(unsigned int size, T value) {
            Details::Init(m_ctr, m_vec, size, value);

            m_observed = false;
        }

        explicit Vector(const ::std::vector<T>& v) {
            Details::Init(m_ctr, m_vec, v);

            m_observed = false;
        }

        explicit Vector(::std::vector<T>&& v) {
            Details::Init(m_ctr, m_vec, ::std::move(v));

            m_observed = false;
        }

        Vector(const T * ptr, unsigned int size) {
            Details::Init(m_ctr, m_vec, ptr, ptr + size);

            m_observed = false;
        }

        template <size_t N> explicit Vector(const T (&arr)[N]) {
            Details::Init(m_ctr, m_vec, arr, arr + N);

            m_observed = false;
        }

        template <size_t N> explicit Vector(const ::std::array<T, N>& a) {
            Details::Init(m_ctr, m_vec, a.begin(), a.end());

            m_observed = false;
        }

        explicit Vector(const Array<T>^ arr) {
            Details::Init(m_ctr, m_vec, arr->begin(), arr->end());

            m_observed = false;
        }

        // SFINAE is unnecessary here.
        template <typename InIt> Vector(InIt first, InIt last) {
            Details::Init(m_ctr, m_vec, first, last);

            m_observed = false;
        }

    public:
        virtual Details::WFC::IIterator<T>^ First() = WFC_Base::First {
            return ref new Details::IteratorForVectorView<T>(m_ctr, m_vec);
        }

        virtual T GetAt(unsigned int index) = WFC_Base::GetAt {
            Details::ValidateBounds(index < m_vec->size());

            return (*m_vec)[index];
        }

        virtual property unsigned int Size {
            virtual unsigned int get() {
                return static_cast<unsigned int>(m_vec->size());
            }
        }

        virtual bool IndexOf(T value, unsigned int * index) = WFC_Base::IndexOf {
            *index = 0;

            return Details::VectorIndexOf<E>(*m_vec, value, index);
        }

        virtual unsigned int GetMany(unsigned int startIndex, WriteOnlyArray<T>^ dest) {
            Details::ValidateBounds(startIndex <= m_vec->size());

            return Details::VectorGetMany(*m_vec, startIndex, dest);
        }

        virtual Details::WFC::IVectorView<T>^ GetView() = WFC_Base::GetView {
            return ref new VectorView<T, E>(m_ctr, m_vec);
        }

        virtual void SetAt(unsigned int index, T item) = WFC_Base::SetAt {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateBounds(index < m_vec->size());

                (*m_vec)[index] = item;

                NotifyChanged(index);
            _COLLECTION_TRANSLATE
        }

        virtual void InsertAt(unsigned int index, T item) = WFC_Base::InsertAt {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateBounds(index <= m_vec->size());

                Details::ValidateSize(m_vec->size() + 1);

                m_vec->insert(m_vec->begin() + index, item);

                NotifyInserted(index);
            _COLLECTION_TRANSLATE
        }

        virtual void Append(T item) = WFC_Base::Append {
            try {
                Details::IncrementCounter(m_ctr);

                size_t n = m_vec->size();

                Details::ValidateSize(n + 1);

                m_vec->push_back(item);

                NotifyInserted(static_cast<unsigned int>(n));
            _COLLECTION_TRANSLATE
        }

        virtual void RemoveAt(unsigned int index) {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateBounds(index < m_vec->size());

                m_vec->erase(m_vec->begin() + index);

                NotifyRemoved(index);
            _COLLECTION_TRANSLATE
        }

        virtual void RemoveAtEnd() {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateBounds(!m_vec->empty());

                m_vec->pop_back();

                NotifyRemoved(static_cast<unsigned int>(m_vec->size()));
            _COLLECTION_TRANSLATE
        }

        virtual void Clear() {
            try {
                Details::IncrementCounter(m_ctr);

                m_vec->clear();

                NotifyReset();
            _COLLECTION_TRANSLATE
        }

        virtual void ReplaceAll(const Array<T>^ arr) {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateSize(arr->Length);

                m_vec->assign(arr->begin(), arr->end());

                NotifyReset();
            _COLLECTION_TRANSLATE
        }

        virtual event WFC_Handler^ VectorChanged {
            virtual Details::Token add(WFC_Handler^ e) = WFC_Base::VectorChanged::add {
                m_observed = true;
                return m_wfc_event += e;
            }

            virtual void remove(Details::Token t) = WFC_Base::VectorChanged::remove {
                m_wfc_event -= t;
            }
        };

    private:
        void Notify(Details::WFC::CollectionChange change, unsigned int index) {
            if (m_observed) {
                auto args = ref new Details::VectorChangedEventArgs(change, index);
                m_wfc_event(this, args);

#if _COLLECTION_WUXI
                m_wuxi_event(this, args);
#endif // _COLLECTION_WUXI

            }
        }

        void NotifyReset() {
            Notify(Details::WFC::CollectionChange::Reset, 0);
        }

        void NotifyInserted(unsigned int index) {
            Notify(Details::WFC::CollectionChange::ItemInserted, index);
        }

        void NotifyRemoved(unsigned int index) {
            Notify(Details::WFC::CollectionChange::ItemRemoved, index);
        }

        void NotifyChanged(unsigned int index) {
            Notify(Details::WFC::CollectionChange::ItemChanged, index);
        }

#if _COLLECTION_WUXI
        virtual Details::WUXI::IBindableIterator^ BindableFirst() = WUXI_Base::First {
            return safe_cast<Details::WUXI::IBindableIterator^>(First());
        }

        virtual Object^ BindableGetAt(unsigned int index) = WUXI_Base::GetAt {
            return GetAt(index);
        }

        virtual bool BindableIndexOf(Object^ value, unsigned int * index) = WUXI_Base::IndexOf {
            *index = 0;

            return Details::VectorBindableIndexOf<E>(Details::is_hat<T>(), *m_vec, value, index);
        }

        virtual Details::WUXI::IBindableVectorView^ BindableGetView() = WUXI_Base::GetView {
            return safe_cast<Details::WUXI::IBindableVectorView^>(GetView());
        }

        virtual void BindableSetAt(unsigned int index, Object^ item) = WUXI_Base::SetAt {
            SetAt(index, safe_cast<T>(item));
        }

        virtual void BindableInsertAt(unsigned int index, Object^ item) = WUXI_Base::InsertAt {
            InsertAt(index, safe_cast<T>(item));
        }

        virtual void BindableAppend(Object^ item) = WUXI_Base::Append {
            Append(safe_cast<T>(item));
        }

        virtual Details::Token BindableEventAdd(WUXI_Handler^ e) = WUXI_Base::VectorChanged::add {
            m_observed = true;
            return m_wuxi_event += e;
        }

        virtual void BindableEventRemove(Details::Token t) = WUXI_Base::VectorChanged::remove {
            m_wuxi_event -= t;
        }
#endif // _COLLECTION_WUXI

        ::std::shared_ptr<unsigned int> m_ctr;
        ::std::shared_ptr< ::std::vector<T>> m_vec;
        bool m_observed;

        event WFC_Handler^ m_wfc_event;

#if _COLLECTION_WUXI
        event WUXI_Handler^ m_wuxi_event;
#endif // _COLLECTION_WUXI

    };


    namespace Details {
        template <typename K, typename V> ref class KeyValuePair sealed : public _COLLECTION_ATTRIBUTES WFC::IKeyValuePair<K, V> {
        internal:
            KeyValuePair(K key, V value)
                : m_key(key), m_value(value) { }

        public:
            virtual property K Key {
                virtual K get() {
                    return m_key;
                }
            }

            virtual property V Value {
                virtual V get() {
                    return m_value;
                }
            }

        private:
            K m_key;
            V m_value;
        };

        template <typename K, typename V, typename C> ref class IteratorForMapView sealed : public _COLLECTION_ATTRIBUTES WFC::IIterator<WFC::IKeyValuePair<K, V>^> {
        private:
            typedef ::std::map<K, V, C> StdMap;

        internal:
            IteratorForMapView(const ::std::shared_ptr<unsigned int>& ctr, const ::std::shared_ptr<StdMap>& m)
                : m_ctr(ctr), m_map(m), m_good_ctr(*ctr), m_iter(m->begin()) { }

        public:
            virtual property WFC::IKeyValuePair<K, V>^ Current {
                virtual WFC::IKeyValuePair<K, V>^ get() {
                    ValidateCounter(m_ctr, m_good_ctr);

                    ValidateBounds(m_iter != m_map->end());

                    return ref new KeyValuePair<K, V>(m_iter->first, m_iter->second);
                }
            }

            virtual property bool HasCurrent {
                virtual bool get() {
                    ValidateCounter(m_ctr, m_good_ctr);

                    return m_iter != m_map->end();
                }
            }

            virtual bool MoveNext() {
                ValidateCounter(m_ctr, m_good_ctr);

                ValidateBounds(m_iter != m_map->end());

                ++m_iter;
                return m_iter != m_map->end();
            }

            virtual unsigned int GetMany(WriteOnlyArray<WFC::IKeyValuePair<K, V>^>^ dest) {
                ValidateCounter(m_ctr, m_good_ctr);

                unsigned int capacity = dest->Length;

                unsigned int actual = 0;

                while (capacity > 0 && m_iter != m_map->end()) {
                    dest->set(actual, ref new KeyValuePair<K, V>(m_iter->first, m_iter->second));
                    ++m_iter;
                    --capacity;
                    ++actual;
                }

                return actual;
            }

        private:
            ::std::shared_ptr<unsigned int> m_ctr;
            ::std::shared_ptr<StdMap> m_map;
            unsigned int m_good_ctr;
            typename StdMap::const_iterator m_iter;
        };
    } // namespace Details

    template <typename K, typename V, typename C = ::std::less<K>> ref class Map;
    template <typename K, typename V, typename C = ::std::less<K>> ref class MapView;

    template <typename K, typename V, typename C> ref class MapView sealed : public _COLLECTION_ATTRIBUTES Details::WFC::IMapView<K, V> {
    private:
        typedef                  ::std::map<K, V, C> StdMap;
        typedef Details::IteratorForMapView<K, V, C> MyIterator;
                       friend ref class Map<K, V, C>;

    internal:
        explicit MapView(const C& comp = C()) {
            Details::Init(m_ctr, m_map, comp);

            m_good_ctr = 0;
        }

        explicit MapView(const StdMap& m) {
            Details::Init(m_ctr, m_map, m);

            m_good_ctr = 0;
        }

        explicit MapView(StdMap&& m) {
            Details::Init(m_ctr, m_map, ::std::move(m));

            m_good_ctr = 0;
        }

        template <typename InIt> MapView(InIt first, InIt last, const C& comp = C()) {
            Details::Init(m_ctr, m_map, first, last, comp);

            m_good_ctr = 0;
        }

    public:
        virtual Details::WFC::IIterator<Details::WFC::IKeyValuePair<K, V>^>^ First() {
            Details::ValidateCounter(m_ctr, m_good_ctr);

            return ref new MyIterator(m_ctr, m_map);
        }

        virtual V Lookup(K key) {
            Details::ValidateCounter(m_ctr, m_good_ctr);

            auto i = m_map->find(key);

            Details::ValidateBounds(i != m_map->end());

            return i->second;
        }

        virtual property unsigned int Size {
            virtual unsigned int get() {
                Details::ValidateCounter(m_ctr, m_good_ctr);

                return static_cast<unsigned int>(m_map->size());
            }
        }

        virtual bool HasKey(K key) {
            Details::ValidateCounter(m_ctr, m_good_ctr);

            return m_map->find(key) != m_map->end();
        }

        virtual void Split(Details::WFC::IMapView<K, V>^ * firstPartition, Details::WFC::IMapView<K, V>^ * secondPartition) {
            *firstPartition = nullptr;
            *secondPartition = nullptr;

            Details::ValidateCounter(m_ctr, m_good_ctr);
        }

    private:
        MapView(const ::std::shared_ptr<unsigned int>& ctr, const ::std::shared_ptr<StdMap>& m)
            : m_ctr(ctr), m_map(m), m_good_ctr(*ctr) { }

        ::std::shared_ptr<unsigned int> m_ctr;
        ::std::shared_ptr<StdMap> m_map;
        unsigned int m_good_ctr;
    };

    template <typename K, typename V, typename C> ref class Map sealed : public _COLLECTION_ATTRIBUTES Details::WFC::IObservableMap<K, V> {
    private:
        typedef                           ::std::map<K, V, C> StdMap;
        typedef          Details::IteratorForMapView<K, V, C> MyIterator;
        typedef                              MapView<K, V, C> MyView;
        typedef Details::WFC::MapChangedEventHandler<K, V>    WFC_Handler;

    internal:
        explicit Map(const C& comp = C()) {
            Details::Init(m_ctr, m_map, comp);

            m_observed = false;
        }

        explicit Map(const StdMap& m) {
            Details::Init(m_ctr, m_map, m);

            m_observed = false;
        }

        explicit Map(StdMap&& m) {
            Details::Init(m_ctr, m_map, ::std::move(m));

            m_observed = false;
        }

        template <typename InIt> Map(InIt first, InIt last, const C& comp = C()) {
            Details::Init(m_ctr, m_map, first, last, comp);

            m_observed = false;
        }

    public:
        virtual Details::WFC::IIterator<Details::WFC::IKeyValuePair<K, V>^>^ First() {
            return ref new MyIterator(m_ctr, m_map);
        }

        virtual V Lookup(K key) {
            auto i = m_map->find(key);

            Details::ValidateBounds(i != m_map->end());

            return i->second;
        }

        virtual property unsigned int Size {
            virtual unsigned int get() {
                return static_cast<unsigned int>(m_map->size());
            }
        }

        virtual bool HasKey(K key) {
            return m_map->find(key) != m_map->end();
        }

        virtual Details::WFC::IMapView<K, V>^ GetView() {
            return ref new MyView(m_ctr, m_map);
        }

        virtual bool Insert(K key, V value) {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateSize(m_map->size() + 1);

                auto p = m_map->insert(::std::make_pair(key, value));

                if (p.second) {
                    NotifyInserted(key);
                } else {
                    p.first->second = value;
                    NotifyChanged(key);
                }

                return !p.second;
            _COLLECTION_TRANSLATE
        }

        virtual void Remove(K key) {
            try {
                Details::IncrementCounter(m_ctr);

                Details::ValidateBounds(m_map->erase(key) == 1);

                NotifyRemoved(key);
            _COLLECTION_TRANSLATE
        }

        virtual void Clear() {
            try {
                Details::IncrementCounter(m_ctr);

                m_map->clear();

                NotifyReset();
            _COLLECTION_TRANSLATE
        }

        virtual event WFC_Handler^ MapChanged {
            virtual Details::Token add(WFC_Handler^ e) {
                m_observed = true;
                return m_wfc_event += e;
            }

            virtual void remove(Details::Token t) {
                m_wfc_event -= t;
            }
        };

    private:
        void NotifyReset() {
            if (m_observed) {
                m_wfc_event(this, ref new Details::MapChangedEventArgs<K>(Details::WFC::CollectionChange::Reset, K()));
            }
        }

        void NotifyInserted(K key) {
            if (m_observed) {
                m_wfc_event(this, ref new Details::MapChangedEventArgs<K>(Details::WFC::CollectionChange::ItemInserted, key));
            }
        }

        void NotifyRemoved(K key) {
            if (m_observed) {
                m_wfc_event(this, ref new Details::MapChangedEventArgs<K>(Details::WFC::CollectionChange::ItemRemoved, key));
            }
        }

        void NotifyChanged(K key) {
            if (m_observed) {
                m_wfc_event(this, ref new Details::MapChangedEventArgs<K>(Details::WFC::CollectionChange::ItemChanged, key));
            }
        }

        ::std::shared_ptr<unsigned int> m_ctr;
        ::std::shared_ptr<StdMap> m_map;
        bool m_observed;

        event WFC_Handler^ m_wfc_event;
    };


    template <typename X> class InputIterator;
    template <typename T> class VectorIterator;
    template <typename T> class VectorViewIterator;
    template <typename T> class BackInsertIterator;
  } // namespace Collections
} // namespace Platform

template <typename X> struct ::std::_Is_checked_helper< ::Platform::Collections::InputIterator<X>>
    : public ::std::true_type { };

template <typename T> struct ::std::_Is_checked_helper< ::Platform::Collections::VectorIterator<T>>
    : public ::std::true_type { };

template <typename T> struct ::std::_Is_checked_helper< ::Platform::Collections::VectorViewIterator<T>>
    : public ::std::true_type { };

template <typename T> struct ::std::_Is_checked_helper< ::Platform::Collections::BackInsertIterator<T>>
    : public ::std::true_type { };

namespace Platform {
  namespace Collections {
    template <typename X> class InputIterator {
    public:
        typedef ::std::input_iterator_tag iterator_category;
        typedef                         X value_type;
        typedef                 ptrdiff_t difference_type;
        typedef                 const X * pointer;
        typedef                 const X & reference;

        InputIterator() { }

        explicit InputIterator(Details::WFC::IIterator<X>^ iter) {
            if (iter->HasCurrent) {
                m_iter = iter;
                m_val = iter->Current;
            }
        }

        bool operator==(const InputIterator& other) const {
            return !!m_iter == !!other.m_iter;
        }

        bool operator!=(const InputIterator& other) const {
            return !(*this == other);
        }

        reference operator*() const {
            return m_val;
        }

        pointer operator->() const {
            return &m_val;
        }

        InputIterator& operator++() {
            if (m_iter->MoveNext()) {
                m_val = m_iter->Current;
            } else {
                m_iter = nullptr;
            }

            return *this;
        }

        InputIterator operator++(int) {
            InputIterator old(*this);
            ++*this;
            return old;
        }

    private:
        Details::WFC::IIterator<X>^ m_iter;
        X m_val;
    };

    namespace Details {
        template <typename T> class VectorProxy {
        public:
            VectorProxy(WFC::IVector<T>^ v, ptrdiff_t n)
                : m_v(v), m_i(static_cast<unsigned int>(n)) { }

            VectorProxy& operator=(const VectorProxy& other) {
                m_v->SetAt(m_i, other.m_v->GetAt(other.m_i));
                return *this;
            }

            VectorProxy& operator=(T t) {
                m_v->SetAt(m_i, t);
                return *this;
            }

            operator T() const {
                return m_v->GetAt(m_i);
            }

            T operator->() const {
                return m_v->GetAt(m_i);
            }

            void swap(const VectorProxy& other) const {
                T t1(m_v->GetAt(m_i));
                T t2(other.m_v->GetAt(other.m_i));

                m_v->SetAt(m_i, t2);
                other.m_v->SetAt(other.m_i, t1);
            }

            void swap(T& t) const {
                T temp(t);
                t = m_v->GetAt(m_i);
                m_v->SetAt(m_i, temp);
            }

        private:
            WFC::IVector<T>^ m_v;
            unsigned int m_i;
        };

        template <typename T> inline void swap(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            l.swap(r);
        }

        template <typename T> inline void swap(const VectorProxy<T>& p, T& t) {
            p.swap(t);
        }

        template <typename T> inline void swap(T& t, const VectorProxy<T>& p) {
            p.swap(t);
        }

        template <typename T> inline bool operator==(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            return static_cast<T>(l) == static_cast<T>(r);
        }

        template <typename T> inline bool operator==(const VectorProxy<T>& l, const T& t) {
            return static_cast<T>(l) == t;
        }

        template <typename T> inline bool operator==(const T& t, const VectorProxy<T>& r) {
            return t == static_cast<T>(r);
        }

        template <typename T> inline bool operator!=(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            return static_cast<T>(l) != static_cast<T>(r);
        }

        template <typename T> inline bool operator!=(const VectorProxy<T>& l, const T& t) {
            return static_cast<T>(l) != t;
        }

        template <typename T> inline bool operator!=(const T& t, const VectorProxy<T>& r) {
            return t != static_cast<T>(r);
        }

        template <typename T> inline bool operator<(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            return static_cast<T>(l) < static_cast<T>(r);
        }

        template <typename T> inline bool operator<(const VectorProxy<T>& l, const T& t) {
            return static_cast<T>(l) < t;
        }

        template <typename T> inline bool operator<(const T& t, const VectorProxy<T>& r) {
            return t < static_cast<T>(r);
        }

        template <typename T> inline bool operator<=(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            return static_cast<T>(l) <= static_cast<T>(r);
        }

        template <typename T> inline bool operator<=(const VectorProxy<T>& l, const T& t) {
            return static_cast<T>(l) <= t;
        }

        template <typename T> inline bool operator<=(const T& t, const VectorProxy<T>& r) {
            return t <= static_cast<T>(r);
        }

        template <typename T> inline bool operator>(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            return static_cast<T>(l) > static_cast<T>(r);
        }

        template <typename T> inline bool operator>(const VectorProxy<T>& l, const T& t) {
            return static_cast<T>(l) > t;
        }

        template <typename T> inline bool operator>(const T& t, const VectorProxy<T>& r) {
            return t > static_cast<T>(r);
        }

        template <typename T> inline bool operator>=(const VectorProxy<T>& l, const VectorProxy<T>& r) {
            return static_cast<T>(l) >= static_cast<T>(r);
        }

        template <typename T> inline bool operator>=(const VectorProxy<T>& l, const T& t) {
            return static_cast<T>(l) >= t;
        }

        template <typename T> inline bool operator>=(const T& t, const VectorProxy<T>& r) {
            return t >= static_cast<T>(r);
        }

        template <typename T> class ArrowProxy {
        public:
            explicit ArrowProxy(T t)
                : m_val(t) { }

            const T * operator->() const {
                return &m_val;
            }

        private:
            T m_val;
        };
    } // namespace Details

    template <typename T> class VectorIterator {
    public:
        typedef ::std::random_access_iterator_tag iterator_category;
        typedef                                 T value_type;
        typedef                         ptrdiff_t difference_type;
        typedef         Details::VectorProxy<T> * pointer;
        typedef         Details::VectorProxy<T>   reference;

        VectorIterator()
            : m_v(nullptr), m_i(0) { }

        explicit VectorIterator(Details::WFC::IVector<T>^ v)
            : m_v(v), m_i(0) { }

        reference operator*() const {
            return reference(m_v, m_i);
        }

        Details::ArrowProxy<T> operator->() const {
            return Details::ArrowProxy<T>(m_v->GetAt(static_cast<unsigned int>(m_i)));
        }

        reference operator[](difference_type n) const {
            return reference(m_v, m_i + n);
        }

        VectorIterator& operator++() {
            ++m_i;
            return *this;
        }

        VectorIterator& operator--() {
            --m_i;
            return *this;
        }

        VectorIterator operator++(int) {
            VectorIterator old(*this);
            ++*this;
            return old;
        }

        VectorIterator operator--(int) {
            VectorIterator old(*this);
            --*this;
            return old;
        }

        VectorIterator& operator+=(difference_type n) {
            m_i += n;
            return *this;
        }

        VectorIterator& operator-=(difference_type n) {
            m_i -= n;
            return *this;
        }

        VectorIterator operator+(difference_type n) const {
            VectorIterator ret(*this);
            ret += n;
            return ret;
        }

        VectorIterator operator-(difference_type n) const {
            VectorIterator ret(*this);
            ret -= n;
            return ret;
        }

        difference_type operator-(const VectorIterator& other) const {
            return m_i - other.m_i;
        }

        bool operator==(const VectorIterator& other) const {
            return m_i == other.m_i;
        }

        bool operator!=(const VectorIterator& other) const {
            return m_i != other.m_i;
        }

        bool operator<(const VectorIterator& other) const {
            return m_i < other.m_i;
        }

        bool operator>(const VectorIterator& other) const {
            return m_i > other.m_i;
        }

        bool operator<=(const VectorIterator& other) const {
            return m_i <= other.m_i;
        }

        bool operator>=(const VectorIterator& other) const {
            return m_i >= other.m_i;
        }

    private:
        Details::WFC::IVector<T>^ m_v;
        difference_type m_i;
    };

    template <typename T> inline VectorIterator<T> operator+(ptrdiff_t n, const VectorIterator<T>& i) {
        return i + n;
    }

    template <typename T> class VectorViewIterator {
    public:
        typedef ::std::random_access_iterator_tag iterator_category;
        typedef                                 T value_type;
        typedef                         ptrdiff_t difference_type;
        typedef                               T * pointer;
        typedef                               T   reference;

        VectorViewIterator()
            : m_v(nullptr), m_i(0) { }

        explicit VectorViewIterator(Details::WFC::IVectorView<T>^ v)
            : m_v(v), m_i(0) { }

        reference operator*() const {
            return m_v->GetAt(static_cast<unsigned int>(m_i));
        }

        Details::ArrowProxy<T> operator->() const {
            return Details::ArrowProxy<T>(m_v->GetAt(static_cast<unsigned int>(m_i)));
        }

        reference operator[](difference_type n) const {
            return m_v->GetAt(static_cast<unsigned int>(m_i + n));
        }

        VectorViewIterator& operator++() {
            ++m_i;
            return *this;
        }

        VectorViewIterator& operator--() {
            --m_i;
            return *this;
        }

        VectorViewIterator operator++(int) {
            VectorViewIterator old(*this);
            ++*this;
            return old;
        }

        VectorViewIterator operator--(int) {
            VectorViewIterator old(*this);
            --*this;
            return old;
        }

        VectorViewIterator& operator+=(difference_type n) {
            m_i += n;
            return *this;
        }

        VectorViewIterator& operator-=(difference_type n) {
            m_i -= n;
            return *this;
        }

        VectorViewIterator operator+(difference_type n) const {
            VectorViewIterator ret(*this);
            ret += n;
            return ret;
        }

        VectorViewIterator operator-(difference_type n) const {
            VectorViewIterator ret(*this);
            ret -= n;
            return ret;
        }

        difference_type operator-(const VectorViewIterator& other) const {
            return m_i - other.m_i;
        }

        bool operator==(const VectorViewIterator& other) const {
            return m_i == other.m_i;
        }

        bool operator!=(const VectorViewIterator& other) const {
            return m_i != other.m_i;
        }

        bool operator<(const VectorViewIterator& other) const {
            return m_i < other.m_i;
        }

        bool operator>(const VectorViewIterator& other) const {
            return m_i > other.m_i;
        }

        bool operator<=(const VectorViewIterator& other) const {
            return m_i <= other.m_i;
        }

        bool operator>=(const VectorViewIterator& other) const {
            return m_i >= other.m_i;
        }

    private:
        Details::WFC::IVectorView<T>^ m_v;
        difference_type m_i;
    };

    template <typename T> inline VectorViewIterator<T> operator+(ptrdiff_t n, const VectorViewIterator<T>& i) {
        return i + n;
    }

    template <typename T> class BackInsertIterator
        : public ::std::iterator< ::std::output_iterator_tag, void, void, void, void> {
    public:
        explicit BackInsertIterator(Details::WFC::IVector<T>^ v) : m_v(v) { }

        BackInsertIterator& operator=(const T& t) {
            m_v->Append(t);
            return *this;
        }

        BackInsertIterator& operator*() {
            return *this;
        }

        BackInsertIterator& operator++() {
            return *this;
        }

        BackInsertIterator operator++(int) {
            return *this;
        }

    private:
        Details::WFC::IVector<T>^ m_v;
    };

    namespace Details {
        template <typename T, typename I> inline ::std::vector<T> ToVector(I^ v) {
            unsigned int size = v->Size;

            ::std::vector<T> ret(size);

            for (unsigned int actual = 0; actual < size; ) {
                Array<T>^ arr = ref new Array<T>(size - actual);

                unsigned int n = v->GetMany(actual, arr);

                if (n == 0) {
                    throw ref new FailureException();
                }

                ::std::copy_n(arr->begin(), n, ret.begin() + actual);

                actual += n;
            }

            return ret;
        }
    } // namespace Details
  } // namespace Collections
} // namespace Platform

namespace Windows {
  namespace Foundation {
    namespace Collections {
        template <typename X> inline ::Platform::Collections::InputIterator<X> begin(IIterable<X>^ i) {
            return ::Platform::Collections::InputIterator<X>(i->First());
        }

        template <typename X> inline ::Platform::Collections::InputIterator<X> end(IIterable<X>^) {
            return ::Platform::Collections::InputIterator<X>();
        }

        template <typename T> inline ::Platform::Collections::VectorIterator<T> begin(IVector<T>^ v) {
            return ::Platform::Collections::VectorIterator<T>(v);
        }

        template <typename T> inline ::Platform::Collections::VectorIterator<T> end(IVector<T>^ v) {
            return ::Platform::Collections::VectorIterator<T>(v) + v->Size;
        }

        template <typename T> inline ::Platform::Collections::VectorViewIterator<T> begin(IVectorView<T>^ v) {
            return ::Platform::Collections::VectorViewIterator<T>(v);
        }

        template <typename T> inline ::Platform::Collections::VectorViewIterator<T> end(IVectorView<T>^ v) {
            return ::Platform::Collections::VectorViewIterator<T>(v) + v->Size;
        }

        template <typename T> inline ::std::vector<T> to_vector(IVector<T>^ v) {
            return ::Platform::Collections::Details::ToVector<T>(v);
        }

        template <typename T> inline ::std::vector<T> to_vector(IVectorView<T>^ v) {
            return ::Platform::Collections::Details::ToVector<T>(v);
        }

        template <typename T> inline ::Platform::Collections::BackInsertIterator<T> back_inserter(IVector<T>^ v) {
            return ::Platform::Collections::BackInsertIterator<T>(v);
        }

        template <typename T> inline ::Platform::Collections::BackInsertIterator<T> back_inserter(IObservableVector<T>^ v) {
            return ::Platform::Collections::BackInsertIterator<T>(v);
        }
    } // namespace Collections
  } // namespace Foundation
} // namespace Windows

namespace Platform {
  namespace Collections {
    template <typename T, typename E> inline BackInsertIterator<T> back_inserter(Vector<T, E>^ v) {
        return BackInsertIterator<T>(v);
    }
  } // namespace Collections
} // namespace Platform

template <> struct ::std::hash< ::Platform::String^>
    : public ::std::unary_function< ::Platform::String^, size_t> {

    size_t operator()(::Platform::String^ s) const {
        return ::std::_Hash_seq(reinterpret_cast<const unsigned char *>(s->Data()), s->Length() * sizeof(wchar_t));
    }
};

#undef _COLLECTION_ATTRIBUTES
#undef _COLLECTION_TRANSLATE
#undef _COLLECTION_WUXI

#pragma warning(pop)
#pragma pack(pop)

#endif // RC_INVOKED

#endif // _COLLECTION_H_
