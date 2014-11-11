//Add License for NVIDIA's thrust

#ifndef BOLT_TYPE_TRAITS_H
#define BOLT_TYPE_TRAITS_H
namespace bolt {
namespace cl {
    namespace detail {

        template<typename _Tp, _Tp __v>
        struct integral_constant
        {
            static const _Tp                      value = __v;
            typedef _Tp                           value_type;
            typedef integral_constant<_Tp, __v>   type;
        };
 
        /// typedef for true_type
        typedef integral_constant<bool, true>     true_type;

        /// typedef for true_type
        typedef integral_constant<bool, false>    false_type;

        //template<typename T> struct is_integral : public std::tr1::is_integral<T> {};
        template<typename T> struct is_integral                           : public false_type {};
        template<>           struct is_integral<bool>                     : public true_type {};
        template<>           struct is_integral<char>                     : public true_type {};
        template<>           struct is_integral<signed char>              : public true_type {};
        template<>           struct is_integral<unsigned char>            : public true_type {};
        template<>           struct is_integral<short>                    : public true_type {};
        template<>           struct is_integral<unsigned short>           : public true_type {};
        template<>           struct is_integral<int>                      : public true_type {};
        template<>           struct is_integral<unsigned int>             : public true_type {};
        template<>           struct is_integral<long>                     : public true_type {};
        template<>           struct is_integral<unsigned long>            : public true_type {};
        template<>           struct is_integral<long long>                : public true_type {};
        template<>           struct is_integral<unsigned long long>       : public true_type {};
        template<>           struct is_integral<const bool>               : public true_type {};
        template<>           struct is_integral<const char>               : public true_type {};
        template<>           struct is_integral<const unsigned char>      : public true_type {};
        template<>           struct is_integral<const short>              : public true_type {};
        template<>           struct is_integral<const unsigned short>     : public true_type {};
        template<>           struct is_integral<const int>                : public true_type {};
        template<>           struct is_integral<const unsigned int>       : public true_type {};
        template<>           struct is_integral<const long>               : public true_type {};
        template<>           struct is_integral<const unsigned long>      : public true_type {};
        template<>           struct is_integral<const long long>          : public true_type {};
        template<>           struct is_integral<const unsigned long long> : public true_type {};

        template<typename T> struct is_floating_point              : public false_type {};
        template<>           struct is_floating_point<float>       : public true_type {};
        template<>           struct is_floating_point<double>      : public true_type {};
        template<>           struct is_floating_point<long double> : public true_type {};

        template<typename T> struct is_arithmetic               : public is_integral<T> {};
        template<>           struct is_arithmetic<float>        : public true_type {};
        template<>           struct is_arithmetic<double>       : public true_type {};
        template<>           struct is_arithmetic<const float>  : public true_type {};
        template<>           struct is_arithmetic<const double> : public true_type {};

        template<typename T> struct is_pointer      : public false_type {};
        template<typename T> struct is_pointer<T *> : public true_type  {};

        template<typename T> struct is_device_ptr  : public false_type {};

        template<typename T> struct is_void             : public false_type {};
        template<>           struct is_void<void>       : public true_type {};
        template<>           struct is_void<const void> : public true_type {};


        template<typename T>
          struct identity_
        {
          typedef T type;
        }; // end identity

        template<typename T1, typename T2>
          struct is_same
            : public false_type
        {
        }; // end is_same

        template<typename T>
          struct is_same<T,T>
            : public true_type
        {
        }; // end is_same

        template<typename T1, typename T2>
          struct lazy_is_same
            : is_same<typename T1::type, typename T2::type>
        {
        }; // end lazy_is_same

        template <typename Condition1,               typename Condition2,              typename Condition3 = false_type,
                  typename Condition4  = false_type, typename Condition5 = false_type, typename Condition6 = false_type,
                  typename Condition7  = false_type, typename Condition8 = false_type, typename Condition9 = false_type,
                  typename Condition10 = false_type>
          struct or_
            : public integral_constant<
                bool,
                Condition1::value || Condition2::value || Condition3::value || Condition4::value || Condition5::value || Condition6::value || Condition7::value || Condition8::value || Condition9::value || Condition10::value
              >
        {
        }; // end or_

        template <typename Condition1, typename Condition2, typename Condition3 = true_type>
          struct and_
            : public integral_constant<bool, Condition1::value && Condition2::value && Condition3::value>
        {
        }; // end and_

        template <typename Boolean>
          struct not_
            : public integral_constant<bool, !Boolean::value>
        {
        }; // end not_

        struct na
        {
            typedef na type;
            enum { value = 0 };
        };

        template<
              bool C
            , typename T1
            , typename T2
            >
        struct if_c
        {
            typedef T1 type;
        };

        template<
              typename T1
            , typename T2
            >
        struct if_c<false,T1,T2>
        {
            typedef T2 type;
        };

        template<
              typename T1 = na
            , typename T2 = na
            , typename T3 = na
            >
        struct if_
        {
         private:
            typedef if_c<
                static_cast<bool>(T1::value)
                , T2
                , T3
                > almost_type_;
 
         public:
            typedef typename almost_type_::type type;
        };        

        ////template <bool, typename Then, typename Else>
        ////  struct eval_if
        ////{
        ////}; // end eval_if

        ////template<typename Then, typename Else>
        ////  struct eval_if<true, Then, Else>
        ////{
        ////  typedef typename Then::type type;
        ////}; // end eval_if

        ////template<typename Then, typename Else>
        ////  struct eval_if<false, Then, Else>
        ////{
        ////  typedef typename Else::type type;
        ////}; // end eval_if

        template<
              typename C = na
            , typename F1 = na
            , typename F2 = na
            >
        struct eval_if
            : if_<C,F1,F2>::type
        {
            //typedef typename if_<C,F1,F2>::type f_;
            //typedef typename f_::type type;
        };


        template<class T> struct addr_impl_ref
        {
            T & v_;

            inline addr_impl_ref( T & v ): v_( v ) {}
            inline operator T& () const { return v_; }

        private:
            addr_impl_ref & operator=(const addr_impl_ref &);
        };

        template<class T> struct addressof_impl
        {
            static inline T * f( T & v, long )
            {
                return reinterpret_cast<T*>(
                    &const_cast<char&>(reinterpret_cast<const volatile char &>(v)));
            }

            static inline T * f( T * v, int )
            {
                return v;
            }
        };

        template<class T> T * addressof( T & v )
        {
            return bolt::cl::detail::addressof_impl<T>::f( bolt::cl::detail::addr_impl_ref<T>( v ), 0 );
        }

        template< bool C_ > struct bool_;

        // shorcuts
        typedef bool_<true> true_;
        typedef bool_<false> false_;

        // is_reference
        template< typename T > struct is_lvalue_reference 
            : public integral_constant<bool,false> 
        { 
        public:
        }; 
        template< typename T > struct is_rvalue_reference 
            : public integral_constant<bool,false> 
        {
        public:
        }; 
        template <bool b1, bool b2, bool b3 = false, bool b4 = false, bool b5 = false, bool b6 = false, bool b7 = false>
        struct ice_or;

        template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
        struct ice_or
        {
            static const bool  value = true;
        };

        template <>
        struct ice_or<false, false, false, false, false, false, false>
        {
            static const bool  value = false;
        };

        
        template <typename T>
        struct is_reference_impl
        {
            static const bool  value =
                  (ice_or<
                     is_lvalue_reference<T>::value, is_rvalue_reference<T>::value
                   >::value);
        };

 

        template< typename T > struct is_reference 
            : public integral_constant<bool, is_reference_impl<T>::value> 
        { 
        public:
        }; 
        
        template <class T>
        struct add_reference
        {
            typedef typename std::remove_reference<T>::type& type;
        };

    } // namespace detail
} // namespace cl
} // namespace bolt


#endif
