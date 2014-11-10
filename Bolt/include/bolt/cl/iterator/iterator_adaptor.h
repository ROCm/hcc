// (C) Copyright David Abrahams 2002.
// (C) Copyright Jeremy Siek    2002.
// (C) Copyright Thomas Witt    2002.
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

// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file BOOST_LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOLT_ITERATOR_ADAPTOR_H
#define BOLT_ITERATOR_ADAPTOR_H

#include <type_traits>
#include <bolt/cl/detail/type_traits.h>
#include <bolt/cl/iterator/iterator_facade.h>
#include <bolt/cl/iterator/iterator_categories.h>
#include <bolt/cl/iterator/facade_iterator_category.h>
namespace bolt
{
namespace cl
{
  // Used as a default template argument internally, merely to
  // indicate "use the default", this can also be passed by users
  // explicitly in order to specify that the default should be used.
  struct use_default;
  
//# ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
//  // the incompleteness of use_default causes massive problems for
//  // is_convertible (naturally).  This workaround is fortunately not
//  // needed for vc6/vc7.
//  template<class To>
//  struct is_convertible<use_default,To>
//    : mpl::false_ {};
//# endif 
  
  namespace detail
  {

    // 
    // Result type used in enable_if_convertible meta function.
    // This can be an incomplete type, as only pointers to 
    // enable_if_convertible< ... >::type are used.
    // We could have used void for this, but conversion to
    // void* is just to easy.
    //
    struct enable_type;
  }


  //
  // enable_if for use in adapted iterators constructors.
  //
  // In order to provide interoperability between adapted constant and
  // mutable iterators, adapted iterators will usually provide templated
  // conversion constructors of the following form
  //
  // template <class BaseIterator>
  // class adapted_iterator :
  //   public iterator_adaptor< adapted_iterator<Iterator>, Iterator >
  // {
  // public:
  //   
  //   ...
  //
  //   template <class OtherIterator>
  //   adapted_iterator(
  //       OtherIterator const& it
  //     , typename enable_if_convertible<OtherIterator, Iterator>::type* = 0);
  //
  //   ...
  // };
  //
  // enable_if_convertible is used to remove those overloads from the overload
  // set that cannot be instantiated. For all practical purposes only overloads
  // for constant/mutable interaction will remain. This has the advantage that
  // meta functions like boost::is_convertible do not return false positives,
  // as they can only look at the signature of the conversion constructor
  // and not at the actual instantiation.
  //
  // enable_if_interoperable can be safely used in user code. It falls back to
  // always enabled for compilers that don't support enable_if or is_convertible. 
  // There is no need for compiler specific workarounds in user code. 
  //
  // The operators implementation relies on boost::is_convertible not returning
  // false positives for user/library defined iterator types. See comments
  // on operator implementation for consequences.
  //

//#  if BOOST_WORKAROUND(BOOST_MSVC, <= 1300)
//  
//  template<typename From, typename To>
//  struct enable_if_convertible
//  {
//     typedef typename mpl::if_<
//         mpl::or_<
//             is_same<From,To>
//           , is_convertible<From, To>
//         >
//      , boost::detail::enable_type
//      , int&
//     >::type type;
//  };
//  
//#  elif defined(BOOST_NO_IS_CONVERTIBLE) || defined(BOOST_NO_SFINAE)
//  
//  template <class From, class To>
//  struct enable_if_convertible
//  {
//      typedef boost::detail::enable_type type;
//  };
//  
//#  elif BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(13102292)) && BOOST_MSVC > 1300
//  
//  // For some reason vc7.1 needs us to "cut off" instantiation
//  // of is_convertible in a few cases.
//  template<typename From, typename To>
//  struct enable_if_convertible
//    : iterators::enable_if<
//        mpl::or_<
//            is_same<From,To>
//          , is_convertible<From, To>
//        >
//      , boost::detail::enable_type
//    >
//  {};
//  
//#  else 
//  
  template<typename From, typename To>
  struct enable_if_convertible
    : std::enable_if<
          std::is_convertible<From, To>::value
        , bolt::cl::detail::enable_type
      >
  {};
//      
//# endif
  
  //
  // Default template argument handling for iterator_adaptor
  //
  namespace detail
  {
    // If T is use_default, return the result of invoking
    // DefaultNullaryFn, otherwise return T.
    template <class T, class DefaultNullaryFn>
    struct ia_dflt_help
      : bolt::cl::detail::eval_if<
            bolt::cl::detail::is_same<T, use_default>
          , DefaultNullaryFn
          , bolt::cl::detail::identity_<T>
        >
    {
    };

    // A metafunction which computes an iterator_adaptor's base class,
    // a specialization of iterator_facade.
    template <
        class Derived
      , class Base
      , class Value
      , class Traversal
      , class Reference
      , class Difference
    >
    struct iterator_adaptor_base
    {
        typedef iterator_facade<
            Derived
            //TODO - Replace bolt::cl with std namespace. 
////# ifdef BOOST_ITERATOR_REF_CONSTNESS_KILLS_WRITABILITY
          , typename bolt::cl::detail::ia_dflt_help<
                Value
              , bolt::cl::detail::eval_if<
                    std::is_same<Reference,use_default>
                  , bolt::cl::iterator_value<Base>
                  , std::remove_reference<Reference>
                >
            >::type
///*# else
//          , typename boost::detail::ia_dflt_help<
//                Value, iterator_value<Base>
//            >::type
//# endif
//            
          , typename bolt::cl::detail::ia_dflt_help<
                Traversal
              , bolt::cl::iterator_traversal<Base>
            >::type

          , typename bolt::cl::detail::ia_dflt_help<
                Reference
              , bolt::cl::detail::eval_if<
                    std::is_same<Value,use_default>
                  , bolt::cl::iterator_reference<Base>
                  , bolt::cl::detail::add_reference<Value>
                >
            >::type

          , typename bolt::cl::detail::ia_dflt_help<
                Difference, bolt::cl::iterator_difference<Base>
            >::type
        >
        type;
    };
  }// namespace detail
  
  //
  // Iterator Adaptor
  //
  // The parameter ordering changed slightly with respect to former
  // versions of iterator_adaptor The idea is that when the user needs
  // to fiddle with the reference type it is highly likely that the
  // iterator category has to be adjusted as well.  Any of the
  // following four template arguments may be ommitted or explicitly
  // replaced by use_default.
  //
  //   Value - if supplied, the value_type of the resulting iterator, unless
  //      const. If const, a conforming compiler strips constness for the
  //      value_type. If not supplied, iterator_traits<Base>::value_type is used
  //
  //   Category - the traversal category of the resulting iterator. If not
  //      supplied, iterator_traversal<Base>::type is used.
  //
  //   Reference - the reference type of the resulting iterator, and in
  //      particular, the result type of operator*(). If not supplied but
  //      Value is supplied, Value& is used. Otherwise
  //      iterator_traits<Base>::reference is used.
  //
  //   Difference - the difference_type of the resulting iterator. If not
  //      supplied, iterator_traits<Base>::difference_type is used.
  //
  template <
      class Derived
    , class Base
    , class Value        = use_default
    , class Traversal    = use_default
    , class Reference    = use_default
    , class Difference   = use_default
  >
  class iterator_adaptor
    : public bolt::cl::detail::iterator_adaptor_base<
        Derived, Base, Value, Traversal, Reference, Difference
      >::type
  {
      friend class bolt::cl::iterator_core_access;

   protected:
      typedef typename bolt::cl::detail::iterator_adaptor_base<
          Derived, Base, Value, Traversal, Reference, Difference
      >::type super_t;
   public:
      iterator_adaptor() {}

      explicit iterator_adaptor(Base const &iter)
          : m_iterator(iter)
      {
      }

      typedef Base base_type;

      Base const& base() const
        { return m_iterator; }

   protected:
      // for convenience in derived classes
      typedef iterator_adaptor<Derived,Base,Value,Traversal,Reference,Difference> iterator_adaptor_;
      
      //
      // lvalue access to the Base object for Derived
      //
      Base const& base_reference() const
        { return m_iterator; }

      Base& base_reference()
        { return m_iterator; }

   private:
      //
      // Core iterator interface for iterator_facade.  This is private
      // to prevent temptation for Derived classes to use it, which
      // will often result in an error.  Derived classes should use
      // base_reference(), above, to get direct access to m_iterator.
      // 
      typename super_t::reference dereference() const
        { return *m_iterator; }

      template <
      class OtherDerived, class OtherIterator, class V, class C, class R, class D
      >   
      bool equal(iterator_adaptor<OtherDerived, OtherIterator, V, C, R, D> const& x) const
      {
          return m_iterator == x.base();
      }

      void advance(typename super_t::difference_type n)
      {
          m_iterator += (int)n;
      }
  
      void increment() { ++m_iterator; }

      void decrement() { --m_iterator; }

      template <
          class OtherDerived, class OtherIterator, class V, class C, class R, class D
      >   
      typename super_t::difference_type distance_to(
          iterator_adaptor<OtherDerived, OtherIterator, V, C, R, D> const& y) const
      {
          return y.base() - m_iterator;
      }

   private: // data members
      Base m_iterator;
  };
} // namespace cl
} // namespace bolt



#endif // BOLT_ITERATOR_ADAPTOR_H
