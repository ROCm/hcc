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

// (C) Copyright David Abrahams 2002.
// (C) Copyright Jeremy Siek    2002.
// (C) Copyright Thomas Witt    2002.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file BOOST_LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOLT_ITERATOR_FACADE_H
#define BOLT_ITERATOR_FACADE_H

#include <type_traits>
#include <bolt/cl/iterator/iterator_categories.h>
#include <bolt/cl/iterator/facade_iterator_category.h>

namespace bolt
{
namespace cl
{
  // This forward declaration is required for the friend declaration
  // in iterator_core_access
  template <class I, class V, class TC, class R, class D> class iterator_facade;

  namespace detail
  {
    // A binary metafunction class that always returns bool.  VC6
    // ICEs on mpl::always<bool>, probably because of the default
    // parameters.
    struct always_bool2
    {
        template <class T, class U>
        struct apply
        {
            typedef bool type;
        };
    };

    //
    // enable if for use in operator implementation.
    //
    template <
        class Facade1
      , class Facade2
      , class Return
    >
    struct enable_if_interoperable
       : std::enable_if<
           bolt::cl::detail::or_<
               std::is_convertible<Facade1, Facade2>
             , std::is_convertible<Facade2, Facade1>
           >::value
         , Return
        >
    {};

    template<typename Facade1, typename Facade2>
      struct distance_from_result
        : bolt::cl::detail::eval_if<
            std::is_convertible<Facade2,Facade1>,
            identity_<typename Facade1::difference_type>,
            identity_<typename Facade2::difference_type>
          >
    {};


        

    // A proxy return type for operator[], needed to deal with
    // iterators that may invalidate referents upon destruction.
    // Consider the temporary iterator in *(a + n)
    template <class Iterator>
    class operator_brackets_proxy
    {
        // Iterator is actually an iterator_facade, so we do not have to
        // go through iterator_traits to access the traits.
        typedef typename Iterator::reference  reference;
        typedef typename Iterator::value_type value_type;

     public:
        operator_brackets_proxy(Iterator const& iter)
          : m_iter(iter)
        {}

        operator reference() const
        {
            return *m_iter;
        }

        operator_brackets_proxy& operator=(value_type const& val)
        {
            *m_iter = val;
            return *this;
        }

     private:
        Iterator m_iter;
    };
        

    struct choose_difference_type
    {
        template <class I1, class I2>
        struct apply
        : bolt::cl::detail::eval_if<
              std::is_convertible<I2,I1>
            , bolt::cl::iterator_difference<I1>
            , bolt::cl::iterator_difference<I2>
          >
        {};

    };
  } // namespace detail


#  define BOLT_ITERATOR_FACADE_INTEROP_HEAD(prefix, op)   \
    template <                                                          \
        class Derived1, class V1, class TC1, class Reference1, class Difference1 \
      , class Derived2, class V2, class TC2, class Reference2, class Difference2 \
    >                                                                   \
    prefix bool                                                         \
    operator op(                                                        \
        iterator_facade<Derived1, V1, TC1, Reference1, Difference1> const& lhs   \
      , iterator_facade<Derived2, V2, TC2, Reference2, Difference2> const& rhs)


  //
  // Helper class for granting access to the iterator core interface.
  //
  // The simple core interface is used by iterator_facade. The core
  // interface of a user/library defined iterator type should not be made public
  // so that it does not clutter the public interface. Instead iterator_core_access
  // should be made friend so that iterator_facade can access the core
  // interface through iterator_core_access.
  //
  class iterator_core_access
  {
      template <class I, class V, class TC, class R, class D> friend class iterator_facade;

#  define BOLT_ITERATOR_FACADE_RELATION(op)                                \
      BOLT_ITERATOR_FACADE_INTEROP_HEAD(friend,op);

      BOLT_ITERATOR_FACADE_RELATION(==)
      BOLT_ITERATOR_FACADE_RELATION(!=)

      BOLT_ITERATOR_FACADE_RELATION(<)
      BOLT_ITERATOR_FACADE_RELATION(>)
      BOLT_ITERATOR_FACADE_RELATION(<=)
      BOLT_ITERATOR_FACADE_RELATION(>=)
#  undef BOLT_ITERATOR_FACADE_RELATION

    template <                                                          
        class Derived1, class V1, class TC1, class Reference1, class Difference1 
        , class Derived2, class V2, class TC2, class Reference2, class Difference2 
    >                                                                   
    friend typename bolt::cl::detail::distance_from_result<
        bolt::cl::iterator_facade<Derived1, V1, TC1, Reference1, Difference1 >,
        bolt::cl::iterator_facade<Derived2, V2, TC2, Reference2, Difference2 >
      >::type                                                             
    operator -(                                                        
        iterator_facade<Derived1, V1, TC1, Reference1, Difference1> const& lhs   
        , iterator_facade<Derived2, V2, TC2, Reference2, Difference2> const& rhs);


    template <class Derived, class V, class TC, class R, class D>   
    friend inline Derived operator+ (iterator_facade<Derived, V, TC, R, D> const&
           , typename Derived::difference_type);

    template <class Derived, class V, class TC, class R, class D>   
    friend inline Derived operator+ (typename Derived::difference_type
           , iterator_facade<Derived, V, TC, R, D> const&);

      template <class Facade>
      static typename Facade::reference dereference(Facade const& f)
      {
          return f.dereference();
      }

      template <class Facade>
      static void increment(Facade& f)
      {
          f.increment();
      }

      template <class Facade>
      static void decrement(Facade& f)
      {
          f.decrement();
      }

      template <class Facade1, class Facade2>
      static bool equal(Facade1 const& f1, Facade2 const& f2)
      {
          return f1.equal(f2);
      }

      template <class Facade1, class Facade2>
      static bool equal(Facade1 const& f1, Facade2 const& f2, std::true_type)
      {
          return f1.equal(f2);
      }

      template <class Facade1, class Facade2>
      static bool equal(Facade1 const& f1, Facade2 const& f2, std::false_type)
      {
          return f2.equal(f1);
      }

      template <class Facade>
      static void advance(Facade& f, typename Facade::difference_type n)
      {
          f.advance(n);
      }

      template <class Facade1, class Facade2>
      static typename Facade1::difference_type distance_from(
          Facade1 const& f1, Facade2 const& f2, std::true_type)
      {
          return -f1.distance_to(f2);
      }

      template <class Facade1, class Facade2>
      static typename Facade2::difference_type distance_from(
          Facade1 const& f1, Facade2 const& f2, std::false_type)
      {
          return f2.distance_to(f1);
      }

      //
      // Curiously Recurring Template interface.
      //
      template <class I, class V, class TC, class R, class D>
      static I& derived(bolt::cl::iterator_facade<I,V,TC,R,D>& facade)
      {
          return *static_cast<I*>(&facade);
      }

      template <class I, class V, class TC, class R, class D>
      static I const& derived(bolt::cl::iterator_facade<I,V,TC,R,D> const& facade)
      {
          return *static_cast<I const*>(&facade);
      }

   private:
      // objects of this class are useless
      iterator_core_access(); //undefined
  }; // end class iterator_core_access

  //
  // iterator_facade - use as a public base class for defining new
  // standard-conforming iterators.
  //
  template <
      class Derived             // The derived iterator type being constructed
    , class Value
    , class CategoryOrTraversal
    , class Reference   = Value&
    , class Difference  = std::ptrdiff_t
  >
  class iterator_facade
  {
   private:
      //
      // Curiously Recurring Template interface.
      //
      Derived& derived()
      {
          return *static_cast<Derived*>(this);
      }

      Derived const& derived() const
      {
          return *static_cast<Derived const*>(this);
      }

   protected:
      // For use by derived classes
      typedef iterator_facade<Derived,Value,CategoryOrTraversal,Reference,Difference> iterator_facade_;
      
   public:

      typedef typename std::remove_const<Value>::type value_type;
      typedef Reference reference;
      typedef Difference difference_type;
      typedef typename std::add_pointer<value_type>::type pointer;

      typedef typename bolt::cl::detail::facade_iterator_category< CategoryOrTraversal, Value, Reference >::type iterator_category;

      reference operator*() const
      {
          return iterator_core_access::dereference(this->derived());
      }

      pointer operator->() const
      {
          return this->derived();
      }

      reference
      operator[](difference_type n) const
      {
          return *(this->derived() + n);
      }

      Derived& operator++()
      {
          iterator_core_access::increment(this->derived());
          return this->derived();
      }

      Derived
      operator++(int)
      {
          Derived tmp(this->derived());
          ++*this;
          return tmp;
      }
      
      Derived& operator--()
      {
          iterator_core_access::decrement(this->derived());
          return this->derived();
      }

      Derived operator--(int)
      {
          Derived tmp(this->derived());
          --*this;
          return tmp;
      }

      Derived& operator+=(difference_type n)
      {
          iterator_core_access::advance(this->derived(), n);
          return this->derived();
      }

      Derived& operator-=(difference_type n)
      {
          iterator_core_access::advance(this->derived(), -n);
          return this->derived();
      }

      Derived operator-(difference_type x) const
      {
          Derived result(this->derived());
          return result -= x;
      }

  }; // end of iterator_facade

  
  //
  // Comparison operator implementation. The library supplied operators
  // enables the user to provide fully interoperable constant/mutable
  // iterator types. I.e. the library provides all operators
  // for all mutable/constant iterator combinations.
  //
  // Note though that this kind of interoperability for constant/mutable
  // iterators is not required by the standard for container iterators.
  // All the standard asks for is a conversion mutable -> constant.
  // Most standard library implementations nowadays provide fully interoperable
  // iterator implementations, but there are still heavily used implementations
  // that do not provide them. (Actually it's even worse, they do not provide
  // them for only a few iterators.)
  //
  // ?? Maybe a BOOST_ITERATOR_NO_FULL_INTEROPERABILITY macro should
  //    enable the user to turn off mixed type operators
  //
  // The library takes care to provide only the right operator overloads.
  // I.e.
  //
  // bool operator==(Iterator,      Iterator);
  // bool operator==(ConstIterator, Iterator);
  // bool operator==(Iterator,      ConstIterator);
  // bool operator==(ConstIterator, ConstIterator);
  //
  //   ...
  //
  // In order to do so it uses c++ idioms that are not yet widely supported
  // by current compiler releases. The library is designed to degrade gracefully
  // in the face of compiler deficiencies. In general compiler
  // deficiencies result in less strict error checking and more obscure
  // error messages, functionality is not affected.
  //
  // For full operation compiler support for "Substitution Failure Is Not An Error"
  // (aka. enable_if) and boost::is_convertible is required.
  //
  // The following problems occur if support is lacking.
  //
  // Pseudo code
  //
  // ---------------
  // AdaptorA<Iterator1> a1;
  // AdaptorA<Iterator2> a2;
  //
  // // This will result in a no such overload error in full operation
  // // If enable_if or is_convertible is not supported
  // // The instantiation will fail with an error hopefully indicating that
  // // there is no operator== for Iterator1, Iterator2
  // // The same will happen if no enable_if is used to remove
  // // false overloads from the templated conversion constructor
  // // of AdaptorA.
  //
  // a1 == a2;
  // ----------------
  //
  // AdaptorA<Iterator> a;
  // AdaptorB<Iterator> b;
  //
  // // This will result in a no such overload error in full operation
  // // If enable_if is not supported the static assert used
  // // in the operator implementation will fail.
  // // This will accidently work if is_convertible is not supported.
  //
  // a == b;
  // ----------------
  //

# define BOLT_ITERATOR_FACADE_INTEROP(op, return_prefix, base_op)  \
  BOLT_ITERATOR_FACADE_INTEROP_HEAD(inline, op)                    \
  {                                                                             \
      /* For those compilers that do not support enable_if */                   \
      return_prefix iterator_core_access::base_op(                              \
          *static_cast<Derived1 const*>(&lhs)                                   \
        , *static_cast<Derived2 const*>(&rhs)                                   \
        , std::is_convertible<Derived2,Derived1>()                              \
      );                                                                        \
  }

# define BOLT_ITERATOR_FACADE_RELATION(op, return_prefix, base_op)              \
  BOLT_ITERATOR_FACADE_INTEROP(                                                 \
      op                                                                        \
    , return_prefix                                                             \
    , base_op                                                                   \
  )

  BOLT_ITERATOR_FACADE_RELATION(==, return, equal);
  BOLT_ITERATOR_FACADE_RELATION(!=, return !, equal);
  BOLT_ITERATOR_FACADE_RELATION(<, return 0 >, distance_from);
  BOLT_ITERATOR_FACADE_RELATION(>, return 0 <, distance_from);
  BOLT_ITERATOR_FACADE_RELATION(<=, return 0 >=, distance_from);
  BOLT_ITERATOR_FACADE_RELATION(>=, return 0 <=, distance_from);

# undef BOLT_ITERATOR_FACADE_RELATION

  
    template <                                                          
          class Derived1, class V1, class TC1, class Reference1, class Difference1 
        , class Derived2, class V2, class TC2, class Reference2, class Difference2 
    >                                                                   
    inline typename bolt::cl::detail::distance_from_result<
        bolt::cl::iterator_facade<Derived1, V1, TC1, Reference1, Difference1 >,
        bolt::cl::iterator_facade<Derived2, V2, TC2, Reference2, Difference2 >
      >::type                                                             
    operator -(                                                        
        iterator_facade<Derived1, V1, TC1, Reference1, Difference1> const& lhs   
        , iterator_facade<Derived2, V2, TC2, Reference2, Difference2> const& rhs)
    {
      return iterator_core_access::distance_from(                     
          *static_cast<Derived1 const*>(&lhs)                          
        , *static_cast<Derived2 const*>(&rhs)                          
        , std::is_convertible<Derived2,Derived1>()
      );                                                               
    }

# undef BOLT_ITERATOR_FACADE_INTEROP
# undef BOLT_ITERATOR_FACADE_INTEROP_HEAD

    template <class Derived, class V, class TC, class R, class D>   
    inline Derived operator+ (bolt::cl::iterator_facade<Derived, V, TC, R, D> const& i, 
                                typename Derived::difference_type n )
    {                                                 
        Derived tmp(static_cast<Derived const&>(i));  
        return tmp += n;                              
    }

    template <class Derived, class V, class TC, class R, class D>   \
    inline Derived operator+ ( typename Derived::difference_type n, 
                               bolt::cl::iterator_facade<Derived, V, TC, R, D> const& i)     
      {                                                 
          Derived tmp(static_cast<Derived const&>(i));  
          return tmp += n;                              
      }

} // namespace cl
} // namespace bolt


#endif // BOLT_ITERATOR_FACADE_H
