using namespace std;
/******************************************************************************
 *  User Defined Data Types - vec2,4,8
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == CL_BENCH
   BOLT_FUNCTOR(vec2,
   struct vec2
    {
        DATA_TYPE a, b;
        vec2  operator =(const DATA_TYPE inp)
        {
            vec2 tmp;
            a = b = tmp.a = tmp.b = inp;
            return tmp;
        }
        vec2 operator =(const vec2 inp)
        {
            vec2 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            return tmp;
        }
        vec2 operator +(const DATA_TYPE inp)
        {  
            vec2 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            return tmp;
        }
        bool operator==(const vec2& rhs) const
        {
            bool l_equal = true;
            l_equal = ( a == rhs.a ) ? l_equal : false;
            l_equal = ( b == rhs.b ) ? l_equal : false;
            return l_equal;
        }
     // friend ostream& operator<<(ostream& os, const vec2& dt);
    };
  );
    BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec2 >::iterator );
    BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );
    BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec2 );
    //BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, DATA_TYPE, vec2 );

    BOLT_FUNCTOR(vec4,
    struct vec4
    {
        DATA_TYPE a, b, c, d;
        vec4  operator =(const DATA_TYPE inp)
        {
        vec4 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=inp;
        return tmp;
        }
        vec4 operator =(const vec4 inp)
        {
            vec4 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            tmp.c = c = inp.c;
            tmp.d = d = inp.d;
            return tmp;
        }
        vec4 operator +(const DATA_TYPE inp)
        {  
            vec4 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            tmp.c = c = c+inp;
            tmp.d = d = d+inp;
            return tmp;
        }
        bool operator==(const vec4& rhs) const
        {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        return l_equal;
        }
        //friend ostream& operator<<(ostream& os, const vec4& dt);
    };
    );
    BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec4 >::iterator );
    BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );
    BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec4 );
    ///*BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec2 >::iterator );
    //BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );*/
    //BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec4 );
    //BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, DATA_TYPE, vec4 );


    BOLT_FUNCTOR(vec8,
    struct vec8
    {
        DATA_TYPE a, b, c, d, e, f, g, h;
        vec8  operator =(const DATA_TYPE inp)
        {
        a = b = c=d=e=f=g=h=inp;
        vec8 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=e=f=g=h=inp;
        tmp.e = tmp.f = tmp.g = tmp.h = inp;
        return tmp;
        }
        vec8 operator =(const vec8 inp)
        {
            vec8 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            tmp.c = c = inp.c;
            tmp.d = d = inp.d;
            tmp.e = e = inp.e;
            tmp.f = f = inp.f;
            tmp.g = g = inp.g;
            tmp.h = h = inp.h;
            return tmp;
        }        
        vec8 operator +(const DATA_TYPE inp)
        {  
            vec8 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            tmp.c = c = c+inp;
            tmp.d = d = d+inp;
            tmp.e = e = e+inp;
            tmp.f = f = f+inp;
            tmp.g = g = g+inp;
            tmp.h = h = h+inp;
            return tmp;
        }
        bool operator==(const vec8& rhs) const
        {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        l_equal = ( e == rhs.e ) ? l_equal : false;
        l_equal = ( f == rhs.f ) ? l_equal : false;
        l_equal = ( g == rhs.g ) ? l_equal : false;
        l_equal = ( h == rhs.h ) ? l_equal : false;
        return l_equal;
        }
        //friend ostream& operator<<(ostream& os, const vec8& dt);
    };
    );
        BOLT_CREATE_TYPENAME( bolt::cl::device_vector< vec8 >::iterator );
        BOLT_CREATE_CLCODE( bolt::cl::device_vector< vec8 >::iterator, bolt::cl::deviceVectorIteratorTemplate );
        BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, DATA_TYPE, vec8 );
        BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, DATA_TYPE, vec8 );


#elif BENCHMARK_CL_AMP == AMP_BENCH
   struct vec2
    {
        
        DATA_TYPE a, b;
        vec2  operator =(const DATA_TYPE inp) restrict(cpu,amp)
        {
            vec2 tmp;
            a = b = tmp.a = tmp.b = inp;
            return tmp;
        }
        vec2 operator =(const vec2 inp) restrict(cpu,amp)
        {
            vec2 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            return tmp;
        }
        vec2 operator +(const DATA_TYPE inp) restrict(cpu,amp)
        {  
            vec2 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            return tmp;
        }
        bool operator==(const vec2& rhs) const restrict(cpu,amp)
        {
            bool l_equal = true;
            l_equal = ( a == rhs.a ) ? l_equal : false;
            l_equal = ( b == rhs.b ) ? l_equal : false;
            return l_equal;
        }

      friend ostream& operator<<(ostream& os, const vec2& dt);
    };
    struct vec4
    {       
        DATA_TYPE a, b, c, d;
        vec4  operator =(const DATA_TYPE inp) restrict(cpu,amp)
        {
        vec4 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=inp;
        return tmp;
        }
        vec4 operator =(const vec4 inp) restrict(cpu,amp)
        {
            vec4 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            tmp.c = c = inp.c;
            tmp.d = d = inp.d;
            return tmp;
        }
        vec4 operator +(const DATA_TYPE inp) restrict(cpu,amp)
        {  
            vec4 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            tmp.c = c = c+inp;
            tmp.d = d = d+inp;
            return tmp;
        }
        bool operator==(const vec4& rhs) const restrict(cpu,amp)
        {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        return l_equal;
        }
        friend ostream& operator<<(ostream& os, const vec4& dt);
    };

    struct vec8
    {
        DATA_TYPE a, b, c, d, e, f, g, h;
        vec8  operator =(const DATA_TYPE inp) restrict(cpu,amp)
        {
        a = b = c=d=e=f=g=h=inp;
        vec8 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=e=f=g=h=inp;
        tmp.e = tmp.f = tmp.g = tmp.h = inp;
        return tmp;
        }
        vec8 operator =(const vec8 inp) restrict(cpu,amp)
        {
            vec8 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            tmp.c = c = inp.c;
            tmp.d = d = inp.d;
            tmp.e = e = inp.e;
            tmp.f = f = inp.f;
            tmp.g = g = inp.g;
            tmp.h = h = inp.h;
            return tmp;
        }        
        vec8 operator +(const DATA_TYPE inp) restrict(cpu,amp)
        {  
            vec8 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            tmp.c = c = c+inp;
            tmp.d = d = d+inp;
            tmp.e = e = e+inp;
            tmp.f = f = f+inp;
            tmp.g = g = g+inp;
            tmp.h = h = h+inp;
            return tmp;
        }
        bool operator==(const vec8& rhs) const restrict(cpu,amp)
        {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        l_equal = ( e == rhs.e ) ? l_equal : false;
        l_equal = ( f == rhs.f ) ? l_equal : false;
        l_equal = ( g == rhs.g ) ? l_equal : false;
        l_equal = ( h == rhs.h ) ? l_equal : false;
        return l_equal;
        }
        friend ostream& operator<<(ostream& os, const vec8& dt);
    };

#endif

      ostream& operator<<(ostream& os, const vec2& dt)
        {
        os<<dt.a<<" "<<dt.b;
        return os;
        }


    ostream& operator<<(ostream& os, const vec4& dt)
        {
        os<<dt.a<<" "<<dt.b<<" "<<dt.c<<" "<<dt.d;
        return os;
        }
        ostream& operator<<(ostream& os, const vec8& dt)
        {
        os<<dt.a<<" "<<dt.b<<" "<<dt.c<<" "<<dt.d<<" "<<dt.e<<" "<<dt.f<<" "<<dt.g<<" "<<dt.h;
        return os;
        }

#else
    struct vec2
        {
            DATA_TYPE a, b;
            __host__ __device__
            vec2  operator =(const DATA_TYPE inp)
            {
            vec2 tmp;
            a = b = tmp.a = tmp.b = inp;
            return tmp;
            }
            vec2 operator =(const vec2 inp)
             {
            vec2 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            return tmp;
             }
            vec2 operator +(const DATA_TYPE inp)
            {  
                vec2 tmp;         
                tmp.a = a = a+inp;
                tmp.b = b = b+inp;
                return tmp;
            }
            bool operator==(const vec2& rhs) const
            {
            bool l_equal = true;
            l_equal = ( a == rhs.a ) ? l_equal : false;
            l_equal = ( b == rhs.b ) ? l_equal : false;
            return l_equal;
            }

        };
    struct vec4
        {
            DATA_TYPE a, b, c, d;
            __host__ __device__
            vec4  operator =(const DATA_TYPE inp)
            {
            vec4 tmp;
            tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=inp;
            return tmp;
            }
            vec4 operator =(const vec4 inp)
            {
                vec4 tmp;
                tmp.a = a = inp.a;
                tmp.b = b = inp.b;
                tmp.c = c = inp.c;
                tmp.d = d = inp.d;
                return tmp;
            }
            vec4 operator +(const DATA_TYPE inp)
            {  
                vec4 tmp;         
                tmp.a = a = a+inp;
                tmp.b = b = b+inp;
                tmp.c = c = c+inp;
                tmp.d = d = d+inp;
                return tmp;
            }
            bool operator==(const vec4& rhs) const
            {
            bool l_equal = true;
            l_equal = ( a == rhs.a ) ? l_equal : false;
            l_equal = ( b == rhs.b ) ? l_equal : false;
            l_equal = ( c == rhs.c ) ? l_equal : false;
            l_equal = ( d == rhs.d ) ? l_equal : false;
            return l_equal;
            }
        };
    struct vec8
        {
            DATA_TYPE a, b, c, d, e, f, g, h;
            __host__ __device__
        vec8  operator =(const DATA_TYPE inp)
        {
        a = b = c=d=e=f=g=h=inp;
        vec8 tmp;
        tmp.a = tmp.b = tmp.c = tmp.d = a = b = c=d=e=f=g=h=inp;
        tmp.e = tmp.f = tmp.g = tmp.h = inp;
        return tmp;
        }
        vec8 operator =(const vec8 inp)
        {
            vec8 tmp;
            tmp.a = a = inp.a;
            tmp.b = b = inp.b;
            tmp.c = c = inp.c;
            tmp.d = d = inp.d;
            tmp.e = e = inp.e;
            tmp.f = f = inp.f;
            tmp.g = g = inp.g;
            tmp.h = h = inp.h;
            return tmp;
        }        
        vec8 operator +(const DATA_TYPE inp)
        {  
            vec8 tmp;         
            tmp.a = a = a+inp;
            tmp.b = b = b+inp;
            tmp.c = c = c+inp;
            tmp.d = d = d+inp;
            tmp.e = e = e+inp;
            tmp.f = f = f+inp;
            tmp.g = g = g+inp;
            tmp.h = h = h+inp;
            return tmp;
        }
        bool operator==(const vec8& rhs) const
        {
        bool l_equal = true;
        l_equal = ( a == rhs.a ) ? l_equal : false;
        l_equal = ( b == rhs.b ) ? l_equal : false;
        l_equal = ( c == rhs.c ) ? l_equal : false;
        l_equal = ( d == rhs.d ) ? l_equal : false;
        l_equal = ( e == rhs.e ) ? l_equal : false;
        l_equal = ( f == rhs.f ) ? l_equal : false;
        l_equal = ( g == rhs.g ) ? l_equal : false;
        l_equal = ( h == rhs.h ) ? l_equal : false;
        return l_equal;
        }
        };
#endif


    
/******************************************************************************
 *  User Defined Binary Functions - DATA_TYPE plus Only for thrust usage
 *****************************************************************************/
#if (BOLT_BENCHMARK == 0)
    struct vec1plus
        {
                    __host__ __device__
            DATA_TYPE operator()(const DATA_TYPE &lhs, const DATA_TYPE &rhs) const
            {
            DATA_TYPE l_result;
            l_result = lhs + rhs;
            return l_result;
            };
        }; 
#endif
/******************************************************************************
 *  User Defined Binary Functions - vec2,4,8plus
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)

#if BENCHMARK_CL_AMP == CL_BENCH
    BOLT_FUNCTOR(vec2plus,
    struct vec2plus
    {
        vec2 operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        vec2 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        return l_result;
        };
    };
    );

    BOLT_FUNCTOR(vec4plus,
    struct vec4plus
    {
        vec4 operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        vec4 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        return l_result;
        };
    }; 
    );
    BOLT_FUNCTOR(vec8plus,
    struct vec8plus
    {
        vec8 operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        vec8 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        l_result.e = lhs.e+rhs.e;
        l_result.f = lhs.f+rhs.f;
        l_result.g = lhs.g+rhs.g;
        l_result.h = lhs.h+rhs.h;
        return l_result;
        };
    };
    );
#elif BENCHMARK_CL_AMP == AMP_BENCH
   struct vec2plus
    {
        vec2 operator()(const vec2 &lhs, const vec2 &rhs) const restrict(cpu,amp)
        {
        vec2 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        return l_result;
        };
    };
        
    struct vec4plus
    {
        vec4 operator()(const vec4 &lhs, const vec4 &rhs) const restrict(cpu,amp)
        {
        vec4 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        return l_result;
        };
    }; 

    struct vec8plus
    {
        vec8 operator()(const vec8 &lhs, const vec8 &rhs) const restrict(cpu,amp)
        {
        vec8 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        l_result.e = lhs.e+rhs.e;
        l_result.f = lhs.f+rhs.f;
        l_result.g = lhs.g+rhs.g;
        l_result.h = lhs.h+rhs.h;
        return l_result;
        };
    };
#endif
#else
    struct vec2plus
    {
        __host__ __device__
        vec2 operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        vec2 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        return l_result;
        }
    }; 
    struct vec4plus
    {
        __host__ __device__
        vec4 operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        vec4 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        return l_result;
        };
    }; 
    struct vec8plus
    {
        __host__ __device__
        vec8 operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        vec8 l_result;
        l_result.a = lhs.a+rhs.a;
        l_result.b = lhs.b+rhs.b;
        l_result.c = lhs.c+rhs.c;
        l_result.d = lhs.d+rhs.d;
        l_result.e = lhs.e+rhs.e;
        l_result.f = lhs.f+rhs.f;
        l_result.g = lhs.g+rhs.g;
        l_result.h = lhs.h+rhs.h;
        return l_result;
        };
}; 
#endif


    /******************************************************************************
 *  User Defined Binary Functions - DATA_TYPE mult Only for thrust usage
 *****************************************************************************/
#if (BOLT_BENCHMARK == 0)
    struct vec1mult
        {
                    __host__ __device__
            DATA_TYPE operator()(const DATA_TYPE &lhs, const DATA_TYPE &rhs) const
            {
            DATA_TYPE l_result;
            l_result = lhs * rhs;
            return l_result;
            };
        }; 
#endif
/******************************************************************************
 *  User Defined Binary Functions - vec2,4,8plus
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)

#if BENCHMARK_CL_AMP == CL_BENCH
    BOLT_FUNCTOR(vec2mult,
    struct vec2mult
    {
        vec2 operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        vec2 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        return l_result;
        };
    };
    );

    BOLT_FUNCTOR(vec4mult,
    struct vec4mult
    {
        vec4 operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        vec4 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        l_result.c = lhs.c*rhs.c;
        l_result.d = lhs.d*rhs.d;
        return l_result;
        };
    }; 
    );
    BOLT_FUNCTOR(vec8mult,
    struct vec8mult
    {
        vec8 operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        vec8 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        l_result.c = lhs.c*rhs.c;
        l_result.d = lhs.d*rhs.d;
        l_result.e = lhs.e*rhs.e;
        l_result.f = lhs.f*rhs.f;
        l_result.g = lhs.g*rhs.g;
        l_result.h = lhs.h*rhs.h;
        return l_result;
        };
    };
    );
#elif BENCHMARK_CL_AMP == AMP_BENCH
   struct vec2mult
    {
        vec2 operator()(const vec2 &lhs, const vec2 &rhs) const restrict(cpu,amp)
        {
        vec2 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        return l_result;
        };
    };
    

    
    struct vec4mult
    {
        vec4 operator()(const vec4 &lhs, const vec4 &rhs) const restrict(cpu,amp)
        {
        vec4 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        l_result.c = lhs.c*rhs.c;
        l_result.d = lhs.d*rhs.d;
        return l_result;
        };
    }; 

    struct vec8mult
    {
        vec8 operator()(const vec8 &lhs, const vec8 &rhs) const restrict(cpu,amp)
        {
        vec8 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        l_result.c = lhs.c*rhs.c;
        l_result.d = lhs.d*rhs.d;
        l_result.e = lhs.e*rhs.e;
        l_result.f = lhs.f*rhs.f;
        l_result.g = lhs.g*rhs.g;
        l_result.h = lhs.h*rhs.h;
        return l_result;
        };
    };
#endif
#else
    struct vec2mult
    {
        __host__ __device__
        vec2 operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        vec2 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        return l_result;
        }
    }; 
    struct vec4mult
    {
        __host__ __device__
        vec4 operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        vec4 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        l_result.c = lhs.c*rhs.c;
        l_result.d = lhs.d*rhs.d;
        return l_result;
        };
    }; 
    struct vec8mult
    {
        __host__ __device__
        vec8 operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        vec8 l_result;
        l_result.a = lhs.a*rhs.a;
        l_result.b = lhs.b*rhs.b;
        l_result.c = lhs.c*rhs.c;
        l_result.d = lhs.d*rhs.d;
        l_result.e = lhs.e*rhs.e;
        l_result.f = lhs.f*rhs.f;
        l_result.g = lhs.g*rhs.g;
        l_result.h = lhs.h*rhs.h;
        return l_result;
        };
}; 
#endif


/******************************************************************************
 *  User Defined Unary Functions-  DATA_TYPE square for thrust usage
 *****************************************************************************/
#if (BOLT_BENCHMARK == 0)
    struct vec1square
    {
            __host__ __device__
        DATA_TYPE operator()(const DATA_TYPE &rhs) const
        {
        DATA_TYPE l_result;
        l_result = rhs * rhs;
        return l_result;
        };
    }; 
#endif
/******************************************************************************
 *  User Defined Unary Functions-  vec2,4,8square
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == CL_BENCH
    BOLT_FUNCTOR(vec2square,
    struct vec2square
    {
        vec2 operator()(const vec2 &rhs) const
        {
        vec2 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        return l_result;
        };
    }; 
    );
    BOLT_FUNCTOR(vec4square,
    struct vec4square
    {
        vec4 operator()(const vec4 &rhs) const
        {
        vec4 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        return l_result;
        };
    }; 
    );
    BOLT_FUNCTOR(vec8square,
    struct vec8square
    {
        vec8 operator()(const vec8 &rhs) const
        {
        vec8 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        l_result.e = rhs.e*rhs.e;
        l_result.f = rhs.f*rhs.f;
        l_result.g = rhs.g*rhs.g;
        l_result.h = rhs.h*rhs.h;
        return l_result;
        };
    }; 
    );

#elif BENCHMARK_CL_AMP == AMP_BENCH
    struct vec2square
    {
        vec2 operator()(const vec2 &rhs) const restrict(cpu,amp)
        {
        vec2 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        return l_result;
        };
    }; 
    
    
    struct vec4square
    {
        vec4 operator()(const vec4 &rhs) const restrict(cpu,amp)
        {
        vec4 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        return l_result;
        };
    }; 

    struct vec8square
    {
        vec8 operator()(const vec8 &rhs) const restrict(cpu,amp)
        {
        vec8 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        l_result.e = rhs.e*rhs.e;
        l_result.f = rhs.f*rhs.f;
        l_result.g = rhs.g*rhs.g;
        l_result.h = rhs.h*rhs.h;
        return l_result;
        };
    }; 
#endif
#else
    struct vec2square
    {
            __host__ __device__
        vec2 operator()(const vec2 &rhs) const
        {
        vec2 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        return l_result;
        };
    }; 
    struct vec4square
    {
            __host__ __device__
        vec4 operator()(const vec4 &rhs) const
        {
        vec4 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        return l_result;
        };
    }; 
    struct vec8square
    {
            __host__ __device__
        vec8 operator()(const vec8 &rhs) const
        {
        vec8 l_result;
        l_result.a = rhs.a*rhs.a;
        l_result.b = rhs.b*rhs.b;
        l_result.c = rhs.c*rhs.c;
        l_result.d = rhs.d*rhs.d;
        l_result.e = rhs.e*rhs.e;
        l_result.f = rhs.f*rhs.f;
        l_result.g = rhs.g*rhs.g;
        l_result.h = rhs.h*rhs.h;
        return l_result;
        };
    }; 
#endif
#if (BOLT_BENCHMARK == 0)
/******************************************************************************
 *  User Defined Binary Predicates-  DATA_TYPE equal for thrust usage
 *****************************************************************************/
    struct vec1equal
    {
            __host__ __device__
        bool operator()(const DATA_TYPE &lhs, const DATA_TYPE &rhs) const
        {
        return lhs == rhs;
        };
    }; 
#endif
/******************************************************************************
 *  User Defined Binary Predicates- vec2,4,8equal   
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == CL_BENCH
    BOLT_FUNCTOR(vec2equal,
    struct vec2equal
    {
        bool operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        return lhs == rhs;
        };
    }; 
    );
    BOLT_FUNCTOR(vec4equal,

    struct vec4equal
    {
        bool operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        return lhs == rhs;
        };
    }; 

    );
    BOLT_FUNCTOR(vec8equal,
    struct vec8equal
    {
        bool operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        return lhs == rhs;
        };
    }; 

    );

#elif BENCHMARK_CL_AMP == AMP_BENCH
   struct vec2equal
    {
        bool operator()(const vec2 &lhs, const vec2 &rhs) const restrict(cpu,amp)
        {
        return lhs == rhs;
        };
    }; 
    struct vec4equal
    {
        bool operator()(const vec4 &lhs, const vec4 &rhs) const restrict(cpu,amp)
        {
        return lhs == rhs;
        };
    }; 

    struct vec8equal
    {
        bool operator()(const vec8 &lhs, const vec8 &rhs) const restrict(cpu,amp)
        {
        return lhs == rhs;
        };
    }; 

#endif
#else
    struct vec2equal
    {
            __host__ __device__
        bool operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        return lhs == rhs;
        };
        }; 
    struct vec4equal
    {
            __host__ __device__
        bool operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        return lhs == rhs;
        };
    }; 
    struct vec8equal
    {
            __host__ __device__
        bool operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        return lhs == rhs;
        };
    }; 
#endif
#if (BOLT_BENCHMARK == 0)
/******************************************************************************
 *  User Defined Binary Predicates DATA_TYPE less than for thrust usage
 *****************************************************************************/
    struct vec1less
    {
            __host__ __device__
        bool operator()(const DATA_TYPE &lhs, const DATA_TYPE &rhs) const
        {
        return (lhs < rhs);
        };
    }; 
#endif
/******************************************************************************
 *  User Defined Binary Predicates- vec2,4,8 less than  
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == CL_BENCH
    BOLT_FUNCTOR(vec2less,
    struct vec2less
    {
        bool operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        bool l_value;
        l_value =  (lhs.a < rhs.a)?true:false;
        l_value =  (lhs.b < rhs.b)?true:false;
        return l_value;
        };
    };
    );
    BOLT_FUNCTOR(vec4less,
    struct vec4less
    {
        bool operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        return false;
        };
    };
    );
    BOLT_FUNCTOR(vec8less,
    struct vec8less
    {
        bool operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        if (lhs.e < rhs.e) return true;
        if (lhs.f < rhs.f) return true;
        if (lhs.g < rhs.g) return true;
        if (lhs.h < rhs.h) return true;
        return false;
        };
    }; 
    );

#elif BENCHMARK_CL_AMP == AMP_BENCH
  struct vec2less
    {
        bool operator()(const vec2 &lhs, const vec2 &rhs) const restrict(cpu,amp)
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        return false;
        };
    };

    struct vec4less
    {
        bool operator()(const vec4 &lhs, const vec4 &rhs) const restrict(cpu,amp)
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        return false;
        };
    };

    struct vec8less
    {
        bool operator()(const vec8 &lhs, const vec8 &rhs) const restrict(cpu,amp)
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        if (lhs.e < rhs.e) return true;
        if (lhs.f < rhs.f) return true;
        if (lhs.g < rhs.g) return true;
        if (lhs.h < rhs.h) return true;
        return false;
        };
    }; 
#endif
#else
    struct vec2less
    {
            __host__ __device__
        bool operator()(const vec2 &lhs, const vec2 &rhs) const
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        return false;
        };
    }; 
    struct vec4less
    {
            __host__ __device__
        bool operator()(const vec4 &lhs, const vec4 &rhs) const
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        return false;
        };
    }; 
    struct vec8less
    {
            __host__ __device__
        bool operator()(const vec8 &lhs, const vec8 &rhs) const
        {
        if (lhs.a < rhs.a) return true;
        if (lhs.b < rhs.b) return true;
        if (lhs.c < rhs.c) return true;
        if (lhs.d < rhs.d) return true;
        if (lhs.e < rhs.e) return true;
        if (lhs.f < rhs.f) return true;
        if (lhs.g < rhs.g) return true;
        if (lhs.h < rhs.h) return true;
        return false;
        };
    }; 
#endif
/******************************************************************************
 *  User Defined generator-  DATATYPE and vec2,4,8
 *****************************************************************************/
#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == CL_BENCH
	
	BOLT_FUNCTOR(intgen,
	struct intgen
	{ 
        DATA_TYPE operator()() const
        {
        DATA_TYPE v =  4;
        return v;
        };
	};
	);

    BOLT_FUNCTOR(vec2gen,
    struct vec2gen
    {
        vec2 operator()() const
        {
        vec2 v = { 4, 5 };
        return v;
        };
    }; 
    );
    BOLT_FUNCTOR(vec4gen,
    struct vec4gen
    {
        vec4 operator()() const
        {
        vec4 v = { 4, 5, 6, 7 };
        return v;
        };
    }; 
    );
    BOLT_FUNCTOR(vec8gen,
    struct vec8gen
    {
        vec8 operator()() const
        {
        vec8 v = { 8, 9, 10, 11, 12, 13, 14, 15 };
        return v;
        };
    }; 
    );

#elif BENCHMARK_CL_AMP == AMP_BENCH

    struct intgen
    {
        DATA_TYPE operator()() const restrict(cpu,amp)
        {
        DATA_TYPE v = 4;
        return v;
        };
    };
    struct vec2gen
    {
        vec2 operator()() const restrict(cpu,amp)
        {
        vec2 v = { 4, 5 };
        return v;
        };
    }; 
    struct vec4gen
    {
        vec4 operator()() const restrict(cpu,amp)
        {
        vec4 v = { 4, 5, 6, 7 };
        return v;
        };
    }; 

    struct vec8gen
    {
        vec8 operator()() const restrict(cpu,amp)
        {
        vec8 v = { 8, 9, 10, 11, 12, 13, 14, 15 };
        return v;
        };
    }; 
#endif
#else
    
    struct intgen
    {
            __host__ __device__
        DATA_TYPE operator()() const
        {
        DATA_TYPE v = 1;
        return v;
        };
    }; 
    struct vec2gen
    {
            __host__ __device__
        vec2 operator()() const
        {
        vec2 v = { 2, 3 };
        return v;
        };
    }; 
    struct vec4gen
    {
            __host__ __device__
        vec4 operator()() const
        {
        vec4 v = { 4, 5, 6, 7 };
        return v;
        };
    }; 
    struct vec8gen
    {
            __host__ __device__
        vec8 operator()() const
        {
        vec8 v = { 8, 9, 10, 11, 12, 13, 14, 15 };
        return v;
        };
    }; 
#endif

    int ValidateBenchmarkKey(const char *par_instr,std::string keys [],int len)
    {
    int loc_mid,
        loc_high,
        loc_low;

    loc_low = 0;
    loc_high = len-1;
    /*
     *Binary search.
     */
    while(loc_low <= loc_high)
    {
        loc_mid =((loc_low + loc_high) / 2);

        if(strcmp((const char  *)keys[loc_mid].c_str(),(const char  *)par_instr) < 0 )
        {
            loc_low = (loc_mid + 1);
        }
        else if (strcmp((const char  *)keys[loc_mid].c_str(),(const char  *)par_instr) > 0 )
        {
            loc_high = (loc_mid - 1);
        }
        else
        { 
            return loc_mid;									

        }
    }

       /* Not a key word */
    return -1;
}
    template< typename keytype>
    void keysGeneration(keytype &keys, int len)
    {	
        int segmentLength = 0;
        int segmentIndex = 0;
        std::vector<DATA_TYPE> key(1);
        key[0] = v1iden;

        for (int i = 0; i < len; i++)
        {
            // start over, i.e., begin assigning new key
            if (segmentIndex == segmentLength)
            {
                segmentLength++;
                segmentIndex = 0;
                key[0] = key[0]+1 ; // key[0]++  is not working in the device_vector
            }
            keys[i] = key[0];
            segmentIndex++;
        }
    }

#if (BOLT_BENCHMARK == 1)
#if BENCHMARK_CL_AMP == AMP_BENCH
    void Amp_GPU_wait(bolt::amp::control& ctrl)
    {
        bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );
        if( runMode == bolt::amp::control::Gpu)
            ctrl.getAccelerator().default_view.wait();
    }
#endif
#endif