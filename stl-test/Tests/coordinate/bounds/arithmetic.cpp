#include <coordinate>
#include <stdio.h>
int main()
{
    bool ret = true;
    {
        std::bounds<4> a({2, 2, 3, 3});
        std::offset<4> b({3, 3, 3, 3});
        std::bounds<4> c = a + b;
        ret &= c[0] == 5;
        ret &= c[3] == 6;
    }
    {
        std::offset<4> a({2, 2, 3, 3});
        std::bounds<4> b({3, 3, 3, 3});
        std::bounds<4> c = b - a;
        ret &= c[0] == 1;
        ret &= c[3] == 0;
    }
    {
        std::bounds<4> a({2, 2, 3, 3});
        std::offset<4> b({3, 3, 3, 3});
        a += b;
        ret &= a[0] == 5;
        ret &= a[3] == 6;
    }
    {
        std::offset<4> a({2, 2, 3, 3});
        std::bounds<4> b({3, 3, 3, 3});
        b -= a;
        ret &= b[0] == 1;
        ret &= b[3] == 0;
    }
    {
        std::bounds<4> a({5, 5, 6, 6});
        std::bounds<4> b = a * 5;
        ret &= b[0] == 25;
        ret &= b[1] == 25;
        ret &= b[2] == 30;
        ret &= b[3] == 30;
    }
    {
        std::bounds<4> a({5, 5, 11, 11});
        std::bounds<4> b = a / 5;
        ret &= b[0] == 1;
        ret &= b[1] == 1;
        ret &= b[2] == 2;
        ret &= b[3] == 2;
    }
    {
        std::bounds<4> a({5, 5, 6, 6});
        a *= 5;
        ret &= a[0] == 25;
        ret &= a[1] == 25;
        ret &= a[2] == 30;
        ret &= a[3] == 30;
    }
    {
        std::bounds<4> a({5, 5, 11, 11});
        a /= 5;
        ret &= a[0] == 1;
        ret &= a[1] == 1;
        ret &= a[2] == 2;
        ret &= a[3] == 2;
    }
    {
        std::bounds<4> a({5, 5, 6, 6});
        std::bounds<4> b = 5 * a;
        ret &= b[0] == 25;
        ret &= b[1] == 25;
        ret &= b[2] == 30;
        ret &= b[3] == 30;
    }
    {
        std::bounds<4> a({5, 5, 6, 6});
        std::offset<4> b({3, 3, 3, 3});
        std::bounds<4> c = b + a;
        ret &= c[0] == 8;
        ret &= c[1] == 8;
        ret &= c[2] == 9;
        ret &= c[3] == 9;
    }
    return !ret;
}
