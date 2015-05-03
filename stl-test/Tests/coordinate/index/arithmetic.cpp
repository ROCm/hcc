#include <coordinate>
#include <stdio.h>
int main()
{
    bool ret = true;
    {
        std::offset<4> a({2, 2, 3, 3});
        std::offset<4> b({3, 3, 3, 3});
        std::offset<4> c = a + b;
        ret &= c[0] == 5;
        ret &= c[3] == 6;
    }
    {
        std::offset<4> a({2, 2, 3, 3});
        std::offset<4> b({3, 3, 3, 3});
        std::offset<4> c = b - a;
        ret &= c[0] == 1;
        ret &= c[3] == 0;
    }
    {
        std::offset<4> a({2, 2, 3, 3});
        std::offset<4> b({3, 3, 3, 3});
        a += b;
        ret &= a[0] == 5;
        ret &= a[3] == 6;
    }
    {
        std::offset<4> a({2, 2, 3, 3});
        std::offset<4> b({3, 3, 3, 3});
        b -= a;
        ret &= b[0] == 1;
        ret &= b[3] == 0;
    }
    {
        std::offset<1> a(5566);
        ++a;
        ret &= a[0] == 5567;
    }
    {
        std::offset<1> a(5566);
        a++;
        ret &= a[0] == 5567;
    }
    {
        std::offset<1> a(5566);
        --a;
        ret &= a[0] == 5565;
    }
    {
        std::offset<1> a(5566);
        a--;
        ret &= a[0] == 5565;
    }
    {
        std::offset<4> a({5, 5, 6, 6});
        ret &= (+a)[0] == 5;
        ret &= (+a)[3] == 6;
    }
    {
        std::offset<4> a({5, 5, 6, 6});
        ret &= (-a)[0] == -5;
        ret &= (-a)[3] == -6;
    }
    {
        std::offset<4> a({5, 5, 6, 6});
        std::offset<4> b = a * 5;
        ret &= b[0] == 25;
        ret &= b[1] == 25;
        ret &= b[2] == 30;
        ret &= b[3] == 30;
    }
    {
        std::offset<4> a({5, 5, 11, 11});
        std::offset<4> b = a / 5;
        ret &= b[0] == 1;
        ret &= b[1] == 1;
        ret &= b[2] == 2;
        ret &= b[3] == 2;
    }
    {
        std::offset<4> a({5, 5, 6, 6});
        a *= 5;
        ret &= a[0] == 25;
        ret &= a[1] == 25;
        ret &= a[2] == 30;
        ret &= a[3] == 30;
    }
    {
        std::offset<4> a({5, 5, 11, 11});
        a /= 5;
        ret &= a[0] == 1;
        ret &= a[1] == 1;
        ret &= a[2] == 2;
        ret &= a[3] == 2;
    }
{
        std::offset<4> a({5, 5, 6, 6});
        std::offset<4> b = 5 * a;
        ret &= b[0] == 25;
        ret &= b[1] == 25;
        ret &= b[2] == 30;
        ret &= b[3] == 30;
    }
    return !ret;
}
