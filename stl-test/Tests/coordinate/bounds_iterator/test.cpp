#include <coordinate>
#include <stdio.h>
int main()
{
    bool ret = true;
    {
        std::bounds<4> a({5, 5, 6, 6});
        auto it1 = a.begin();
        std::bounds<4> b({5, 5, 6, 6});
        auto it2 = b.begin();
        ret &= it1 == it2;
        ret &= it1 <= it2;
        ret &= it1 >= it2;
        it2++;
        ret &= *it2 == std::index<4>({0, 0, 0, 1});
        ret &= it1 < it2;
        ret &= it2 > it1;
    }
    return !ret;
}
