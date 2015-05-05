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
        ret &= !(it1 < it2);
        ret &= it1 >= it2;
        it2++;
        ret &= *it2 == std::offset<4>({0, 0, 0, 1});
        ret &= it1 < it2;
        ret &= it2 > it1;
        for (int i = 0; i < 410; ++i)
            it2++;
        ret &= *it2 == std::offset<4>({2, 1, 2, 3});
        for (int i = 0; i < 410; ++i)
            it2--;
        ret &= *it2 == std::offset<4>({0, 0, 0, 1});
        for (int i = 0; i < 410; ++i)
            ++it1;
        ret &= *it1 == std::offset<4>({2, 1, 2, 2});
        for (int i = 0; i < 410; ++i)
            --it1;
        ret &= *it1 == std::offset<4>({0, 0, 0, 0});
        auto it3 = it1 + 410;
        ret &= *it3 == std::offset<4>({2, 1, 2, 2});
        it3 += 410;
        ret &= *it3 == std::offset<4>({4, 2, 4, 4});
        it3 -= 410;
        ret &= *it3 == std::offset<4>({2, 1, 2, 2});
        it3 = it3 - 410;
        ret &= *it3 == std::offset<4>({0, 0, 0, 0});
        ret &= it3[410] == std::offset<4>({2, 1, 2, 2});
    }
    return !ret;
}
