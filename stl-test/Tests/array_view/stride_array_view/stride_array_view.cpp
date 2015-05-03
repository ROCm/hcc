#include <array_view>
#include <vector>
#include <stdio.h>
int main()
{
    bool ret = true;
    {
        std::strided_array_view<int, 3> av;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec, std::bounds<3>{3, 4, 10}); 
        std::strided_array_view<int, 3> bv(av); 
        ret &= av.size() == bv.size();
        ret &= bv[{1, 2, 3}] == 63;
        ret &= bv[{2, 2, 3}] == 103;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec, std::bounds<3>{3, 4, 10}); 
        std::strided_array_view<int, 3> cv(av);
        std::strided_array_view<const int, 3> bv(cv);
        ret &= av.size() == bv.size();
        ret &= bv[{1, 2, 3}] == 63;
        ret &= bv[{2, 2, 3}] == 103;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec.data(), std::bounds<3>{3, 4, 10});
        ret &= av.size() == vec.size();
        ret &= av.bounds() == std::bounds<3>{3, 4, 10};
        ret &= av[{1, 2, 3}] == 63;
        ret &= av[{2, 2, 3}] == 103;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec.data(), std::bounds<3>{3, 4, 10});
        ret &= av[1][2][3] == 63;
        ret &= av[2][2][3] == 103;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec.data(), std::bounds<3>{3, 4, 10});
        auto p = av.section(std::offset<3>{0, 1, 2});
        ret &= p[1][1][1] == 63;
        ret &= p[2][1][1] == 103;
        auto k = p.section(std::offset<3>{1, 1, 1});
        ret &= k[0][0][0] == 63;
        ret &= k[1][0][0] == 103;
        auto m = k.section(std::offset<3>{1, 1, 2});
        ret &= m[0][0][0] == 115;
        ret &= av.section(std::offset<3>{1, 1, 1}).
                  section(std::offset<3>{0, 1, 1}).
                  section(std::offset<3>{0, 0, 1})[0][0][0] == 63;
    }
    return !ret;
}
