#include <array_view>
#include <vector>
#include <stdio.h>
int main()
{
    bool ret = true;
    {
        std::array_view<int, 3> av;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 1> av(vec); 
        ret &= av.size() == vec.size();
        ret &= av.data() == vec.data();
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 1> av(vec); 
        std::array_view<int, 1> bv(av); 
        ret &= av.size() == bv.size();
        ret &= av.data() == bv.data();
    }
    {
        char a[3][1][4] {{{'H', 'i'}}};
        auto av = std::array_view<char, 3>{a};
        ret &= av.bounds() == std::bounds<3>{3, 1, 4};
        ret &= av[{0, 0, 0}] == 'H';
        ret &= av[{0, 0, 1}] == 'i';
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec, std::bounds<3>{3, 4, 10}); 
        std::array_view<const int, 3> bv(av); 
        ret &= av.size() == bv.size();
        ret &= av.data() == bv.data();
        ret &= bv[{1, 2, 3}] == 63;
        ret &= bv[{2, 2, 3}] == 103;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec, std::bounds<3>{3, 4, 10}); 
        std::array_view<int, 3> bv(av, std::bounds<3>{6, 4, 5}); 
        ret &= av.size() == bv.size();
        ret &= av.data() == bv.data();
        ret &= av[{1, 2, 3}] == 63;
        ret &= av[{2, 2, 3}] == 103;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec.data(), std::bounds<3>{3, 4, 10});
        ret &= av.size() == vec.size();
        ret &= av.data() == vec.data();
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
        auto p = av.section(std::index<3>{0, 1, 2});
        ret &= p[1][1][1] == 63;
        ret &= p[2][1][1] == 103;
        auto k = p.section(std::index<3>{1, 1, 1});
        ret &= k[0][0][0] == 63;
        ret &= k[1][0][0] == 103;
        auto m = k.section(std::index<3>{1, 1, 2});
        ret &= m[0][0][0] == 115;
    }
    return !ret;
}
