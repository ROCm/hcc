#include <array_view>
#include <vector>
#include <string>
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
        for (int i = 0; i < 120; i++)
            ret &= av.data()[i] == i;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 1> av(vec); 
        std::array_view<int, 1> bv(av); 
        ret &= av.size() == bv.size();
        ret &= av.data() == bv.data();
        for (int i = 0; i < 120; i++)
            ret &= bv.data()[i] == i;
    }
    {
        char a[12] {'H', 'i'};
        auto av = std::array_view<char, 1>{a};
        ret &= av.bounds() == std::bounds<1>{12};
        ret &= av[{0}] == 'H';
        ret &= av[{1}] == 'i';
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec, std::bounds<3>{3, 4, 10}); 
        std::array_view<const int, 3> bv(av); 
        ret &= av.size() == bv.size();
        ret &= av.data() == bv.data();
        for (int i = 0; i < 120; i++)
            ret &= bv.data()[i] == i;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec, std::bounds<3>{3, 4, 10}); 
        std::array_view<int, 3> bv(av, std::bounds<3>{6, 10, 2}); 
        ret &= av.size() == bv.size();
        ret &= av.data() == bv.data();
        for (int i = 0; i < 120; i++)
            ret &= bv.data()[i] == i;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 10; j++)
                for (int k = 0; k < 2; k++)
                    ret &= bv[{i, j, k}] == i * 20 + j * 2 + k;
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec.data(), std::bounds<3>{3, 4, 10});
        ret &= av.size() == vec.size();
        ret &= av.data() == vec.data();
        ret &= av.bounds() == std::bounds<3>{3, 4, 10};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 10; k++)
                    ret &= av[{i, j, k}] == i * 40 + j * 10 + k;
 
    }
    {
        std::vector<int> vec(120);
        for (int i = 0; i < 120; i++)
            vec[i] = i;
        std::array_view<int, 3> av(vec.data(), std::bounds<3>{3, 4, 10});
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 10; k++)
                    ret &= av[i][j][k] == i * 40 + j * 10 + k;
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
    }
    return !ret;
}
