#include <coordinate>
int main()
{
    bool ret = true;
    {
        std::offset<4> a({5, 5, 6, 6});
        std::offset<4> b({5, 5, 6, 6});
        ret &= a == b;
    }
    {
        std::offset<4> a({5, 5, 6, 6});
        std::offset<4> b({5, 5, 6, 7});
        ret &= a != b;
    }
    return !ret;
}
