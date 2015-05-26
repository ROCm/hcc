#include <coordinate>
int main()
{
    bool ret = true;
    {
        std::bounds<4> a({5, 5, 6, 6});
        std::bounds<4> b({5, 5, 6, 6});
        ret &= a == b;
    }
    {
        std::bounds<4> a({5, 5, 6, 6});
        std::bounds<4> b({5, 5, 6, 7});
        ret &= a != b;
    }
    return !ret;
}
