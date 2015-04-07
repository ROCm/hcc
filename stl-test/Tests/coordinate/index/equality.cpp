#include <coordinate>
int main()
{
    bool ret = true;
    {
        std::index<4> a({5, 5, 6, 6});
        std::index<4> b({5, 5, 6, 6});
        ret &= a == b;
    }
    {
        std::index<4> a({5, 5, 6, 6});
        std::index<4> b({5, 5, 6, 7});
        ret &= a != b;
    }
    return !ret;
}
