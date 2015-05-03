#include <coordinate>
int main()
{
    bool ret = true;
    {
        std::bounds<4> a({5, 5, 6, 6});
        auto it = a.begin();
        ret &= *it == std::offset<4>{};
    }
    {
        std::bounds<4> a({0, 5, 6, 6});
        auto it = a.begin();
        ret &= it == a.end();
    }
    return !ret;
}
