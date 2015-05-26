#include <coordinate>
int main()
{
    bool ret = true;
    {
        std::bounds<4> a({5, 5, 6, 6});
        ret &= a.size() == 900;
    }
    {
        std::bounds<4> bnd({5, 5, 6, 6});
        std::offset<4> ia({5, 5, 6, 5});
        std::offset<4> ib({5, 5, 6, 7});
        ret &= bnd.contains(ia);
        ret &= !bnd.contains(ib);
    }
    return !ret;
}
