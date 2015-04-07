#include <coordinate>
int main()
{
    bool ret = true;
    {
        std::index<4> a({5, 5, 6, 6});
        ret &= a[0] == 5;
        ret &= a[3] == 6;
    }
    return !ret;
}
