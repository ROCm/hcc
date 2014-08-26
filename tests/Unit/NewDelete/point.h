#ifndef _POINT_H
#define _POINT_H

class Point
{
  int _x;
  int _y;
public:
  Point() restrict(amp, cpu) : _x(0), _y(0) {}
  Point(int x, int y) restrict(amp, cpu) : _x(x), _y(y) {}

  int get_x() { return _x; }
  int get_y() { return _y; }
};

#endif
