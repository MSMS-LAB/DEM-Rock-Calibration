#ifndef MATH_CONSTANTS_H_INCLUDED
#define MATH_CONSTANTS_H_INCLUDED
#include <math.h>
#include <stdlib.h>
/*#include <fstream>
#include <istream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>/**/


const double MC_1d3 = 1.0/3.0;
const double MC_1d6 = 1.0/6.0;
const double MC_1d9 = 1.0/9.0;
const double MC_1d18 = 1.0/18.0;
const double MC_1d27 = 1.0/27.0;
const double MC_2d3 = 2.0/3.0;
const double MC_2d6 = 2.0/6.0;
const double MC_1ds3 = 1.0/sqrt(3.0);
const double MC_2ds3 = 2.0/sqrt(3.0);
const double MC_s3 = sqrt(3.0);
const double MC_s3d2 = sqrt(3.0)/2.0;
const double MC_s3d3 = sqrt(3.0)/3.0;
const double MC_s3d4 = sqrt(3.0)/4.0;
const double MC_s2 = sqrt(2.0);
const double MC_s2d2 = sqrt(2.0)/2.0;
const double MC_1ds2 = MC_s2d2;
const double MC_2s2 = 2.0*sqrt(2.0);
const double MC_2s3 = 2.0*sqrt(3.0);
const double MC_2s2d3 = 2.0*sqrt(2.0)/3.0;
const double MC_4s2d3 = 4.0*sqrt(2.0)/3.0;
const double MC_cos30 = MC_s3d2;
const double MC_cos45 = MC_s2d2;
const double MC_cos60 = 0.5;
const double MC_sin30 = 0.5;
const double MC_sin45 = MC_s2d2;
const double MC_sin60 = MC_s3d2;
const double MC_tan30 = MC_sin30/MC_cos30;
const double MC_tan45 = MC_sin45/MC_cos45;
const double MC_tan60 = MC_sin60/MC_cos60;
const double MC_pi = 3.1415926535897932384626433832795;
const double MC_e = 2.7182818284590452353602874713527;
const double MC_min_double = -1.7e-308;
const double MC_max_double = 1.7e+308;
const double MC_zero_double = 1e-15;
const double MC_1d72 = 1.0/72.0;
const double MC_1d_60 = 1.0 / 60.0;
const double MC_1d_3600 = 1.0 / 3600.0;
const double RAND_MAX_double = (double)RAND_MAX;
const double _1d_RAND_MAX_double = 1.0/RAND_MAX_double;
//const uint UINT64_MAX = 0xffffffffffffffff;
const double near_zero = 1e-14;

#define MCf_1d3 0.3333333333f
#define MCf_1d6 0.1666666667f
#define MCf_1d9 0.1111111111f
#define MCf_1d18 0.0555555555f
#define MCf_1d27 0.03703703704f
#define MCf_1d64 0.015625f
#define MCf_2d3 0.6666666667f
#define MCf_2d6 0.33333333333f
#define MCf_1ds3 0.57735026919f
#define MCf_2ds3 1.15470053838f
#define MCf_s3 1.732050807f
#define MCf_s3d2 0.8660254037844f
#define MCf_s3d3 0.5773502691896f
#define MCf_s3d4 0.433012702f
#define MCf_s2 1.41421356237309f
#define MCf_s2d2 0.7071067811865f
#define MCf_1ds2 0.7071067811865f
#define MCf_2s2 2.82842712474619f
#define MCf_2s3 3.464101615f
#define MCf_2s2d3 0.94280904158206f
#define MCf_4s2d3 1.88561808316412f
#define MCf_cos30 0.8660254037844f
#define MCf_cos45 0.7071067811865f
#define MCf_cos60 0.5f
#define MCf_sin30 0.5f
#define MCf_sin45 0.7071067811865f
#define MCf_sin60 0.8660254037844f
#define MCf_tan30 0.57735026918962f
#define MCf_tan45 1.0f
#define MCf_tan60 1.73205080756888f
#define MCf_pi 3.14159265358979f
#define MCf_e 2.71828182845904f

#endif // MATH_CONSTANTS_H_INCLUDED
