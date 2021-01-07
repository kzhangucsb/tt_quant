#include<iostream>
#include <cnl/all.h>
#include <vector>
using namespace std;
using cnl::power;
using cnl::scaled_integer;
typedef scaled_integer<std::int8_t, power<-7>> TYPE_WEIGHT ;

int main() {
    TYPE_WEIGHT a = 0.2;
    cout << a << endl;
    TYPE_WEIGHT b = (a + TYPE_WEIGHT(0x08 / pow(2.0, 7))) & TYPE_WEIGHT(0xf0 / pow(2.0, 7));
    cout << b << endl;
    std::vector<int> out_shape = {4, 8};
    cout << out_shape[0] << out_shape[1] << endl;
}