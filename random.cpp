#include "tt_nn.h"
#ifdef SYNTHESIS
#include <ap_fixed.h>


static TYPE_RINT lfsr = 1;

void seed(TYPE_RINT init){
    lfsr = init;
}

TYPE_RINT pseudo_random() {
    bool b_32 = lfsr.get_bit(32-32);
    bool b_22 = lfsr.get_bit(32-22);
    bool b_2 = lfsr.get_bit(32-2);
    bool b_1 = lfsr.get_bit(32-1);
    bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
    lfsr = lfsr >> 1;
    lfsr.set_bit(31, new_bit);
    return lfsr;

}

#elif defined QUANTIZE
static TYPE_RINT lfsr = 1;
inline bool get_bit(TYPE_RINT num, int pos) {
    return (num >> pos) & 1;
}

TYPE_RINT pseudo_random() {
    bool b_32 = get_bit(lfsr, 32-32);
    bool b_22 = get_bit(lfsr, 32-22);
    bool b_2 = get_bit(lfsr, 32-2);
    bool b_1 = get_bit(lfsr, 32-1);
    bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
    lfsr = lfsr >> 1;
    lfsr |= int(new_bit) << 31;
    return lfsr;
}

cnl::from_rep<TYPE_INTER, TYPE_RINT> int2inter;

TYPE_INTER randadj(TYPE_RINT rn, int pos) {
    pos = pos % 8;
    TYPE_RINT rn_act = (rn >> (pos * 4)) & 0x0f;
    TYPE_INTER ret = int2inter(rn_act);
    return 0;
}

#else
TYPE_RINT pseudo_random() {
    return 0;
}
TYPE_INTER randadj(TYPE_RINT rn, int pos) {
    return 0;
}

#endif

