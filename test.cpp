#define QUANTIZE
#include "tt_nn.h"
#include <iostream>
#include <math.h>
using namespace std;

bool iter_index(
    int* index,
    int* shape,
    int dim
);

//cnl::from_rep<TYPE_WEIGHT, int> int2wt;
#define WEIGHT_MASK TYPE_WEIGHT(0xf0 / pow(2, 7))
#define WEIGHT_ADJ  TYPE_WEIGHT(0x08 / pow(2, 7))


int main(){
    

    // int a[10] = {1};
    // cout << a[1] << endl;
    //cout << cnl::from_rep<0xf0, 0xf0> (0xf0) << endl;
    TYPE_WEIGHT a = 0.8;
    cnl::from_rep<TYPE_INTER, TYPE_RINT> cvt;
    TYPE_INTER t = cvt(0xfffffff8);
    cout << t << endl;
    cout << (t >> 1) << endl;
    cout << (t >> 2) << endl;

    TYPE_INTER c;
    c = 100000000.0;
    cout << c << endl;
    cout << c / pow(2, 5) << endl;
    TYPE_DATA b = 0.8;
    c = float(a) * float(b);
    c += float(a) * float(b);
    cout << c << endl;
    //cout << (TYPE_WEIGHT(0.8) + WEIGHT_ADJ & WEIGHT_MASK) << endl;
    cout << WEIGHT_ADJ << ' ' << WEIGHT_MASK << endl;
    for (TYPE_WEIGHT t = -0.98; t < 0.98; t+=0.02) {
        TYPE_WEIGHT t1 = t;
        t1 += WEIGHT_ADJ;
		t1 &= WEIGHT_MASK;
        cout << t << ' ' << t1 << endl;
    }

    cnl::from_rep<TYPE_INTER, TYPE_RINT> int2inter;
    cout << int2inter(13) << TYPE_DATA(int2inter(13)) << endl;
}

// TYPE_DATA* data_in = new TYPE_DATA[28*32];
    // TYPE_DATA* grad_in = new TYPE_DATA[28*32];
    // TYPE_DATA* grad_out = new TYPE_DATA[512];
    // TYPE_WEIGHT* weight[8];
    // weight[0] = new TYPE_WEIGHT[1*4*7*16];
    // weight[1] = new TYPE_WEIGHT[16*4*4*16];
    // weight[2] = new TYPE_WEIGHT[16*2*2*16];
    // weight[3] = new TYPE_WEIGHT[16*16*16*1];
    // weight[4] = new TYPE_WEIGHT[16*16*16*1];
    // weight[5] = new TYPE_WEIGHT[32*16];
    // weight[6] = new TYPE_WEIGHT[16*16*16*1];
    // weight[7] = new TYPE_WEIGHT[16*16*16*1];
    // TYPE_GRAD* weight_grad[8];
    // weight_grad[0] = new TYPE_GRAD[1*4*7*16];
    // weight_grad[1] = new TYPE_GRAD[16*4*4*16];
    // weight_grad[2] = new TYPE_GRAD[16*2*2*16];
    // weight_grad[3] = new TYPE_GRAD[16*16*16*1];
    // weight_grad[4] = new TYPE_GRAD[32*16];
    // weight_grad[5] = new TYPE_GRAD[16*16*16*1];
    // TYPE_GRAD* wg = new TYPE_GRAD[28*32*512];
    // TYPE_DATA* bias_grad = new TYPE_DATA[512];
    // TYPE_DATA* out_grad = new TYPE_DATA[16];
    // int input_shape[] = {7,4,2,16};
    // int output_shape[] = {4,4,2,16};
    // int rank[] = {16,16,16};
    // int input_shape2[] = {32,16};
    // int output_shape2[] = {1,16};
    // int rank2[] = {16};
    // int shift[18] = {0};
    // TYPE_INTER max[18];
    // TYPE_GRAD* tmp[4] = {0};
    // tensor_train_forward(
    //     data_in,
    //     grad_out,
    //     weight,
    //     bias_grad,
    //     input_shape,
    //     output_shape,
    //     rank,
    //     4,
    //     shift,
    //     max
    // );
    // tensor_train_backward(
    //     data_in,
    //     grad_out,
    //     grad_in,
    //     weight,
    //     weight_grad,
    //     bias_grad,
    //     input_shape,
    //     output_shape,
    //     rank,
    //     4,
    //     shift,
    //     max
    // );
    // tensor_train_backward(
    //     grad_out,
    //     out_grad,
    //     grad_out,
    //     weight + 5,
    //     weight_grad + 4,
    //     out_grad,
    //     input_shape2,
    //     output_shape2,
    //     rank2,
    //     2,
    //     shift,
    //     max
    // );
    // tensor_cont_outer_prod(
    //     data_in,
    //     grad_out,
    //     wg,
    //     input_shape,
    //     output_shape,
    //     4,
    //     0,
    //     max
    // );
    // for (int i = 0; i < 4; i++) {
    //     tensor_train_factors_grad(
    //         wg,
    //         weight,
    //         weight_grad,
    //         tmp,
    //         input_shape,
    //         output_shape,
    //         rank,
    //         4,
    //         i,
    //         shift,
    //         max
    //     );
    // }
    // int shape[] = {5, 4, 3};
    // int index[] = {0, 0, 0};
    // do{
    //     cout << index[0] << ' ' << index[1] << ' ' << index[2] << endl;
    // }
    // while(iter_index(index, shape, 3));