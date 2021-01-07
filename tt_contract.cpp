#include "tt_nn.h"
#ifndef SYNTHESIS
#include <assert.h>
#include <iostream>
#include <math.h>
using namespace std;
#endif




void tensor_cont_mid(
    TYPE_DATA* data_in,
    TYPE_DATA* data_out,
    TYPE_WEIGHT* weight,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2,
    float shift,
    TYPE_INTER* max
){
    /* tensor contraction on the second dimension
    ABCxDBE->ADEC
    * array_in: size (array_in_size_0*array_in_size_1*array_in_size_2),
    * array_weight: size (array_weight_size_0*array_in_size_1*array_weight_size_2),
    * array_out: size(array_in_size_0*array_weight_size_0*array_weight_size_2*array_in_size_2)
    * All arrays are in C order
    */
    #ifndef SYNTHESIS
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    #endif 
    TYPE_INTER res[PARALLEL_DEGREE];
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
        for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
            for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2+=PARALLEL_DEGREE) {
				for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
                        res[i_in_o] = 0;
					}
					for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
						int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2, array_in_size_1, array_in_size_2);
						int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
						for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
							res[i_in_o] += float(data_in[ind_in + i_in_o]) * float(weight[ind_w]);
						}
					}
					int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
						array_weight_size_0, array_weight_size_2, array_in_size_2);

                    TYPE_RINT rn = pseudo_random();
					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
                        res[i_in_o] /= pow(2, shift);
                        data_out[ind_out+i_in_o] = TYPE_GRAD(res[i_in_o] + randadj(rn, i_in_o));
                        if (*max < abs(res[i_in_o])) {
                            *max = abs(res[i_in_o]);
                        }
                    }
                }
            }
        }
    }
}

void tensor_cont_last(
    TYPE_DATA* data_in,
    TYPE_DATA* data_out,
    TYPE_WEIGHT* weight,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
    float shift,
    TYPE_INTER* max
){
    /* tensor contraction on the first and last dimension
    ABCxBDC->AD
    */
    #ifndef SYNTHESIS
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    #endif 
    //TYPE_INTER res;
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
        for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
            TYPE_INTER res = 0;
            for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
                for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE) {
                    int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2,  array_in_size_1, array_in_size_2);
                    int ind_w = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
                    for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
                        res += TYPE_INTER(data_in[ind_in+i_in_o]) * TYPE_INTER(weight[ind_w+i_in_o]);
                    }
                }
            }
            int ind_out = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
            res /= pow(2, shift);
            TYPE_RINT rn = pseudo_random();
            data_out[ind_out] = TYPE_GRAD(res + randadj(rn, 0));
            if (*max < abs(res)) {
                *max = abs(res);
            }
        }
    }
}

void getstride1(
    int* shape,
    int dim,
    int* res
){
    int st = 1;
    for (int i = dim - 1; i >= 0; i--) {
        res[i] = st;
        st *= shape[i];
    }
}
void getstride2(
    int* shape1, // input
    int* shape2, // output
    int dim,
    int* res
){
    int st = 1;
    for (int i = dim - 1; i >= 0; i--) {
        res[i*2+1] = st;
        st *= shape1[i];
        res[i*2] = st;
        st *= shape2[i];
    }
}

int ind2offset(
    int* index,
    int* stride,
    int dim
){
    int ret = 0;
    for (int i = 0; i < dim; i++) {
        ret += index[i] * stride[i];
    }
    return ret;
}

int ind22offset(
    int* index1,
    int* index2,
    int* stride,
    int dim
){
    int ret = 0;
    for (int i = 0; i < dim; i++) {
        ret += index1[i] * stride[i * 2 + 1];
        ret += index2[i] * stride[i * 2];
    }
    return ret;
}


void tensor_cont_outer_prod(
    TYPE_DATA *data_in, // input
    TYPE_DATA *grad_out, // output_grad
    TYPE_GRAD *grad_wt,
    int* input_shape,
    int* output_shape,
    int dim,
    float shift,
    TYPE_INTER* max
){
    int stride1[5];
    int stride2[5];
    int strideo[10];
    int index1[5] = {0};
    int index2[5] = {0};
    assert(dim <= 5); // dim too high
    int shape1exp[5] = {1, 1, 1, 1, 1};
    int shape2exp[5] = {1, 1, 1, 1, 1};
    for (int i = 0; i < dim; i++) {
        shape1exp[i + 5 - dim] = input_shape[i];
        shape2exp[i + 5 - dim] = output_shape[i];
    }
    getstride1(shape1exp, 5, stride1);
    getstride1(shape2exp, 5, stride2);
    getstride2(shape1exp, shape2exp, 5, strideo);
    int ind_i = 0, ind_o = 0, ind_r = 0;
    for (index2[0] = 0; index2[0] < shape2exp[0]; index2[0]++) {
    for (index2[1] = 0; index2[1] < shape2exp[1]; index2[1]++) {
    for (index2[2] = 0; index2[2] < shape2exp[2]; index2[2]++) {
    for (index2[3] = 0; index2[3] < shape2exp[3]; index2[3]++) {
    for (index2[4] = 0; index2[4] < shape2exp[4]; index2[4]++) {
    for (index1[0] = 0; index1[0] < shape1exp[0]; index1[0]++) {
    for (index1[1] = 0; index1[1] < shape1exp[1]; index1[1]++) {
    for (index1[2] = 0; index1[2] < shape1exp[2]; index1[2]++) {
    for (index1[3] = 0; index1[3] < shape1exp[3]; index1[3]++) {    
    for (index1[4] = 0; index1[4] < shape1exp[4]; index1[4]++) {
        // assert(ind_i == ind2offset(index1, stride1, 5));
        // assert(ind_o == ind2offset(index2, stride2, 5));
        // assert(ind_r == ind22offset(index1, index2, strideo, 5));
        
        TYPE_INTER res = float(data_in[ind_i]) * float(grad_out[ind_o]);
        res /= pow(2.0, shift);
        if (*max < abs(res)) {
            *max = abs(res);
        }
        TYPE_RINT rn = pseudo_random();
        res += randadj(rn, 0);
        grad_wt[ind_r] += TYPE_GRAD(res);
        ind_i += stride1[4];
        ind_r += strideo[9];
    }
        ind_i -= stride1[4] * shape1exp[4];
        ind_r -= strideo[9] * shape1exp[4];
        ind_i += stride1[3];
        ind_r += strideo[7];
    }   
        ind_i -= stride1[3] * shape1exp[3];
        ind_r -= strideo[7] * shape1exp[3];
        ind_i += stride1[2];
        ind_r += strideo[5];
    }  
        ind_i -= stride1[2] * shape1exp[2];
        ind_r -= strideo[5] * shape1exp[2];
        ind_i += stride1[1];
        ind_r += strideo[3];
    }   
        ind_i -= stride1[1] * shape1exp[1];
        ind_r -= strideo[3] * shape1exp[1];
        ind_i += stride1[0];
        ind_r += strideo[1];
    }
        ind_i = 0;
        ind_r -= strideo[1];
        ind_o += stride2[4];
        ind_r += strideo[8];
    }
        ind_o -= stride2[4] * shape2exp[4];
        ind_r -= strideo[8] * shape2exp[4];
        ind_o += stride2[3];
        ind_r += strideo[6];
    }
        ind_o -= stride2[3] * shape2exp[3];
        ind_r -= strideo[6] * shape2exp[3];
        ind_o += stride2[2];
        ind_r += strideo[4];
    }
        ind_o -= stride2[2] * shape2exp[2];
        ind_r -= strideo[4] * shape2exp[2];   
        ind_o += stride2[1];
        ind_r += strideo[2];
    }
        ind_o -= stride2[1] * shape2exp[1];
        ind_r -= strideo[2] * shape2exp[1];
        ind_o += stride2[0];
        ind_r += strideo[0];
    }
}


// ************* below should not be used *******************************
// void tensor_cont_end_backward(
//     TYPE_DATA *data_in_1,
//     TYPE_DATA *data_in_2,
//     TYPE_GRAD *grad_out,
//     int array_in_size_0,
//     int array_in_size_1,
//     int array_in_size_2,
//     int array_in_size_3,
//     int array_weight_size_1,
//     int shift,
//     TYPE_INTER* max
// ){
//     /* tensor contraction on the first and last dimension
//     ABCDxAED->BEC
//     */
//     #ifndef SYNTHESIS
//     assert (array_in_size_3 % PARALLEL_DEGREE == 0);
//     #endif 
//     TYPE_INTER res;
//     for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
//         for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2++) {
//             for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
//                 res = 0;
//                 for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//                     for (int i_in_3 = 0; i_in_3 < array_in_size_3; i_in_3 += PARALLEL_DEGREE) {
//                         int ind_in = sub2ind4(i_in_0, i_in_1, i_in_2, i_in_3, 
//                             array_in_size_1, array_in_size_2, array_in_size_3);
//                         int ind_w = sub2ind3(i_in_0, i_w_1, i_in_3, array_weight_size_1, array_in_size_3);
//                         for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
//                             res += data_in_1[ind_in+i_in_o] * data_in_2[ind_w+i_in_o];
//                         }
//                     }
//                 }
//                 int ind_out = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
//                 res /= pow(2, shift);
//                 grad_out[ind_out] += res;
//                 if (*max < abs(res)) {
//                     *max = abs(res);
//                 }
//             }
//         }
//     }
   
// }

// void tensor_cont_head_backward(
//     TYPE_DATA *data_in_1,
//     TYPE_DATA *data_in_2,
//     TYPE_GRAD *grad_out,
//     int array_in_size_0,
//     int array_in_size_1,
//     int array_weight_size_1,
//     int shift,
//     TYPE_INTER* max
// ){
//     /* tensor contraction on the first and last dimension
//     ABxAE->BE
//     */
//     #ifndef SYNTHESIS
//     assert (array_in_size_1 % PARALLEL_DEGREE == 0);
//     #endif 
//     TYPE_INTER res[PARALLEL_DEGREE];
//     for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
//         for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1+= PARALLEL_DEGREE) {
//             for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//                 res[i_in_o] = 0;
//             }
//             for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//                 int ind_in = sub2ind3(0, i_in_0, i_in_1, array_in_size_0, array_in_size_1);
//                 int ind_w = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
//                 for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//                     res[i_in_o] += data_in_1[ind_in] * data_in_2[ind_w + i_in_o];
//                 }
//             }
//             int ind_out = sub2ind3(0, i_in_1, i_w_1, array_in_size_1, array_weight_size_1);
//             for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//                 res[i_in_o] /= pow(2, shift);
//                 grad_out[ind_out + i_in_o] += res[i_in_o];
//                 if (*max < abs(res[i_in_o])) {
//                     *max = abs(res[i_in_o]);
//                 }
//             }
//         }
//     }
// }
