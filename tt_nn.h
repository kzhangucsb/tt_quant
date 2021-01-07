//#define QUANTIZE
#ifdef SYNTHESIS
#pragma message "compile for synthesis"
#include <ap_fixed.h>
#define TYPE_WEIGHT ap_fixed<8, 0>
#define TYPE_DATA ap_fixed<8, 0>
#define TYPE_INTER ap_fixed<16, 0>
#define TYPE_GRAD ap_fixed<16, 0>
#define TYPE_BUFFER ap_fixed<32, 0>
#define TYPE_RINT ap_uint<32>
#elif defined QUANTIZE
#pragma message "compile for quantize"
#include <cnl/all.h>
using cnl::power;
using cnl::scaled_integer;
using cnl::overflow_integer;
// typedef overflow_integer<scaled_integer<std::int8_t, power<-7>>, cnl::saturated_overflow_tag> TYPE_WEIGHT ;
// typedef overflow_integer<scaled_integer<std::int16_t, power<-7>>, cnl::saturated_overflow_tag> TYPE_DATA;
// typedef overflow_integer<scaled_integer<std::int32_t, power<-11>>, cnl::saturated_overflow_tag> TYPE_INTER;
typedef scaled_integer<std::int8_t, power<-7>> TYPE_WEIGHT ;
typedef scaled_integer<std::int16_t, power<-7>> TYPE_DATA;
typedef scaled_integer<std::int32_t, power<-11>> TYPE_INTER;
typedef TYPE_DATA TYPE_GRAD;
typedef uint32_t TYPE_RINT;

//typedef float TYPE_BUFFER;
#else
#pragma message "compile for float"
#include <math.h>
#include <stdint.h>
typedef float TYPE_WEIGHT ;
typedef float TYPE_DATA;
typedef float TYPE_INTER;
typedef float TYPE_BUFFER;
typedef float TYPE_GRAD;
typedef uint32_t TYPE_RINT;
#endif

TYPE_RINT pseudo_random();
TYPE_INTER randadj(TYPE_RINT rn, int pos);

#define PARALLEL_DEGREE 16

inline int sub2ind3(
    int ind0,
    int ind1,
    int ind2,
    int size1,
    int size2
){
    return (ind0 * size1 + ind1) * size2 + ind2;
}

inline int sub2ind4(
    int ind0,
    int ind1,
    int ind2,
    int ind3,
    int size1,
    int size2,
    int size3
){
    return ((ind0 * size1 + ind1) * size2 + ind2) * size3 + ind3;
}

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
);

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
);

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
// );

// void tensor_cont_head_backward(
//     TYPE_DATA *data_in_1,
//     TYPE_DATA *data_in_2,
//     TYPE_GRAD *grad_out,
//     int array_in_size_0,
//     int array_in_size_1,
//     int array_weight_size_1,
//     int shift,
//     TYPE_INTER* max
// );

void tensor_train_forward(
    TYPE_DATA* data_in,
    TYPE_DATA* data_out,
    TYPE_WEIGHT** weights,
    TYPE_DATA* bias,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    float shift[4],
    TYPE_INTER max[4]
);

void tensor_train_input_grad(
    TYPE_DATA* data_in,
    TYPE_DATA* grad_out,
    TYPE_DATA* grad_in,
    TYPE_WEIGHT** weights,
    TYPE_DATA** tmp,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    float shift[4],
    TYPE_INTER max[4]
);

// void tensor_train_weight_grad(
//     TYPE_DATA* data_in,
//     TYPE_DATA* grad_out,
//     TYPE_WEIGHT** weights,
//     TYPE_GRAD** weights_grad,
//     TYPE_DATA** tmp,
//     int input_shape[4],
//     int output_shape[4],
//     int rank[4],
//     int dim,
//     int dim_grad,
//     int shift[6],
//     TYPE_INTER max[6]
// );

// void tensor_train_backward(
//     TYPE_DATA* data_in,
//     TYPE_DATA* grad_out,
//     TYPE_DATA* grad_in,
//     TYPE_WEIGHT** weights,
//     TYPE_GRAD** weights_grad,
//     TYPE_GRAD* bias_grad,
//     int input_shape[4],
//     int output_shape[4],
//     int rank[4],
//     int dim,
//     int shift[14],
//     TYPE_INTER max[14]
// );

void tensor_cont_outer_prod(
    TYPE_DATA *data_in_1, // input
    TYPE_DATA *data_in_2, // output_grad
    TYPE_GRAD *grad_out,
    int* input_shape,
    int* output_shape,
    int dim,
    float shift,
    TYPE_INTER* max
);

void tensor_train_factors_grad(
    TYPE_GRAD* weight_grad,
    TYPE_WEIGHT** weights,
    TYPE_GRAD** factors_grad,
    TYPE_DATA** tmp,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    int dim_grad,
    float shift[6],
    TYPE_INTER max[6]
);

// void relu_inplace(
//     TYPE_DATA* data,
//     int shape
// );

// void relu_backward_inplace(
//     TYPE_DATA* data,
//     TYPE_DATA* grad,
//     int shape
// );

// void softmax_ce_grad(
//     TYPE_DATA* data,
//     TYPE_DATA* grad,
//     unsigned char label,
//     unsigned char num_class
// );