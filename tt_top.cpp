#include "tt_nn.h"
#include <assert.h>
// #include <iostream>
// #include <string.h>

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
){
    TYPE_DATA* tmp[4] = {0};
    for (int dim_mut = dim - 1; dim_mut >= 0; dim_mut--) {
        int in_shape_0 = 1;
        int in_shape_2 = 1;
        int rank_left;
        int rank_right;
        TYPE_DATA* opr_in;
        TYPE_DATA* opr_out;
        for (int i = 0; i < dim_mut; i++) {
            in_shape_0 *= input_shape[i];
        }
        for (int i = dim_mut + 1; i < dim; i++) {
            in_shape_2 *= output_shape[i];
        }

        if (dim_mut == dim - 1) {
            opr_in = data_in;
            rank_right = 1;
        }
        else {
            opr_in = tmp[dim_mut];
            rank_right = rank[dim_mut];
        }
        if (dim_mut == 0) {
            opr_out = data_out;
            rank_left = 1;
        }
        else {
            rank_left = rank[dim_mut - 1];
            tmp[dim_mut - 1] = new TYPE_DATA[in_shape_0 * in_shape_2 * rank_left * output_shape[dim_mut]];
            opr_out = tmp[dim_mut - 1];
        }

        if (dim_mut == dim - 1) {
            tensor_cont_last(
                opr_in,
                opr_out,
                weights[dim_mut],
                in_shape_0,
                1,
                input_shape[dim_mut],
                rank_left * output_shape[dim_mut],
                shift[dim_mut],
                max + dim_mut
            );
        }
        else{
            tensor_cont_mid(
                opr_in,
                opr_out,
                weights[dim_mut],
                in_shape_0,
                input_shape[dim_mut] * rank_right,
                in_shape_2,
                rank_left * output_shape[dim_mut],
                1,
                shift[dim_mut],
                max + dim_mut
            );
        }
    }
    int output_size = 1;
    for (int i = 0; i < dim; i++) {
        output_size *= output_shape[i];
    }
    for (int i = 0; i < output_size; i++) {
        data_out[i] += bias[i];
    }
    for (int i = 0; i < 4; i++) {
        if (tmp[i] != 0) {
            delete[] tmp[i];
        }
    }
}



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
){
    TYPE_GRAD* opr_in;
    TYPE_GRAD* opr_out;
    int offset = 0;

    if (dim_grad == 0) {
        for(int dim_mut = dim - 1; dim_mut >= 1; dim_mut--) {
            int in_shape_0 = 1;
            for (int i = 0; i < dim_mut; i++) {
                in_shape_0 *= input_shape[i];
                in_shape_0 *= output_shape[i];
            }

            int rank_left = rank[dim_mut - 1];
            int rank_right;
            if (dim_mut < dim - 1) {
                rank_right = rank[dim_mut];
                opr_in = tmp[dim_mut - 1];
            } // dim_mut < dim - 1
            else {
                rank_right = 1;
                opr_in = weight_grad;
            } // dim_mut == dim - 1
            if (dim_mut > 1){
                tmp[dim_mut - 2] = new TYPE_GRAD[in_shape_0 * rank_left];
                opr_out = tmp[dim_mut - 2];
            }
            else {
                opr_out = factors_grad[0];
            }
            
            tensor_cont_last(
                opr_in,
                opr_out,
                weights[dim_mut],
                in_shape_0,
                1,
                input_shape[dim_mut] * output_shape[dim_mut] * rank_right,
                rank_left,
                shift[offset],
                max + offset
            );
            offset++;
        }
    }

    for (int dim_mut = 0; dim_mut < dim_grad; dim_mut++) {
        int in_shape_2 = 1;
        int rank_left;
        if (dim_mut == 0) {
            rank_left = 1;
        }
        else {
            rank_left = rank[dim_mut - 1];
        }
        int rank_right = rank[dim_mut];
        // int w_shape_0;
        for (int i = dim_mut + 1; i <= dim_grad; i++) {
            in_shape_2 *= input_shape[i];
            in_shape_2 *= output_shape[i];
        }
        if (dim_grad < dim - 1) {
            in_shape_2 *= rank[dim_grad];
        }
                 
        if (dim_grad < dim - 1 || dim_mut != 0) {
            opr_in = tmp[dim_grad - dim_mut - 1];
        }
        else {
            opr_in = weight_grad;
        }
       


        if (dim_mut == dim_grad - 1) {
            opr_out = factors_grad[dim_grad];
        }
        else {
            if (tmp[dim_grad - dim_mut - 2] != 0) {
                delete[] tmp[dim_grad - dim_mut - 2];
            }
            tmp[dim_grad - dim_mut - 2] = new TYPE_DATA[rank_right * in_shape_2];
            opr_out = tmp[dim_grad - dim_mut - 2];
        }

        tensor_cont_mid(
            opr_in,
            opr_out,
            weights[dim_mut],
            1,
            rank_left * input_shape[dim_mut] * output_shape[dim_mut],
            in_shape_2,
            1,
            rank_right,
            shift[offset],
            max + offset
        );
        offset++;
    } // for (int dim_mut = dim_grad + 1; dim_mut < dim; dim_mut++)
}



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
){
    TYPE_DATA* opr_in;
    TYPE_DATA* opr_out;

    for (int dim_mut = 0; dim_mut < dim; dim_mut++) {
        int in_shape_0 = 1;
        int in_shape_2 = 1;
        int rank_left;
        int rank_right;
        for (int i = 0; i < dim_mut; i++) {
            in_shape_0 *= input_shape[i];
        }
        for (int i = dim_mut + 1; i < dim; i++) {
            in_shape_2 *= output_shape[i];
        }

        if (dim_mut == dim - 1) {
            opr_out = grad_in;
            rank_right = 1;
        }
        else {
            rank_right = rank[dim_mut];
            if (tmp[dim_mut] == 0){
                tmp[dim_mut] = new TYPE_DATA[in_shape_0 * in_shape_2 * input_shape[dim_mut] * rank_right];
            }
            opr_out = tmp[dim_mut];
        }
        if (dim_mut == 0) {
            opr_in = grad_out;
            rank_left = 1;
        }
        else {
            opr_in = tmp[dim_mut - 1];
            rank_left = rank[dim_mut - 1];
        }

        if (dim_mut == dim - 1) {
            tensor_cont_last(
                opr_in,
                opr_out,
                weights[dim],
                in_shape_0,
                rank_left,
                output_shape[dim_mut],
                input_shape[dim_mut],
                shift[dim_mut],
                max + dim_mut
            );
        }
        else{
            tensor_cont_mid(
                opr_in,
                opr_out,
                weights[dim_mut],
                in_shape_0,
                rank_left * output_shape[dim_mut],
                in_shape_2,
                1,
                input_shape[dim_mut] * rank_right,
                shift[dim_mut],
                max + dim_mut
            );
        }
    }
}

// ******* below should not be used ******************

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
// ){
//     TYPE_DATA* opr_in;
//     TYPE_DATA* opr_out;
//     int offset = 0;
//     if (dim_grad == 0) {
//         for (int dim_mut = dim - 1; dim_mut > 0; dim_mut--) {
//             int in_shape_0 = 1;
//             int in_shape_2 = 1;
//             for (int i = 0; i < dim_mut; i++) {
//                 in_shape_0 *= input_shape[i];
//             }
//             for (int i = dim_mut + 1; i < dim; i++) {
//                 in_shape_2 *= output_shape[i];
//             }
            
//             if (tmp[dim_mut - 1] != 0) {
//                 delete[] tmp[dim_mut - 1];
//             }
//             tmp[dim_mut - 1] = new TYPE_DATA[in_shape_0 * in_shape_2 * rank[dim_mut - 1] * output_shape[dim_mut]];
//             opr_out = tmp[dim_mut - 1];
//             if (dim_mut == dim - 1) {  
//                 tensor_cont_last(
//                     data_in,
//                     opr_out,
//                     weights[dim_mut],
//                     in_shape_0,
//                     1,
//                     input_shape[dim_mut],
//                     rank[dim_mut - 1] * output_shape[dim_mut],
//                     shift[offset],
//                     max + offset
//                 );
//                 offset++;
//             } // dim_mut == dim - 1
//             else{
//                 tensor_cont_mid(
//                     tmp[dim_mut],
//                     opr_out,
//                     weights[dim_mut],
//                     in_shape_0,
//                     input_shape[dim_mut] * rank[dim_mut],
//                     in_shape_2,
//                     rank[dim_mut - 1] * output_shape[dim_mut],
//                     1,
//                     shift[offset],
//                     max + offset
//                 );
//                 offset++;
//             } // dim_mul < dim - 1
//         } // for (int dim_mut = dim - 1; dim_mut > 0; dim_mut--)
//         int in_shape_3 = 1;
//         for (int i = 1; i < dim; i++) {
//             in_shape_3 *= output_shape[i];
//         }
//         tensor_cont_end_backward(
//             tmp[0],
//             grad_out,
//             weights_grad[0],
//             1, 
//             1, 
//             input_shape[0] * rank[0],
//             in_shape_3,
//             output_shape[0],
//             shift[offset],
//             max + offset
//         );
//         offset++;
//     } // dim_grad == 0
        
//     // dim_grad > 0
//     else{  
//         for (int dim_mut = dim_grad + 1; dim_mut < dim; dim_mut++) {
//             int in_shape_0 = 1;
//             int in_shape_2 = 1;
//             int rank_left;
//             int rank_right;
//             int w_shape_0;
//             for (int i = 0; i < dim_grad; i++) {
//                 in_shape_0 *= input_shape[i];
//             }
//             in_shape_0 *= rank[dim_grad - 1] * output_shape[dim_grad];
//             if (dim_mut > dim_grad + 1) {
//                 in_shape_0 *= rank[dim_grad];
//             }
//             for (int i = dim_grad + 1; i < dim_mut; i++) {
//                 in_shape_0 *= input_shape[i];
//             }
//             for (int i = dim_mut + 1; i < dim; i++) {
//                 in_shape_2 *= output_shape[i];
//             }
                     
//             if (dim_mut == dim - 1) {
//                 rank_right = 1;
//             }
//             else {
//                 rank_right = rank[dim_mut];
//             }
//             opr_in = tmp[dim_mut - 2];
//             if (tmp[dim_mut - 1] != 0) {
//                 delete [] tmp[dim_mut - 1];
//             }
//             if (dim_mut == dim_grad + 1) {
//                 rank_left = 1;
//                 w_shape_0 = rank[dim_mut - 1];
//             }
//             else {
//                 rank_left = rank[dim_mut - 1];
//                 w_shape_0 = 1;
//             }
//             tmp[dim_mut - 1] = new TYPE_DATA[in_shape_0 * in_shape_2 * w_shape_0 * input_shape[dim_mut] * rank_right];
//             opr_out = tmp[dim_mut - 1];

//             if (dim_mut == dim - 1) { 
//                 tensor_cont_last(
//                     opr_in,
//                     opr_out,
//                     weights[dim],
//                     in_shape_0,
//                     rank_left,
//                     output_shape[dim_mut],
//                     w_shape_0 * input_shape[dim_mut],
//                     shift[offset],
//                     max + offset
//                 );
//                 offset++;
//             }
//             else{
//                 tensor_cont_mid(
//                     opr_in,
//                     opr_out,
//                     weights[dim_mut],
//                     in_shape_0,
//                     rank_left * output_shape[dim_mut],
//                     in_shape_2,
//                     w_shape_0,
//                     input_shape[dim_mut] * rank_right,
//                     shift[offset],
//                     max + offset
//                 );
//                 offset++;
//             }
//         } // for (int dim_mut = dim_grad + 1; dim_mut < dim; dim_mut++)
//         int in_shape_0 = 1;
//         int in_shape_3 = 1;
//         for (int i = 0; i < dim_grad; i++) {
//             in_shape_0 *= input_shape[i];
//         }
//         for (int i = dim_grad + 1; i < dim; i++) {
//             in_shape_3 *= input_shape[i];
//         }
//         int rank_right;
//         if (dim_grad == dim - 1){
//             rank_right = 1;
//         }
//         else {
//             rank_right = rank[dim_grad];
//         }
//         if (dim_grad == dim - 1) {
//             tensor_cont_head_backward(
//                 tmp[dim - 2],
//                 data_in,
//                 weights_grad[dim_grad],
//                 in_shape_0,
//                 rank[dim_grad - 1] * output_shape[dim_grad],
//                 input_shape[dim_grad],
//                 shift[offset],
//                 max + offset
//             );
//             offset++;
//         }
//         else{
//             tensor_cont_end_backward(
//                 tmp[dim - 2],
//                 data_in,
//                 weights_grad[dim_grad],
//                 in_shape_0,
//                 rank[dim_grad - 1] * output_shape[dim_grad],
//                 rank_right,
//                 in_shape_3,
//                 input_shape[dim_grad],
//                 shift[offset],
//                 max + offset
//             );
//             offset++;
//         }
//     }// dim_grad > 0
    
// }

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
// ){
//     int offset = dim;
//     TYPE_DATA* tmp[4] = {0};
//     tensor_train_input_grad(
//         data_in,
//         grad_out,
//         grad_in,
//         weights,
//         tmp,
//         input_shape,
//         output_shape,
//         rank,
//         dim,
//         shift + offset,
//         max + offset
//     );
//     offset += dim;
//     for (int dim_grad = dim - 1; dim_grad >= 0; dim_grad--) {
//         tensor_train_weight_grad(
//             data_in,
//             grad_out, 
//             weights,
//             weights_grad,
//             tmp,
//             input_shape,
//             output_shape,
//             rank,
//             dim,
//             dim_grad,
//             shift + offset,
//             max + offset
//         );
//         offset += (dim - dim_grad);
//     }
//     for (int i = 0; i < 4; i++) {
//         if (tmp[i] != 0) {
//             delete[] tmp[i];
//         }
//     }
// }

