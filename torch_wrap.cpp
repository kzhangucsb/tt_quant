#include <torch/extension.h>
#include <vector>
#include <assert.h>
#include "tt_nn.h"

#ifdef QUANTIZE
cnl::from_rep<TYPE_WEIGHT, TYPE_RINT> int2wt;
#define WEIGHT_MASK int2wt(0xf0).
#define WEIGHT_ADJ  int2wt(0x08)
#endif


std::vector<at::Tensor> tt_nn_forward(
    torch::Tensor input_tensor,
    std::vector<torch::Tensor> weights_tensor,
    torch::Tensor bias_tensor,
    torch::Tensor shift
) {
	int in_shape[10];
	int out_shape[10];
	int rank[10];
	int dim = weights_tensor.size();
	TYPE_WEIGHT* weights[10] = {0};
	TYPE_INTER max[100] = {0};
	// dimension
	TORCH_CHECK(dim < 10, "dimension is too high");
	// std::cout << "dim is " << dim << std::endl; 
	TORCH_CHECK(input_tensor.dim() == 2, "input dimension must be 2");
	//std::cout << "shift size " << shift.size(0) << std::endl;
	TORCH_CHECK(shift.dim() == 1, "shift must be 1-d");
	TORCH_CHECK(shift.numel() >= dim, "shift not enough");
	// size
	int in_size = 1;
	int out_size = 1;
	int batchsize = input_tensor.size(0);
	for (int i = 0; i < dim; i++) {
		in_shape[i] = weights_tensor[i].size(2);
		in_size *= in_shape[i];
		out_shape[i] = weights_tensor[i].size(1);
		out_size *= out_shape[i];
		// std::cout << in_shape[i] << ' ' << out_shape[i] << std::endl;
	}
	//std::cout << "insize " << in_size << " outsize " << out_size << std::endl;
	for (int i = 0; i < dim - 1; i++) {
		rank[i] = weights_tensor[i].size(3);
		TORCH_CHECK(weights_tensor[i+1].size(0) == rank[i], "rank mismatch");
	}
	TORCH_CHECK(input_tensor.size(1) == in_size, "input size doesn't match");
	// weight
	for (int i = 0; i < dim; i++) {
		TORCH_CHECK(weights_tensor[i].is_contiguous(), "weights must be contiguous");
		float* wt = weights_tensor[i].data_ptr<float>();
		weights[i] = new TYPE_WEIGHT[weights_tensor[i].numel()];
		for (int j = 0; j < weights_tensor[i].numel(); j++) {
			weights[i][j] = wt[j];
			// #ifdef QUANTIZE
			// weights[i][j] += WEIGHT_ADJ;
			// weights[i][j] &= WEIGHT_MASK;
			// #endif
		}
	}
	// bias
	TORCH_CHECK(bias_tensor.dim() == 1, "bias dimension must be 1");
	TORCH_CHECK(bias_tensor.size(0) == out_size, "bias dimension mismatch");
	TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
	TYPE_DATA* bias = new TYPE_DATA[out_size];
	float* bt = bias_tensor.data_ptr<float>();
	for (int j = 0; j < out_size; j++) {
		bias[j] = bt[j];
	}
	// io
	TYPE_DATA* data_in = new TYPE_DATA[in_size];
	TYPE_DATA* data_out = new TYPE_DATA[out_size];
	//std::vector<int64_t> out_tensor_shape = {batchsize, out_size};
	torch::Tensor ret = torch::zeros({batchsize, out_size});
	//std::cout << ret.sizes() << std::endl;
	TORCH_CHECK(input_tensor.is_contiguous(), "input must be contiguous");

	// compute
	for (int b = 0; b < batchsize; b++){
		float* di = input_tensor[b].data_ptr<float>();
		for (int i = 0; i < in_size; i++) {
			data_in[i] = di[i];
		}
		tensor_train_forward(
			data_in,
			data_out,
			weights,
			bias,
			in_shape,
			out_shape,
			rank,
			dim,
			shift.data_ptr<float>(),
			max
		);
		for (int i = 0; i < out_size; i++) {
			ret[b][i] = float(data_out[i]);
		}
	}
	std::vector<float> max_float(shift.numel());
	for (int i = 0; i < dim; i++) {
		max_float[i] = float(max[i]);
	}
	torch::Tensor max_tensor = torch::tensor(max_float);

	// delete
	for (int i = 0; i < 10; i++) {
		if (weights[i] != 0) {
			delete[] weights[i];
		}
	}
	delete[] data_in;
	delete[] data_out;
	delete[] bias;
  	
	return {ret, max_tensor};
}

std::vector<at::Tensor> tt_nn_backward(
	torch::Tensor grad_out_tensor,
    torch::Tensor input_tensor,
    std::vector<torch::Tensor> weights_tensor,
    torch::Tensor shift
){
	int in_shape[10];
	int out_shape[10];
	int rank[10];
	int dim = weights_tensor.size();
	TYPE_WEIGHT* weights[10] = {0};
	TYPE_GRAD* weights_grad[10] = {0};
	TYPE_INTER max[100] = {0};
	// dimension
	TORCH_CHECK(dim < 10, "dimension is too high");
	// std::cout << "dim is " << dim << std::endl; 
	TORCH_CHECK(input_tensor.dim() == 2, "input dimension must be 2");
	TORCH_CHECK(grad_out_tensor.dim() == 2, "grad out dimension must be 2");
	// size
	int in_size = 1;
	int out_size = 1;
	int batchsize = input_tensor.size(0);
	for (int i = 0; i < dim; i++) {
		in_shape[i] = weights_tensor[i].size(2);
		in_size *= in_shape[i];
		out_shape[i] = weights_tensor[i].size(1);
		out_size *= out_shape[i];
		// std::cout << in_shape[i] << ' ' << out_shape[i] << std::endl;
	}
	//std::cout << "insize " << in_size << " outsize " << out_size << std::endl;
	for (int i = 0; i < dim - 1; i++) {
		rank[i] = weights_tensor[i].size(3);
		TORCH_CHECK(weights_tensor[i+1].size(0) == rank[i], "rank mismatch");
	}
	TORCH_CHECK(input_tensor.size(1) == in_size, "input size doesn't match");
	TORCH_CHECK(grad_out_tensor.size(1) == out_size, "grad out size doesn't match");
	// weight
	for (int i = 0; i < dim; i++) {
		TORCH_CHECK(weights_tensor[i].is_contiguous(), "weights must be contiguous");
		float* wt = weights_tensor[i].data_ptr<float>();
		weights[i] = new TYPE_WEIGHT[weights_tensor[i].numel()];
		for (int j = 0; j < weights_tensor[i].numel(); j++) {
			weights[i][j] = wt[j];
			// #ifdef QUANTIZE
			// weights[i][j] += WEIGHT_ADJ;
			// weights[i][j] &= WEIGHT_MASK;
			// #endif
		}
		weights_grad[i] = new TYPE_GRAD[weights_tensor[i].numel()];
		memset(weights_grad[i], 0, sizeof(TYPE_GRAD) * weights_tensor[i].numel());
	}
	weights[dim] = new TYPE_WEIGHT[weights_tensor[dim - 1].numel()];
	for (int i_0 = 0; i_0 < rank[dim - 2]; i_0++) {
        for (int i_1 = 0; i_1 < out_shape[dim - 1]; i_1++) {
            for (int i_2 = 0; i_2 < in_shape[dim - 1]; i_2++) {
                int ind_i = sub2ind3(i_0, i_1, i_2, out_shape[dim - 1], in_shape[dim - 1]);
                int ind_o = sub2ind3(i_0, i_2, i_1, in_shape[dim - 1], out_shape[dim - 1]);
                weights[dim][ind_o] = weights[dim - 1][ind_i];
            }
        }
    }
	// io
	TYPE_DATA* data_in = new TYPE_DATA[in_size];
	TYPE_DATA* grad_out = new TYPE_DATA[out_size];
	TYPE_DATA* grad_in = new TYPE_DATA[in_size];
	TYPE_GRAD* bias_grad = new TYPE_GRAD[out_size];
	memset(bias_grad, 0, sizeof(TYPE_GRAD) * out_size);
	//std::vector<int64_t> in_tensor_shape = {batchsize, in_size};
	torch::Tensor grad_in_tensor = torch::zeros({batchsize, in_size});
	//std::cout << grad_in_tensor.sizes() << std::endl;
	TORCH_CHECK(input_tensor.is_contiguous(), "input must be contiguous");
	TYPE_DATA* tmp[10] = {0};
	TYPE_GRAD* grad_weight = new TYPE_GRAD[in_size * out_size];
	memset(grad_weight, 0, sizeof(TYPE_GRAD) * in_size * out_size);
	TORCH_CHECK(shift.dim() == 1, "shift must be 1-d");
	TORCH_CHECK(shift.numel() >= dim * 2 + dim * (dim + 1) / 2, "shift not enough");
	// compute
	for (int b = 0; b < batchsize; b++){
		float* di = input_tensor[b].data_ptr<float>();
		float* go = grad_out_tensor[b].data_ptr<float>();
		for (int i = 0; i < in_size; i++) {
			data_in[i] = di[i];
		}
		for (int i = 0; i < out_size; i++) {
			grad_out[i] = go[i];
			bias_grad[i] += grad_out[i];
		}
		// tensor_train_backward(
		// 	data_in,
		// 	grad_out,
		// 	grad_in,
		// 	weights,
		// 	weights_grad,
		// 	bias_grad,
		// 	in_shape,
		// 	out_shape,
		// 	rank,
		// 	dim,
		// 	shift.data<int>(),
		// 	max
		// );
		
		tensor_train_input_grad(
			data_in,
			grad_out,
			grad_in,
			weights,
			tmp,
			in_shape,
			out_shape,
			rank,
			dim,
			shift.data_ptr<float>() + dim,
			max + dim
		);
		
		tensor_cont_outer_prod(
			data_in,
			grad_out,
			grad_weight,
			in_shape,
			out_shape,
			dim,
			shift[dim * 2].item<int>(),
			max + dim * 2
		);

		for (int i = 0; i < in_size; i++) {
			grad_in_tensor[b][i] = float(grad_in[i]);
		}
	}

	// free memory
	for (int i = 0; i < 10; i++) {
		if (tmp[i] != 0) {
			delete[] tmp[i];
			tmp[i] = 0;
		}
	}

	int offset = dim * 2 + 1;
	for (int i = 0; i < dim; i++) {
		tensor_train_factors_grad(
			grad_weight,
			weights,
			weights_grad,
			tmp,
			in_shape,
			out_shape,
			rank,
			dim,
			i,
			shift.data_ptr<float>() + offset,
			max + offset
		);
		if (i == 0) {
			offset += dim - 1;
		}
		else {
			offset += i;
		}
	}


	std::vector<torch::Tensor> ret;
	ret.push_back(grad_in_tensor);
	torch::Tensor bias_grad_tensor = torch::zeros(out_size);
	for (int j = 0; j < out_size; j++) {
		bias_grad_tensor[j] = float(bias_grad[j]);
	}
	ret.push_back(bias_grad_tensor);

	std::vector<float> max_float(shift.numel());
	for (int i = 0; i < shift.numel(); i++) {
		max_float[i] = float(max[i]);
	}
	torch::Tensor max_tensor = torch::tensor(max_float);
	ret.push_back(max_tensor);

	for (int i = 0; i < dim; i++) {
		torch::Tensor wg = torch::zeros_like(weights_tensor[i]);
		torch::Tensor wgv = wg.view({-1});
		for (int j = 0; j < wg.numel(); j++) {
			wgv[j] = float(weights_grad[i][j]);
		}
		ret.push_back(wg);
	}

	delete[] data_in;
	delete[] grad_out;
	delete[] grad_in;
	delete[] bias_grad;
	delete[] grad_weight;
	for (int i = 0; i < 10; i++) {
		if (weights[i] != 0) {
			delete[] weights[i];
		}
		if (weights_grad[i] != 0) {
			delete[] weights_grad[i];
		}
		if (tmp[i] != 0) {
			delete[] tmp[i];
		}
	}
	return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tt_nn_forward, "LLTM forward");
  m.def("backward", &tt_nn_backward, "LLTM backward");
}
