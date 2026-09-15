#ifndef LINEAR_H
# define LINEAR_H

# include "../core/tensor.h"

class Linear
{
	private:
		Tensor	_weights;
		Tensor	_bias;

		static Tensor	create_weights(size_t input_size, size_t output_size);
		static Tensor	create_bias(size_t output_size);

	public:
		Linear(size_t input_size, size_t output_size);

		Tensor	forward(const Tensor& input);

		Tensor&	weights();
		Tensor&	bias();
};

#endif
