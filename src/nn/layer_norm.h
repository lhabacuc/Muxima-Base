#ifndef LAYER_NORM_H
# define LAYER_NORM_H

# include "../core/tensor.h"

class LayerNorm
{
	private:
		Tensor	_gamma;
		Tensor	_beta;
		float	_eps;

		static Tensor	create_parameter(size_t embedding_dim);

	public:
		LayerNorm(size_t embedding_dim, float eps);

		Tensor	forward(const Tensor& input);

		Tensor&	gamma();
		Tensor&	beta();
};

#endif
