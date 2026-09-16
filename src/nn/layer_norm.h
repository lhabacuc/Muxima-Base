#ifndef LAYER_NORM_H
# define LAYER_NORM_H

# include "../core/variable.h"

class LayerNorm
{
	private:
		Variable	_gamma;
		Variable	_beta;
		float	_eps;

		static Tensor	create_parameter(size_t embedding_dim);

	public:
		LayerNorm(size_t embedding_dim, float eps);

		Tensor	forward(const Tensor& input);

		Variable&	gamma();
		Variable&	beta();
};

#endif
