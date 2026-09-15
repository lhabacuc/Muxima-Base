#ifndef FEED_FORWARD_H
# define FEED_FORWARD_H

# include "../core/tensor.h"
# include "linear.h"

class FeedForward
{
	private:
		size_t	_embedding_dim;
		size_t	_hidden_dim;

		Linear	_up;
		Linear	_down;

	public:
		FeedForward(size_t embedding_dim, size_t hidden_dim);

		Tensor	forward(const Tensor& input);

		Linear&	up();
		Linear&	down();
};

#endif