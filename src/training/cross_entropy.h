#ifndef CROSS_ENTROPY_H
# define CROSS_ENTROPY_H

# include "../core/tensor.h"

class CrossEntropy
{
	public:
		static float	forward(
			const Tensor& logits,
			const Tensor& targets);
};

#endif