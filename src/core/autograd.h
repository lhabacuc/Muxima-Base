#ifndef AUTOGRAD_H
# define AUTOGRAD_H

# include "variable.h"

class Autograd
{
	public:
		static void	add_backward(
			Variable& left,
			Variable& right,
			const Tensor& output_gradient);

		static void	subtract_backward(
			Variable& left,
			Variable& right,
			const Tensor& output_gradient);

		static void	multiply_backward(
			Variable& left,
			Variable& right,
			const Tensor& output_gradient);

		static void	matmul_backward(
			Variable& left,
			Variable& right,
			const Tensor& output_gradient);

		static void	relu_backward(
			Variable& input,
			const Tensor& output_gradient);

		static void	gelu_backward(
			Variable& input,
			const Tensor& output_gradient);

		static void	cross_entropy_backward(
			Variable& logits,
			const Tensor& targets,
			const Tensor& output_gradient);
};

#endif
