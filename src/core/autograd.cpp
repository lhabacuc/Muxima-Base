#include "autograd.h"

#include "math.h"

void	Autograd::add_backward(
	Variable& left,
	Variable& right,
	const Tensor& output_gradient)
{
	if (left.requires_grad())
	{
		left.gradient() = add(
			left.gradient(),
			output_gradient);
	}

	if (right.requires_grad())
	{
		right.gradient() = add(
			right.gradient(),
			output_gradient);
	}
}

void	Autograd::subtract_backward(
	Variable& left,
	Variable& right,
	const Tensor& output_gradient)
{
	if (left.requires_grad())
	{
		left.gradient() = add(
			left.gradient(),
			output_gradient);
	}

	if (right.requires_grad())
	{
		right.gradient() = subtract(
			right.gradient(),
			output_gradient);
	}
}

void	Autograd::multiply_backward(
	Variable& left,
	Variable& right,
	const Tensor& output_gradient)
{
	Tensor	left_gradient = multiply(
		output_gradient,
		right.value());
	Tensor	right_gradient = multiply(
		output_gradient,
		left.value());

	if (left.requires_grad())
	{
		left.gradient() = add(
			left.gradient(),
			left_gradient);
	}

	if (right.requires_grad())
	{
		right.gradient() = add(
			right.gradient(),
			right_gradient);
	}
}

void	Autograd::matmul_backward(
	Variable& left,
	Variable& right,
	const Tensor& output_gradient)
{
	Tensor	left_gradient = matmul(
		output_gradient,
		transpose(right.value()));
	Tensor	right_gradient = matmul(
		transpose(left.value()),
		output_gradient);

	if (left.requires_grad())
	{
		left.gradient() = add(
			left.gradient(),
			left_gradient);
	}

	if (right.requires_grad())
	{
		right.gradient() = add(
			right.gradient(),
			right_gradient);
	}
}
