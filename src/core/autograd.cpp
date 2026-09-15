#include "autograd.h"

#include "math.h"

#include "../training/cross_entropy.h"

#include <cmath>

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

void	Autograd::relu_backward(
	Variable& input,
	const Tensor& output_gradient)
{
	Tensor	gradient(input.value().shape());
	size_t	i;

	i = 0;
	while (i < input.value().size())
	{
		if (input.value().data()[i] > 0.0f)
			gradient.data()[i] = output_gradient.data()[i];
		else
			gradient.data()[i] = 0.0f;
		i++;
	}

	if (input.requires_grad())
	{
		input.gradient() = add(
			input.gradient(),
			gradient);
	}
}

void	Autograd::gelu_backward(
	Variable& input,
	const Tensor& output_gradient)
{
	Tensor	gradient(input.value().shape());
	float	x;
	float	derivative;
	float	sqrt_two_pi;
	size_t	i;

	sqrt_two_pi = std::sqrt(2.0f * 3.14159265358979323846f);

	i = 0;
	while (i < input.value().size())
	{
		x = input.value().data()[i];

		derivative = 0.5f
			* (1.0f + ::erf(
				x / std::sqrt(2.0f)));

		derivative += x
			* std::exp(
				-(x * x) / 2.0f)
			/ sqrt_two_pi;

		gradient.data()[i]
			= output_gradient.data()[i]
			* derivative;

		i++;
	}

	if (input.requires_grad())
	{
		input.gradient() = add(
			input.gradient(),
			gradient);
	}
}

void	Autograd::cross_entropy_backward(
	Variable& logits,
	const Tensor& targets,
	const Tensor& output_gradient)
{
	Tensor	gradient = CrossEntropy::backward(
		logits.value(),
		targets);
	size_t	i;
	float	scale;


	scale = output_gradient.data()[0];

	i = 0;
	while (i < gradient.size())
	{
		gradient.data()[i] *= scale;
		i++;
	}

	if (logits.requires_grad())
	{
		logits.gradient() = add(
			logits.gradient(),
			gradient);
	}
}
