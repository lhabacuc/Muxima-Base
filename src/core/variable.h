#ifndef VARIABLE_H
# define VARIABLE_H

# include "tensor.h"

class Variable
{
	private:
		Tensor	_value;
		Tensor	_gradient;
		bool	_requires_grad;

	public:
		Variable(
			const Tensor& value,
			bool requires_grad);

		Tensor&	value();
		Tensor&	gradient();

		const Tensor&	value() const;
		const Tensor&	gradient() const;

		bool	requires_grad() const;

		void	zero_grad();
};

#endif
