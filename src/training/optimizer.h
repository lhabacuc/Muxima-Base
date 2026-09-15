#ifndef OPTIMIZER_H
# define OPTIMIZER_H

# include "../core/variable.h"

class Optimizer
{
	public:
		virtual ~Optimizer();

		virtual void	step(Variable& variable) = 0;
		virtual void	zero_grad(Variable& variable) = 0;
};

#endif
