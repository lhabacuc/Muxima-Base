#ifndef SGD_H
# define SGD_H

# include "optimizer.h"

class SGD : public Optimizer
{
	private:
		float	_learning_rate;

	public:
		SGD(float learning_rate);

		void	step(Variable& variable);
		void	zero_grad(Variable& variable);

		float	learning_rate() const;
};

#endif
