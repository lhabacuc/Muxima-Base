#ifndef MATH_H
# define MATH_H

# include "tensor.h"

Tensor	add(const Tensor& a, const Tensor& b);
Tensor	subtract(const Tensor& a, const Tensor& b);
Tensor	multiply(const Tensor& a, const Tensor& b);
Tensor	scalar_multiply(const Tensor& tensor, float scalar);

#endif
