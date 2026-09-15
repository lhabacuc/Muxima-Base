#ifndef MATH_H
# define MATH_H

# include "tensor.h"

Tensor	add(const Tensor& a, const Tensor& b);
Tensor	subtract(const Tensor& a, const Tensor& b);
Tensor	multiply(const Tensor& a, const Tensor& b);
Tensor	scalar_multiply(const Tensor& tensor, float scalar);
Tensor	matmul(const Tensor& a, const Tensor& b);
Tensor	transpose(const Tensor& tensor);
Tensor	softmax(const Tensor& tensor);

float	sum(const Tensor& tensor);
float	mean(const Tensor& tensor);
Tensor	exp(const Tensor& tensor);
Tensor	log(const Tensor& tensor);
Tensor	sqrt(const Tensor& tensor);

Tensor	relu(const Tensor& tensor);
Tensor	gelu(const Tensor& tensor);

#endif
