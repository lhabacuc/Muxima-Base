#ifndef TENSOR_H
# define TENSOR_H

# include <vector>
# include <cstddef>

class Tensor
{
	private:
		std::vector<float> _data;
		std::vector<size_t> _shape;

		size_t	index(size_t row, size_t column) const;

	public:
		Tensor(const std::vector<size_t>& shape);

		float&	operator()(size_t row, size_t column);
		const float&	operator()(size_t row, size_t column) const;

		const std::vector<float>& data() const;
		const std::vector<size_t>& shape() const;
		size_t size() const;
};

#endif
