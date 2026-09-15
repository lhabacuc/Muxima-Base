#ifndef TENSOR_H
# define TENSOR_H

# include <vector>
# include <cstddef>

class Tensor
{
	private:
		std::vector<float> _data;
		std::vector<size_t> _shape;

		size_t	index(const std::vector<size_t>& position) const;

	public:
		Tensor(const std::vector<size_t>& shape);

		float&	at(const std::vector<size_t>& position);
		const float&	at(const std::vector<size_t>& position) const;

		void	reshape(const std::vector<size_t>& new_shape);

		std::vector<float>&	data();
		const std::vector<float>&	data() const;

		const std::vector<size_t>& shape() const;
		size_t size() const;
};

#endif
