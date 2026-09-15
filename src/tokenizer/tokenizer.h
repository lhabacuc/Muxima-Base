#ifndef TOKENIZER_H
# define TOKENIZER_H

# include "../io/json.hpp"

# include <map>
# include <string>
# include <vector>

class Tokenizer
{
	private:
		std::map<std::string, size_t>	_vocab;
		std::map<std::string, size_t>	_merge_ranks;
		std::map<size_t, std::string>	_id_to_token;

		size_t	_unknown_id;
		size_t	_pad_id;
		size_t	_bos_id;
		size_t	_eos_id;

		std::vector<std::string>	split_text(
			const std::string& text) const;

		std::vector<std::string>	split_word(
			const std::string& word) const;

		std::vector<std::string>	apply_bpe(
			const std::string& word) const;

		std::string	merge_key(
			const std::string& left,
			const std::string& right) const;

	public:
		Tokenizer();

		void	load(
			const std::string& path);

		void	save(
			const std::string& path) const;

		std::vector<size_t>	encode(
			const std::string& text) const;

		std::string	decode(
			const std::vector<size_t>& tokens) const;

		size_t	vocab_size() const;

		size_t	unknown_id() const;
		size_t	pad_id() const;
		size_t	bos_id() const;
		size_t	eos_id() const;
};

#endif
