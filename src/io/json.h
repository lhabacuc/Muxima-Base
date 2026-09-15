#ifndef JSON_H
# define JSON_H

# include <nlohmann/json.hpp>

# include <string>

class Json
{
	public:
		static nlohmann::json	load(
			const std::string& path);

		static void	save(
			const std::string& path,
			const nlohmann::json& data);
};

#endif
