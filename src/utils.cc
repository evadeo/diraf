#include <map>
#include <numeric>
#include "error_functions.hh"
#include "utils.hh"

std::function<float(std::vector<int>)> get_error_function(const std::string& criterion)
{
    //A compléter si jamais on rajoute des fonction supplémentaires pour essayer.
    static const std::map<std::string, std::function<float(std::vector<int>)>>
        error_functions {
            { "gini", gini_error },
            { "default", gini_error},
    };

    auto err_func = error_functions.find(criterion);
    if (err_func != error_functions.end())
        return err_func->second;
    else
        return error_functions.find("default")->second;
}

int get_number_of_elems(std::vector<int> count_label_vector)
{
    return std::accumulate(count_label_vector.begin(), count_label_vector.end(), 0.0);
}
