#include <map>
#include <numeric>
#include <set>
#include "error_functions.hh"
#include "utils.hh"

err_func get_error_function(const std::string& criterion)
{
    //A compléter si jamais on rajoute des fonction supplémentaires pour essayer.
    static const std::map<std::string, err_func>
        error_functions {
            { "gini", total_gini_error },
            { "default", total_gini_error },
    };

    auto err_func = error_functions.find(criterion);
    if (err_func != error_functions.end())
        return err_func->second;
    else
        return error_functions.find("default")->second;
}

std::vector<int> get_count_by_label(std::vector<int> feature_values,
                                     std::vector<int> labels,
                                     int split_value,
                                     const std::function<bool(int, int)>& comp)
{
    std::set<int> unique_labels(labels.begin(), labels.end());
    std::vector<int> labels_count(unique_labels.size());

    for (size_t i = 0; i < feature_values.size(); ++i)
        if (comp(feature_values[i], split_value))
            labels_count[labels[i]] += 1;

    return labels_count;
}

int get_label(const std::vector<int>& features, const std::vector<int>& labels,
              int split_value, const std::function<bool(int, int)>& comp)
{
    for (size_t i = 0; i < features.size(); ++i)
        if (comp(features[i], split_value))
            return labels[i];

    return -1;
}

int get_label_for_left_split(const std::vector<int>& features,
                             const std::vector<int>& labels, int split_value)
{
    return get_label(features, labels, split_value, [](int a, int b) { return a < b; });
}

int get_label_for_right_split(const std::vector<int>& features,
                              const std::vector<int>& labels, int split_value)
{
    return get_label(features, labels, split_value, [](int a, int b) { return a >= b; });
}

int get_number_of_elems(std::vector<int> count_label_vector)
{
    return std::accumulate(count_label_vector.begin(), count_label_vector.end(), 0.0);
}
