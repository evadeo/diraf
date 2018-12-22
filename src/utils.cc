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

std::vector<int> build_feature_vector(const std::vector<std::vector<int>>& features,
                                      size_t split_index)
{
    std::vector<int> selected_feature;
    for (size_t i = 0; i < features.size(); ++i)
        selected_feature.push_back(features[i][split_index]);
    return selected_feature;
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

int get_label(const std::vector<std::vector<int>>& features, const std::vector<int>& labels,
              int split_value, size_t split_index, const std::function<bool(int, int)>& comp)
{
    for (size_t i = 0; i < features.size(); ++i)
        if (comp(features[i][split_index], split_value))
            return labels[i];

    return -1;
}

int get_label_for_left_split(const std::vector<std::vector<int>>& features,
                             const std::vector<int>& labels, int split_value, size_t split_index)
{
    return get_label(features, labels, split_value, split_index, [](int a, int b) { return a < b; });
}

int get_label_for_right_split(const std::vector<std::vector<int>>& features,
                              const std::vector<int>& labels, int split_value, size_t split_index)
{
    return get_label(features, labels, split_value, split_index, [](int a, int b) { return a >= b; });
}

int get_number_of_elems(std::vector<int> count_label_vector)
{
    return std::accumulate(count_label_vector.begin(), count_label_vector.end(), 0.0);
}

std::vector<std::vector<int>> build_split_feature(
        const std::vector<std::vector<int>>& features,
        int split_value, size_t split_index, const std::function<bool(int, int)>& comp)
{
    std::vector<std::vector<int>> splitted_features;
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (comp(features[i][split_index], split_value))
        {
            std::vector<int> selected_features;
            for (size_t j = 0; j < features[i].size(); ++j)
            {
                if (j == split_index)
                    continue;
                selected_features.push_back(features[i][j]);
            }
            //std::copy(features[i].begin(), features[i].end(), std::back_inserter(splitted_features[i]));
            //splitted_features[i].erase(splitted_features[i] + split_index);
            splitted_features.push_back(selected_features);
        }
    }
    return splitted_features;
}

std::vector<std::vector<int>> build_split_feature_left(
        const std::vector<std::vector<int>>& features, int split_value, size_t split_index)
{
    return build_split_feature(features, split_value, split_index,
                [](int a, int b) { return a < b; });
}

std::vector<std::vector<int>> build_split_feature_right(
        const std::vector<std::vector<int>>& features, int split_value, size_t split_index)
{
    return build_split_feature(features, split_value, split_index,
                [](int a, int b) { return a >= b; });
}

std::vector<int> build_split_labels(
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels, int split_value, size_t split_index,
        const std::function<bool(int, int)>& comp)
{
    std::vector<int> splitted_labels;
    for (size_t i = 0; i < features.size(); ++i)
        if (comp(features[i][split_index], split_value))
            splitted_labels.push_back(labels[i]);
    return splitted_labels;
}

std::vector<int> build_split_labels_left(
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels, int split_value, size_t split_index)
{

    return build_split_labels(features, labels, split_value, split_index,
                [](int a, int b) { return a < b; });
}

std::vector<int> build_split_labels_right(
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels, int split_value, size_t split_index)
{
    return build_split_labels(features, labels, split_value, split_index,
                [](int a, int b) { return a >= b; });
}
