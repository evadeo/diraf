#include "tree.hh"

#include <algorithm>
#include <iterator>
#include <random>
#include <set>

static int get_split_value(const std::vector<int>& feature_values,
                           const std::vector<int>& labels)
{
    const auto max_index = std::max_element(feature_values.begin(), feature_values.end());
    const auto min_index = std::min_element(feature_values.begin(), feature_values.end());

    //TODO: update this
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(feature_values[*min_index], feature_values[*max_index]);

    return dis(gen);
}

static std::vector<std::vector<int>> update_features(
        const std::vector<std::vector<int>>& features,
        int split_feature,
        int split_value,
        const std::function<bool(int, int)>& split_function)
{
    std::vector<std::vector<int>> filtered_features(features.size(), std::vector<int>(0));

    for (size_t i = 0; i < features[split_feature].size(); ++i)
    {
        if (split_function(features[split_feature][i], split_value))
        {
            for (size_t j = 0; j < features.size(); ++j)
                filtered_features[j].push_back(features[j][i]);
        }
    }

    return filtered_features;
}

static std::vector<int> update_labels(
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        int split_feature,
        int split_value,
        const std::function<bool(int, int)>& split_function)
{
    std::vector<int> filtered_labels;

    for (size_t i = 0; i < features[split_feature].size(); ++i)
        if (split_function(features[split_feature][i], split_value))
                filtered_labels.push_back(labels[i]);

    return filtered_labels;
}

static ssize_t get_label_index(std::vector<int> labels_count)
{
    for (size_t i = 0; i < labels_count.size(); ++i)
        if (labels_count[i] != 0)
            return i;
    return -1;
}

static std::vector<int> get_number_by_label(std::vector<int> feature_values,
                                            std::vector<int> labels, int threshold,
                                            std::function<bool(int, int)> comp)
{
    std::set<int> unique_labels = std::set<int>(labels.begin(), labels.end());
    std::vector<int> labels_count = std::vector<int>(unique_labels.size());
    for (size_t i = 0; i < feature_values.size(); ++i)
        if (comp(feature_values[i], threshold))
            labels_count[labels[i]] += 1;

    return labels_count;
}

DecisionTree::Node::Node(int feature_index_splitted, int threshold, int label, bool is_leaf)
    : left_(nullptr)
    , right_(nullptr)
    , feature_index_splitted_(feature_index_splitted)
    , threshold_(threshold)
    , label_(label)
    , is_leaf_(is_leaf)
{}

std::unique_ptr<DecisionTree::Node> DecisionTree::build_node(
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        const std::function<float(std::vector<int>, std::vector<int>)>& err_function,
        int n_estimators)
{

    float g_err = 1;
    size_t g_split_index;
    int g_split_value;
    float g_left_error;
    float g_right_error;
    //Probablement à supprimer mais pour faciliter les choses,
    //je garde les meilleurs count pour créer le noeud ensuite
    std::vector<int> g_l_split_class;
    std::vector<int> g_r_split_class;
    for (size_t f_index = 0; f_index < features.size(); ++f_index)
    {
        int split_value = get_split_value(features[f_index], labels);

        auto left_split_class = get_number_by_label(features[f_index],
                                                    labels, split_value,
                                                    [](int a, int b) { return a < b; });

        auto right_split_class = get_number_by_label(features[f_index],
                                                    labels, split_value,
                                                    [](int a, int b) { return a >= b; });

        float err_left = err_function(left_split_class, labels);
        float err_right = err_function(right_split_class, labels);
        float total_err = ((left_split_class.size() / features[f_index].size()) * err_left)
                          + ((right_split_class.size() / features[f_index].size()) * err_right);
        if (total_err < g_err)
        {
            g_err = total_err;
            g_split_index = f_index;
            g_split_value = split_value;
            g_left_error = err_left;
            g_right_error = err_right;
            g_l_split_class = left_split_class;
            g_r_split_class = right_split_class;
        }
    }
    
    auto node = std::make_unique<DecisionTree::Node>(g_split_index, g_split_value, 0, false);

    //TODO: peut-être évité la recréation total d'un vecteur en séparant in place (hint: std::remove_if) /
    //      retourner une paire des vecteurs (peut être plus compréhensible) /
    //      4 appels pour faire quasiment les mêmes parcours, peut-être regrouper dans un objet le retour
    auto left_features = update_features(features, g_split_index, g_split_value,
                                         [](int a, int b) { return a < b; });
    auto left_labels = update_labels(features, labels, g_split_index, g_split_value,
                                     [](int a, int b) { return a < b; });
    auto right_features = update_features(features, g_split_index, g_split_value,
                                          [](int a, int b) { return a >= b; });
    auto right_labels = update_labels(features, labels, g_split_index, g_split_value,
                                    [](int a, int b) { return a >= b; });

    if (g_left_error == 0)
    {
        size_t label_index = get_label_index(g_l_split_class);
        node->left_ = std::make_unique<DecisionTree::Node>(g_split_index, g_split_value,
                                                           label_index, true);
    }
    else
        node->left_ = build_node(left_features, left_labels, err_function, n_estimators);

    if (g_right_error == 0)
    {
        size_t label_index = get_label_index(g_r_split_class);
        node->right_ = std::make_unique<DecisionTree::Node>(g_split_index, g_split_value,
                                                            label_index, true);
    }
    else
        node->right_ = build_node(right_features, right_labels, err_function, n_estimators);

    return node;
}

DecisionTree::DecisionTree(const std::vector<std::vector<int>>& features,
                           const std::vector<int>& labels,
                           const std::function<float(std::vector<int>, std::vector<int>)>& err_function,
                           int n_estimators)
{
    this->root_ = build_node(features, labels, err_function, n_estimators);
}
