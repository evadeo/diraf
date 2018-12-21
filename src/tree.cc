#include "tree.hh"
#include "utils.hh"
#include <algorithm>
#include <iterator>
#include <iostream>

static int get_split_value(const std::vector<int>& feature_values)
{
    const auto max_index = std::max_element(feature_values.begin(), feature_values.end());
    const auto min_index = std::min_element(feature_values.begin(), feature_values.end());

    //TODO: update this
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(*min_index, *max_index);

    return dis(gen);
}

/*
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
*/
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
    std::set<int> unique_labels(labels.begin(), labels.end());
    std::vector<int> labels_count(unique_labels.size());

    for (size_t i = 0; i < feature_values.size(); ++i)
        if (comp(feature_values[i], threshold))
            labels_count[labels[i]] += 1;

    return labels_count;
}

DecisionTree::Node::Node(int feature_index_split, int threshold, int label, bool is_leaf)
    : left_(nullptr)
    , right_(nullptr)
    , feature_index_split_(feature_index_split)
    , threshold_(threshold)
    , label_(label)
    , is_leaf_(is_leaf)
{}

std::shared_ptr<DecisionTree::Node> DecisionTree::build_node(
        std::vector<std::vector<int>> features,
        const std::vector<int>& labels,
        std::vector<int> features_index,
        const std::function<float(std::vector<int>)>& err_function)
{

    if (features.size() == 0)
        return nullptr;

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
        int split_value = get_split_value(features[f_index]);
        auto left_split_class = get_number_by_label(features[f_index],
                                                    labels, split_value,
                                                    [](int a, int b) { return a < b; });

        auto right_split_class = get_number_by_label(features[f_index],
                                                    labels, split_value,
                                                    [](int a, int b) { return a >= b; });

        float err_left = err_function(left_split_class);
        float err_right = err_function(right_split_class);
        float total_err = ((get_number_of_elems(left_split_class) / features[f_index].size()) * err_left)
                          + ((get_number_of_elems(right_split_class) / features[f_index].size()) * err_right);
        if (total_err <= g_err)
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
    
    int real_index = features_index[g_split_index];
    features_index.erase(features_index.begin() + g_split_index);
    
    auto node = std::make_unique<DecisionTree::Node>(real_index, g_split_value, 0, false);

    features.erase(features.begin() + g_split_index);

    std::cout << "End of feature split: " << std::endl
              << "g_split_index: " << g_split_index << " | g_split_value: " << g_split_value
              << " | g_left_error: " << g_left_error << " | g_right_error: " << g_right_error << std::endl;

    if (g_left_error == 0)
    {
        size_t label_index = get_label_index(g_l_split_class);
        node->left_ = std::make_unique<DecisionTree::Node>(real_index, g_split_value,
                                                           label_index, true);
    }
    else
    {
        node->left_ = build_node(features, labels, features_index, err_function);
    }

    if (g_right_error == 0)
    {
        size_t label_index = get_label_index(g_r_split_class);
        node->right_ = std::make_unique<DecisionTree::Node>(g_split_index, g_split_value,
                                                            label_index, true);
    }
    else
    {
        node->right_ = build_node(features, labels, features_index, err_function);
    }

    return node;
}

int DecisionTree::predict(const std::vector<int>& elem) const
{
    auto root = root_;
    while (!root->is_leaf_)
    {
        bool left = elem[root->feature_index_split_] < root->threshold_;
        if (left)
            root = root->left_;
        else
            root = root->right_;
    }
    return root->label_;
}


DecisionTree::DecisionTree(std::vector<std::vector<int>> features,
                           const std::vector<int>& labels,
                           std::vector<int> features_index,
                           const std::function<float(std::vector<int>)>& err_function)
{
    this->root_ = build_node(features, labels, features_index, err_function);
}
