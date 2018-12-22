#include <algorithm>
#include <iterator>
#include <iostream>
#include "error_struct.hh"
#include "tree.hh"
#include "utils.hh"

static Error find_best_split(
        const std::vector<int>& feature_values,
        const std::vector<int>& labels,
        size_t split_index,
        const err_func& error_function)
{
    Error final_error;
    for (size_t i = 0; i < feature_values.size(); ++i)
    {
        Error err = error_function(feature_values, labels, feature_values[i], split_index);
        if (err < final_error)
            final_error = err;
    }
    return final_error;
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
        const err_func& err_function)
{
    Error g_err;
    for (size_t f_index = 0; f_index < features[0].size(); ++f_index)
    {
        auto selected_feature = build_feature_vector(features, f_index);
        Error err = find_best_split(selected_feature, labels, f_index, err_function);
        if (err < g_err)
            g_err = err;
    }
    
    int real_index = features_index[g_err.split_index_];
    features_index.erase(features_index.begin() + g_err.split_index_);
    
    auto node = std::make_shared<DecisionTree::Node>(real_index, g_err.split_value_, 0, false);

    //for (size_t i = 0; i < features.size(); ++i)
     //   features[i].erase(features[i].begin() + g_err.split_index_);

    std::cout << "End of feature split: " << std::endl
              << "g_split_index: " << g_err.split_index_ << " | g_split_value: " << g_err.split_value_
              << " | g_left_error: " << g_err.err_left_ << " | g_right_error: " << g_err.err_right_ << std::endl;

    if (g_err.err_left_ == 0)
    {
        int label = get_label_for_left_split(features, labels, g_err.split_value_, g_err.split_index_);
        node->left_ = std::make_shared<DecisionTree::Node>(real_index, g_err.split_value_, label, true);
    }
    else
    {
        auto left_features = build_split_feature_left(features, g_err.split_value_, g_err.split_index_);
        auto left_labels = build_split_labels_left(features, labels, g_err.split_value_, g_err.split_index_);
        
        //TODO: EMERGENCY EXIT, THIS NEED TO BE CHANGED
        if (features_index.size() == 0 || left_features.size() == 0)
            node->left_ = std::make_shared<DecisionTree::Node>(0, 0, labels[0], true);
        else
        {
           node->left_ = build_node(left_features, left_labels, features_index, err_function);
        }
    }

    if (g_err.err_right_ == 0)
    {
        int label = get_label_for_right_split(features, labels, g_err.split_value_, g_err.split_index_);
        node->right_ = std::make_shared<DecisionTree::Node>(real_index, g_err.split_value_, label, true);
    }
    else
    {
        auto right_features = build_split_feature_right(features, g_err.split_value_, g_err.split_index_);
        auto right_labels = build_split_labels_right(features, labels, g_err.split_value_, g_err.split_index_);
            
        //TODO: EMERGENCY EXIT, THIS NEED TO BE CHANGED
        if (features_index.size() == 0 || right_features.size() == 0)
            node->right_ = std::make_shared<DecisionTree::Node>(0, 0, labels[0], true);
        else
        {
           node->right_ = build_node(right_features, right_labels, features_index, err_function);
        }
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
                           const err_func& err_function)
{
    this->root_ = build_node(features, labels, features_index, err_function);
}
