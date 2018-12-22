#pragma once

#include <functional>
#include <memory>
#include <random>
#include <set>
#include <vector>
#include "utils.hh"

class DecisionTree
{
    struct Node
    {
        Node(int feature_index_split, int threshold, int label, bool is_leaf);

        std::shared_ptr<Node> left_;
        std::shared_ptr<Node> right_;
        int feature_index_split_;
        int threshold_;
        int label_;
        bool is_leaf_;
    };

    public:
        DecisionTree() : root_(nullptr) {}
        DecisionTree(std::vector<std::vector<int>> features,
                const std::vector<int>& labels,
                std::vector<int> features_index,
                const err_func& err_function);
        void build_decision_tree(std::vector<std::vector<int>> features,
                                 std::vector<int> labels);

        int predict(const std::vector<int>& elem) const;

    private:
         std::shared_ptr<DecisionTree::Node> build_node(
                std::vector<std::vector<int>> features,
                const std::vector<int>& labels,
                std::vector<int> features_index,
                const err_func& err_function);
        
        std::shared_ptr<Node> root_;
};
