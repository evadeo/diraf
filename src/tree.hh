#pragma once

#include <functional>
#include <memory>
#include <vector>

class DecisionTree
{
    struct Node
    {
        Node(int feature_index_splitted, int threshold, int label, bool is_leaf);

        std::unique_ptr<Node> left_;
        std::unique_ptr<Node> right_;
        int feature_index_splitted_;
        int threshold_;
        int label_;
        bool is_leaf_;
    };

    public:
        DecisionTree(const std::vector<std::vector<int>>& features,
                     const std::vector<int>& labels,
                     const std::function<float(std::vector<int>, std::vector<int>)>& err_function,
                     int n_estimators);
        void build_decision_tree(std::vector<std::vector<int>> features,
                                 std::vector<int> labels);

    private:
         std::unique_ptr<DecisionTree::Node> build_node(
                         const std::vector<std::vector<int>>& features,
                         const std::vector<int>& labels,
                         const std::function<float(std::vector<int>, std::vector<int>)>& err_function,
                         int n_estimators);
        
        std::unique_ptr<Node> root_;
};
