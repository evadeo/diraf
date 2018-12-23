#pragma once

/**
 * \class DecisionTree
 *
 * \brief Decision Tree class
 *
 * This class represents one Decision Tree. 
 *
 */

#include <functional>
#include <memory>
#include <random>
#include <set>
#include <vector>
#include "utils.hh"

class DecisionTree
{

    /*
     * \struct Node
     *
     * \brief Inner Node class
     *
     * This class represents one node of the Decision Tree.
     * It will contain information on the split to perform and the possible 
     * output class if the node is a leaf.
     *
     */
    struct Node
    {
        /**
         * \brief Constructor for the Node struct
         * \param feature_index_split Integer that represents the index of the feature to perform the split on
         * \param threshold Integer that will be the value to split on (if < then it will be on the left side, right side otherwise)
         * \param label Integer that will indicated the output class if is a leaf
         * \param is_leaf Boolean to tell if this node is a leaf or not
         * 
         */
        Node(int feature_index_split, int threshold, int label, bool is_leaf);

        ///Left children
        std::shared_ptr<Node> left_;
        ///Right children
        std::shared_ptr<Node> right_;
        ///Integer representing the index of the feature to perform the split
        int feature_index_split_;
        ///Integer that will be the value to split on
        int threshold_;
        ///Integer representing the output class
        int label_;
        ///Boolean to tell if the node is a leaf or not
        bool is_leaf_;
    };

    public:
        /*
         * \brief Empty constructor
         */
        DecisionTree() : root_(nullptr) {}

        /*
         * \brief Constructor for the Decision Tree class
         *
         * \param features Matrix representing the elements and their associated features. 
         *                 The elements are stored in line and the features are the columns.
         * \param labels Vector representing the associated class to an element
         * \param features_index Vector representing the real indexes of the features
         *                       Because we only selected a few features previously,
         *                       it may not be their original position.
         * \param err_function Error function to minimize when doing a split
         */
        DecisionTree(std::vector<std::vector<int>> features,
                const std::vector<int>& labels,
                std::vector<int> features_index,
                const err_func& err_function);

        /*
         * \brief Function to predict the output class of an element.
         *
         * \param elem Vector containing all the features of one element.
         */
        int predict(const std::vector<int>& elem) const;

    private:
        /*
         * \brief Function to recursively build the Decision Tree
         *
         * \param features Matrix representing the elements and their associated features. 
         *                 The elements are stored in line and the features are the columns.
         * \param labels Vector representing the associated class to an element
         * \param features_index Vector representing the real indexes of the features
         *                       Because we only selected a few features previously,
         *                       it may not be their original position.
         * \param err_function Error function to minimize when doing a split
         *
         * \return Returns the head of the Decision Tree
         *
         */
        std::shared_ptr<DecisionTree::Node> build_node(
                std::vector<std::vector<int>> features,
                const std::vector<int>& labels,
                std::vector<int> features_index,
                const err_func& err_function);
        
        ///Root of the Decision Tree
        std::shared_ptr<Node> root_;
};
