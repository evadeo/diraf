#pragma once

/**
 * \class DistributedRF
 *
 * \brief Distributed Random Forest Class
 *
 * This class is the distritribued Random Forest Class. 
 * It will the class the user will instanciate once and use for the rest
 * of the program.
 * It contains the methods in order to train the forest and to predict classes.
 *
 */

#include <iostream>
#include <map>
#include <string>
#include "mpi.h"
#include "tree.hh"
#include "utils.hh"

class DistributedRF
{
    public:
        /**
         * \brief Constructor for the DistributedRF class
         * \param n_estimators integer representing the number of decision trees in one forest.
         * \param criterion string representing the error mesurement to use when building a decision tree.
         * \param max_depth integer representing the maximum depth of a decision tree.
         * \param max_features integer representing the maximum number of feature to select when building a decision tree.
         * \param distributed boolean to indicate if we are running this class in a distributed environment.
         *
         * This constructor will instanciate all the instances, thus calling `MPI_Init`.
         * 
         */
        DistributedRF(int n_estimators = 10, const std::string& criterion = "mse", int max_depth = 10, int max_features = -1, bool distributed = false);

        /**
         * \brief Destructor for the DistributedRF class
         *
         * The destructor will terminate all the instances and call `MPI_Finalize` on process rank 0.
         */
        ~DistributedRF();
        
        /**
         * \brief Fit method in order to train the forest
         * \param features matrix representing the elements and their associated features. 
         *                 The elements are stored in line and the features are the columns.
         * \param labels vector representing the associated class to an element
         *
         * This method will build all the decision trees that best fit the given features and labels.
         * 
         */
        void fit(const std::vector<std::vector<int>>& features, const std::vector<int>& labels);

        /**
         * \brief Distributed fit method
         * \param features matrix representing the elements and their associated features. 
         *                 The elements are stored in line and the features are the columns.
         * \param labels vector representing the associated class to an element
         *
         * This method will build all the decision trees that best fit the given features and labels
         * in a distributed environment. It relies on the 
         * 
         */
        void distributed_fit(const std::vector<std::vector<int>>& features,
                const std::vector<int>& labels);

        /**
         * \brief Predict method in order to get the predicitions
         * \param features matrix representing the elements and their associated features. 
         *                 The elements are stored in line and the features are the columns.
         *
         * \return Returns the vector containing all the associated labels to each element.
         *
         */
        std::vector<int> predict(const std::vector<std::vector<int>>& features);

        /**
         * \brief Distributed predict method in order to build all the predicitions
         * \param features matrix representing the elements and their associated features. 
         *                 The elements are stored in line and the features are the columns.
         *
         * Builds all the predictions for the given elements.
         *
         */
        void distributed_predict(const std::vector<std::vector<int>>& features);

        /**
         * \brief Looper function
         *
         * Each instance of the distributed environment will call this function.
         * They will loop until receiving a message from master containing the next function to call.
         * It will be repeated until receiving `EXIT`, when the looper will stop, meaning the master program will be finished.
         */
        void looper();

        /**
         * \brief Method to get the predictions once the `distributed_predict` method was called
         *
         * \return Returns the predictions made by the `distributed_predict` method.
         *
         */
        std::vector<int> get_predictions() const;

        /**
         * \brief Enum to send the next instruction to an instance 
         * 
         */
        enum CallMeMaybe {
            FIT,            /// Call the distributed_fit method when sending a message containing `FIT`
            PREDICT,        /// Call the distributed_predict method when sending a message containing `PREDICT`
            EXIT            /// Exit the looper method when sending a message containing `EXIT`
        };
    private:
        int predict_label(const std::vector<int>& elem);
        ///Number of decision trees in the forest.
        int n_estimators_;
        ///Error computation method when building the decision tree.
        std::string criterion_;
        ///Maximum depth of a decision tree
        int max_depth_;
        ///Maximum number of features to select when building a tree
        int max_features_;
        ///Boolean indicating if we are in a distributed environment
        bool distributed_;
        ///Vector containing all the decision trees
        std::vector<DecisionTree> trees_;
        ///Vector containing the predicitions
        std::vector<int> predictions_;

        ///Integer corresponding to the current instance
        int rank_;
        ///Number of instances when in the distributed environment
        int size_;
};
