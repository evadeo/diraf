#pragma once

#include <iostream>
#include <string>
#include "mpi.h"
#include "tree.hh"
#include "utils.hh"

class DistributedRF
{
public:
   DistributedRF(int n_estimators = 10, const std::string& criterion = "mse", int max_depth = 10, int max_features = -1, bool distributed = false);

   //DistributedRF(const DistributedRF& drf) = default;

   ~DistributedRF();
   void fit(const std::vector<std::vector<int>>& features, const std::vector<int>& labels);
   void distributed_fit(const std::vector<std::vector<int>>& features,
			const std::vector<int>& labels);
   void predict();
   void looper();
   void distributed_predict(const std::vector<std::vector<int>>& features);

                                                     enum CallMeMaybe {
							FIT,
	 					PREDICT,
		       EXIT
   };


private:
   int n_estimators_;
   std::string criterion_;
   int max_depth_;
   int max_features_;
   std::vector<DecisionTree> trees_;

   int rank_;
   int size_;
};
