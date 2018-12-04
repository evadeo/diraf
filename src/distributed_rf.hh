#pragma once

#include <iostream>
#include <string>
#include "mpi.h"

class DistributedRF
{
    public:
        DistributedRF(int n_estimators = 10, const std::string& criterion = "mse", int max_depth = 10);
        ~DistributedRF();
        int fit();
        void predict();

    private:
        int n_estimators_;
        std::string criterion_;
        int max_depth_;

        int rank_;
        int size_;
};
