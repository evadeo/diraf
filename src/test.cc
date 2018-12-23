#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "distributed_rf.hh"

typedef std::vector<int> vec_t;
typedef std::vector<vec_t> mat_t;

mat_t deserialize_mat(std::string name)
{
    std::ifstream inxtrain(name);
    std::string line;
    mat_t my_mat;
    while (std::getline(inxtrain, line))
    {
        if (line.length() < 1)
            continue;
        vec_t v;
        std::istringstream iss(line);
        int nb;
        while (iss)
        {
            iss >> nb;
            v.push_back(nb);
        }
        my_mat.push_back(v);
    }

    inxtrain.close();
    return my_mat;
}

vec_t deserialize_vec(std::string name)
{
    std::ifstream inxtrain(name);
    std::string line;
    std::getline(inxtrain, line);
    vec_t v;
    std::istringstream iss(line);
    int nb;
    while (iss)
    {
        iss >> nb;
        v.push_back(nb);
    }
    inxtrain.close();
    return v;
}

int main(void)
{
    mat_t x_train = deserialize_mat("../test/x_train");
    mat_t x_test = deserialize_mat("../test/x_test");
    vec_t y_train = deserialize_vec("../test/y_train");
    vec_t y_test = deserialize_vec("../test/y_test");

    auto drf = DistributedRF(3, "gini", 10, -1, true);
    drf.distributed_fit(x_train, y_train);
    drf.distributed_predict(x_test);

    auto preds = drf.get_predictions();

    if (preds.size() != 0)
    {
        std::cout << "PRINTING PREDS" << std::endl;
        for (auto pred : preds)
            std::cout << pred << " | ";
        std::cout << std::endl;

        std::cout << "EXPECTED PREDS" << std::endl;
        for (auto pred : y_test)
            std::cout << pred << " | ";
        std::cout << std::endl;
    }

    return 0;
}
