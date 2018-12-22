#include "distributed_rf.hh"

#include <vector>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    std::vector<int> features_a { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::vector<int> features_b { 1, 1, 1, 3, 4, 5, 8, 9, 4, 5, 2, 10, 20, 12, 7 };
    std::vector<int> features_c { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int> features_d { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int> features_e { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    std::vector<std::vector<int>> features;
    features.push_back(features_a);
    features.push_back(features_b);
    features.push_back(features_c);
    features.push_back(features_d);
    features.push_back(features_e);

    std::vector<int> labels { 0, 1, 2, 3, 4 };

    auto drf = DistributedRF(10, "gini", 10, 3);
    drf.fit(features, labels);

    std::cout << "FINISHED FIT" << std::endl;

    std::vector<int> t { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<int> u { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::vector<std::vector<int>> test;
    test.push_back(t);
    test.push_back(u);

    auto preds = drf.predict(test);

    for (auto pred : preds)
        std::cout << pred << " | ";
    std::cout << std::endl;
}
