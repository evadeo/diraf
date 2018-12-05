#include "distributed_rf.hh"

DistributedRF::DistributedRF(int n_estimators, const std::string& criterion, int max_depth,
                             int max_features)
    : n_estimators_(n_estimators)
    , criterion_(criterion)
    , max_depth_(max_depth)
    , max_features_(max_features)
    , trees_(n_estimators)
{
/*
    MPI_Init(nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
*/
}

DistributedRF::~DistributedRF()
{
/*
    MPI_Finalize();
*/
}

static std::vector<std::vector<int>> get_random_features(
                                const std::vector<std::vector<int>>& features,
                                int max_features)
{
    int m_features = max_features == - 1 ? features.size(): max_features;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, features.size() - 1);

    std::set<int> features_index;
    for (int i = 0; i < m_features; ++i)
        if (!features_index.insert(dis(gen)).second)
            --i;

    std::cout << "Selected features: ";
    std::vector<std::vector<int>> random_features;
    for (auto f_index : features_index)
    {
        //debug
        std::cout << f_index << " | ";
        random_features.push_back(features[f_index]);
    }
    std::cout << std::endl;
    return random_features;
}


void DistributedRF::fit(const std::vector<std::vector<int>>& features,
                        const std::vector<int>& labels)
{
    
    //TODO: ceci est à distibuer.

    for (int i = 0; i < n_estimators_; ++i)
    {
        // On récupère les features random pour cet arbre de décision
        auto random_features = get_random_features(features, max_features_);
        auto err_function = get_error_function(criterion_);
        DecisionTree d_tree(random_features, labels, err_function);
        trees_.push_back(d_tree);
    }
}

void DistributedRF::predict()
{

}
