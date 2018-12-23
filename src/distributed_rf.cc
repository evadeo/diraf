#include <algorithm>
#include <utility>
#include "distributed_rf.hh"

DistributedRF::DistributedRF(int n_estimators, const std::string& criterion, int max_depth,
        int max_features, bool distributed)
    : criterion_(criterion)
    , max_depth_(max_depth)
    , max_features_(max_features)
    , distributed_(distributed)
{
    if (!distributed)
    {
        n_estimators_ = n_estimators;
        criterion_ = std::string(criterion);
        trees_ = std::vector<DecisionTree>();
        predictions_ = std::vector<int>();
    }
    else
    {
        MPI_Init(nullptr, nullptr);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        int criterion_size;
        char* criterion_str;
        if (rank_ == 0)
        {
            n_estimators_ = n_estimators / size_;
            criterion_size = criterion_.size();
        }

        std::cout << "ON BROADCAST N_ESTIMATORS POUR LE RANK: " << rank_ << std::endl;
        MPI_Bcast(&n_estimators_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::cout << "ON BROADCAST MAX_DEPTH POUR LE RANK: " << rank_ << std::endl;
        MPI_Bcast(&max_depth_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::cout << "ON BROADCAST MAX_FEATURES POUR LE RANK: " << rank_ << std::endl;
        MPI_Bcast(&max_features_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::cout << "ON BROADCAST CRITERION SIZE POUR LE RANK: " << rank_ << std::endl;
        MPI_Bcast(&criterion_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        criterion_str = (char*)calloc(1, criterion_size + 1);
        if (rank_ == 0)
            std::copy(criterion.data(), criterion.data() + criterion_size, criterion_str);
        MPI_Bcast(criterion_str, criterion_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        criterion_ = std::string(criterion_str);
        free(criterion_str);

        trees_ = std::vector<DecisionTree>();
        predictions_ = std::vector<int>();
        MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << "ON A TERMINE LA FIN DU CONSTRUCTEUR POUR LE RANK: " << rank_ << std::endl;
        if (rank_ != 0)
        {
            //std::cout << "ON VA ALLER DANS LE LOOPER POUR LA RANK: " << rank_ << std::endl;
            looper();
            std::cout << "ON A FINI LE LOOPER POUR LE RANK: " << rank_ << std::endl;
        }
    }
}

void DistributedRF::looper()
{
    bool cont = true;
    std::cout << "ON EST DANS UN LOOPER POUR LE RANK: " << rank_ << std::endl;
    while (cont)
    {
        enum CallMeMaybe cmm;
        MPI_Recv(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        auto zero = std::vector<int>(0);
        auto zerozero = std::vector<std::vector<int>>(0);
        switch (cmm)
        {
            case FIT:
                distributed_fit(zerozero, zero);
                break;

            case PREDICT:
                distributed_predict(zerozero);
                break;

            case EXIT:
                cont = false;
                std::cout << "ON QUITTE LE LOOPER POUR LE RANK: " << rank_ << std::endl;
		std::exit(0);
                break;
        }
    }
}


DistributedRF::~DistributedRF()
{
    if (distributed_)
    {
        enum CallMeMaybe cmm = CallMeMaybe::EXIT;
        for (int i = 1; i < size_; ++i)
        {
            MPI_Send(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }

        MPI_Finalize();
    }
}


static std::vector<std::vector<int>> get_random_features(
        const std::vector<std::vector<int>>& features,
        std::vector<int>& real_indexes,
        int max_features)
{
    int m_features = max_features == - 1 ? features[0].size(): max_features;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, features[0].size() - 1);

    std::set<int> features_index;
    for (int i = 0; i < m_features; ++i)
        if (!features_index.insert(dis(gen)).second)
            --i;

    std::copy(features_index.begin(), features_index.end(), std::back_inserter(real_indexes));

    std::cout << "Selected features: ";
    for (auto f_index : features_index)
        std::cout << f_index << " | ";
    std::cout << std::endl;
    std::vector<std::vector<int>> random_features(features.size());
    for (size_t i = 0; i < features.size(); ++i)
        for (auto f_index : features_index)
            random_features[i].push_back(features[i][f_index]);
    return random_features;
}


void DistributedRF::distributed_fit(const std::vector<std::vector<int>>& features_root,
        const std::vector<int>& labels_root)
{
    int nbFeats, nbValue, nbLabels;
    std::vector<int> labels;
    std::vector<std::vector<int>> features;
    // Broadcast all the data
    if (rank_ == 0)
    {
        nbFeats = features_root.size();
        //std::cout << "TOTOTOTOT" << std::endl;
        //std::cout << "FEATURES SIZE" << nbFeats << std::endl;
        nbValue = features_root[0].size();
        nbLabels = labels_root.size();
        labels = std::vector<int>(labels_root);
        features = std::vector<std::vector<int>>(features_root);
        enum CallMeMaybe cmm = CallMeMaybe::FIT;
        for (int i = 1; i < size_; ++i)
        {
            std::cout << "ON ENVOIE L'ENUM AU RANK: " << i << std::endl;
            MPI_Send(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&nbFeats, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbLabels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0)
    {
        features = std::vector<std::vector<int>>(nbFeats, std::vector<int>(nbValue));
        labels = std::vector<int>(nbLabels);
    }
    for (int i = 0; i < nbFeats; ++i)
        MPI_Bcast(features[i].data(), nbValue, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(labels.data(), nbLabels, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);


    std::cout << "ON EXECUTE LE FIT POUR LE RANK: " << rank_ << std::endl;

    this->fit(features, labels);

    // Creating Random Forest
    /*for (int i = 0; i < n_estimators_ / size_; ++i)
      {
    // On récupère les features random pour cet arbre de décision
    std::vector<int> real_indexes;
    auto random_features = get_random_features(features, real_indexes, max_features_);
    auto err_function = get_error_function(criterion_);
    trees_.emplace_back(random_features, labels, real_indexes, err_function);
    }*/

    // Block all process until the next command
    //if (rank_ != 0)
        //MPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


void DistributedRF::fit(const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels)
{
    for (int i = 0; i < n_estimators_; ++i)
    {
        // On récupère les features random pour cet arbre de décision
        std::vector<int> real_indexes;
        auto random_features = get_random_features(features, real_indexes, max_features_);
        auto err_function = get_error_function(criterion_);
        std::cout << "Indexes selected for estimator " << i << ": ";
        for (size_t i = 0; i < real_indexes.size(); ++i)
            std::cout << real_indexes[i] << " | ";
        std::cout << std::endl;
        trees_.emplace_back(random_features, labels, real_indexes, err_function);
    }
}

static int choose_prediction(const std::vector<int>& predictions)
{
    std::map<int, int> counts;

    for (int pred : predictions)
        ++counts[pred];

    const auto max = std::max_element(counts.begin(), counts.end(),
            [](const auto& p1, const auto& p2) { return p1.second < p2.second; });

    return max->first;
}

int DistributedRF::predict_label(const std::vector<int>& elem)
{
    std::vector<int> predictions;
    for (const auto& node : trees_)
        predictions.push_back(node.predict(elem));

    int pred = choose_prediction(predictions);

    return pred;
}

std::vector<int> DistributedRF::predict(const std::vector<std::vector<int>>& features)
{
    std::vector<int> predictions;

    std::cout << "STARTING PREDICT FOR RANK: " << rank_ << std::endl;
    for (size_t i = 0; i < features.size(); ++i)
    {
        int pred = this->predict_label(features[i]);
        predictions.push_back(pred);
    }
    return predictions;
}

void DistributedRF::distributed_predict(const std::vector<std::vector<int>>& features_root)
{
    int nbFeats, nbValue;
    std::vector<std::vector<int>> features;
    // Broadcast all the data
    if (rank_ == 0)
    {
        nbFeats = features_root.size();
        nbValue = features_root[0].size();
        features = std::vector<std::vector<int>>(features_root);
        enum CallMeMaybe cmm = CallMeMaybe::PREDICT;
        for (int i = 1; i < size_; ++i)
        {
            std::cout << "ON ENVOIE L'ENUM AU RANK: " << i << std::endl;
            MPI_Send(&cmm, sizeof(enum CallMeMaybe), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&nbFeats, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbValue, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0)
        features = std::vector<std::vector<int>>(nbFeats, std::vector<int>(nbValue));
    for (int i = 0; i < nbFeats; ++i)
        MPI_Bcast(features[i].data(), nbValue, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> predictions = this->predict(features);
    
    if (rank_ == 0)
    {
        std::vector<std::vector<int>> all_preds(size_, std::vector<int>(predictions.size()));
        all_preds[0] = predictions;
        for (int i = 1; i < size_; ++i)
	  MPI_Recv(all_preds[i].data(), predictions.size(), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (size_t i = 0; i < all_preds[0].size(); ++i)
        {
            std::vector<int> pred_elem;
            for (size_t j = 0; j < all_preds.size(); ++j)
                pred_elem.push_back(all_preds[j][i]);
            predictions_.push_back(choose_prediction(pred_elem));
        }
    }
    else
        MPI_Send(predictions.data(), predictions.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
}

std::vector<int> DistributedRF::get_predictions() const
{
    return predictions_;
}
