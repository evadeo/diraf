#include "distributed_rf.hh"

DistributedRF::DistributedRF(int n_estimators, const std::string& criterion, int max_depth,
                             int max_features, bool distributed)
    : criterion_(criterion)
    , max_depth_(max_depth)
    , max_features_(max_features)
{
   if (!distributed)
   {
      n_estimators_ = n_estimators;
      criterion_ = std::string(criterion);
      trees_ = std::vector<DecisionTree>(n_estimators_);

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

      MPI_Bcast(&n_estimators_, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&max_depth_, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&max_features_, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&criterion_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

      criterion_str = (char*)calloc(1, criterion_size + 1);
      if (rank_ == 0)
         std::copy(criterion.data(), criterion.data() + criterion_size, criterion_str);
      MPI_Bcast(criterion_str, criterion_size, MPI_CHAR, 0, MPI_COMM_WORLD);
      criterion_ = std::string(criterion_str);
      free(criterion_str);


      trees_ = std::vector<DecisionTree>(n_estimators_);
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank_ != 0)
         looper();
   }
}

void DistributedRF::looper()
{
   bool cont = true;
   while (cont)
   {
      enum CallMeMaybe cmm;
      MPI_Recv(&cmm, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
	    break;
      }
   }
}


DistributedRF::~DistributedRF()
{
    MPI_Finalize();
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


void DistributedRF::distributed_fit(const std::vector<std::vector<int>>& features_root,
				    const std::vector<int>& labels_root)
{
   int nbFeats, nbValue, nbLabels;
   std::vector<int> labels;
   std::vector<std::vector<int>> features;
   // Broadcast all the data
   if (rank_ == 0)
   {
      nbFeats = features.size();
      nbValue = features[0].size();
      nbLabels = labels_root.size();
      labels = std::vector<int>(labels_root);
      features = std::vector<std::vector<int>>(features_root);
      enum CallMeMaybe cmm = CallMeMaybe::FIT;
      for (int i = 1; i < size_; ++i)
         MPI_Send(&cmm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
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


   // Creating Random Forest
   for (int i = 0; i < n_estimators_ / size_; ++i)
   {
      // On récupère les features random pour cet arbre de décision
      auto random_features = get_random_features(features, max_features_);
      auto err_function = get_error_function(criterion_);
      DecisionTree d_tree(random_features, labels, err_function);
      trees_.push_back(d_tree); // std_forward or emplace back
   }

   // Block all process until the next command
   if (rank_ != 0)
      MPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        trees_.emplace_back(random_features, labels, err_function);
    }
}

void DistributedRF::predict()
{

}

void DistributedRF::distributed_predict(const std::vector<std::vector<int>>& features)
{
   (void)features;
}
