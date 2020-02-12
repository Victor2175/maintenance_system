function train_dtree(model,nb_samples,sample_size,param)
    """Function that train decision tree using samples
       Returns [sklearn.DecisionTreeClassifier]

       model: HMM parameters of our Gaussian HMM (hmmlearn.hmm.GaussianHMM)
       nb_samples: number of samples (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters (Object simulation_parameters)
    """
    # println(param.dimension)
    y_train,x_train,Lengths = get_samples(rand(param.dimension),nb_samples,sample_size,param)
    #concatenate all samples of the list 
    seq = x_train[1]
    for i in 2:length(x_train)
        seq = cat(seq,x_train[i],dims=1)
    end
    x_train = seq
    
      
    x_predicted = model.predict(x_train, lengths=Int.(Lengths))
    dtree = tree.DecisionTreeClassifier(max_depth=15,max_leaf_nodes=10)
    dtree.fit(x_train, x_predicted)
    return dtree
end


function test_dtree(model,dtree,nb_samples,sample_size,param)
    """Function that tests decision tree using samples
       Returns [Score of the test (Float64)]

       model: HMM parameters of our Gaussian HMM (hmmlearn.hmm.GaussianHMM)
       dtree: decision tree learned (sklearn.DecisionTreeClassifier)
       nb_samples: number of samples (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters (Object simulation_parameters)
    """
    y_test,x_test,Lengths = get_samples(rand(param.dimension),nb_samples,sample_size,param)
    
    #concatenate all samples of the list 
    seq = x_test[1]
    for i in 2:length(x_test)
        seq = cat(seq,x_test[i],dims=1)
    end
    x_test = seq
    #y_test = np.reshape(y_test,(-1,1))
    
    x_predicted = model.predict(x_test,lengths=Int.(Lengths))
    val = dtree.score(x_test,x_predicted)
    return val
end

function accuracy_dtree(nb_tests,model,dtree,nb_samples,sample_size,param)
    """Function that compute the accuracy of a decision tree on multiple tests
       Returns [Score of the test (Float64)]

       nb_tests: number of test we do to estimate accuracy of our model (Int64)
       model: HMM parameters of our Gaussian HMM (hmmlearn.hmm.GaussianHMM)
       dtree: decision tree learned (sklearn.DecisionTreeClassifier)
       nb_samples: number of samples (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters (Object simulation_parameters)
    """
    accuracy = 0
    for test in 1:nb_tests
        accuracy += test_dtree(model,dtree,nb_samples,sample_size,param)
    end
    accuracy = accuracy/nb_tests
    return accuracy
end

function build_dtrees(nb_training_samples,nb_tests,nb_test_samples,models,sample_size,params)
    """Function that compute the accuracy of a decision tree on multiple tests
       Returns [Decision tree on each component (List of M sklearn.DecisionTreeClassifier), Score of the test on each component (List of M Float64)]

       nb_training_samples: number of training samples (Int64)
       nb_tests: number of test we do to estimate accuracy of our models (Int64)
       nb_test_samples: number of training samples (Int64)
       models: HMM parameters of our Gaussian HMM on each component (list of M hmmlearn.hmm.GaussianHMM)
       sample_size: sample size to draw trajectories (Int64) 
       params: simulation parameters on each component (List of M Object simulation_parameters)
    """
    dtrees = []
    scores = []
    for (m,model) in enumerate(models)
        #println("Component ", idx)
        dtree = train_dtree(models[m],nb_training_samples,sample_size,params[m])
        val = accuracy_dtree(nb_tests,models[m],dtree,nb_test_samples,sample_size,params[m])
        push!(dtrees,dtree)
	    push!(scores,val)
        println("Score for component ",m," : ",val*100," %")
    end
    return dtrees,scores
end
