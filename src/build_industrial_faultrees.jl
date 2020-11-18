function label(x,dimension,q)
    """Function that labelizes the data
       Returns [Vector of labels (Array{Float64})]

       x: vector of observations (Array{Float64})
       dimension: dimension of features (Int64)
       q: q-th quantile (Float64)
    """
    output = zeros(size(x,1))
    vals = zeros(dimension)
    for d in 1:dimension
        vals[d] = quantile(x[:,d],q)
    end
    for idx in 1:size(x,1)
        if sum([x[idx,d] .>= vals[d] for d in 1:dimension]) == dimension
           output[idx] =  1
        end
    end
    return output
end


function train_industry_dtree(q,nb_samples,sample_size,param)
    """Function that trains decision tree 
       Returns [sklearn.tree.DecisionTreeClassifier]

       q: q-th quantile (Float64)
       nb_samples: number of samples (Int64)
       sample_size: size of sample (Int64)
       param: simulation parameters (Object simulation_parameters)
    """
    y,x, Lengths = get_samples(rand(param.dimension),nb_samples,sample_size,param)
    
    #concatenate all samples of the list 
    seq_x = x[1]
    seq_output = label(x[1],param.dimension,q)
    for i in 2:length(x)
        seq_x = cat(seq_x,x[i],dims=1)
        output = label(x[i],param.dimension,q)
        seq_output = cat(seq_output,output,dims=1)
    end
    x = seq_x
    output = seq_output
    dtree = tree.DecisionTreeClassifier(max_depth =5,max_leaf_nodes=10)
    dtree.fit(x, output)

    return dtree
end


function test_industry_dtree(q,dtree,nb_samples,sample_size,param)
    """Function that tests decision tree 
       Returns [Score of the decision tree (Float64)]

       q: q-th quantile (Float64)
       dtree: decision tree (sklearn.tree.DecisionTreeClassifier)
       nb_samples: number of samples (Int64)
       sample_size: size of sample (Int64)
       param: simulation parameters (Object simulation_parameters)
    """
    y,x,Lengths = get_samples(rand(param.dimension),nb_samples,sample_size,param)
    
    #concatenate all samples of the list 
    seq_y = y[1]
    seq_x = x[1]
    
    seq_label_y = label(y[1],param.dimension,q)
    for i in 2:length(x)
        seq_x = cat(seq_x,x[i],dims=1)
        seq_y = cat(seq_y,y[i],dims=1)
	output_y = label(y[i],param.dimension,q)
	seq_label_y = cat(seq_label_y,output_y,dims=1)
    end
    y = seq_y
    x = seq_x
    output_y = seq_label_y
    
    val = dtree.score(x,output_y)
    return val
end

function accuracy_industry_dtree(q,nb_tests,dtree,nb_samples,sample_size,param)
    """Function that tests decision trees on multiple tests 
       Returns [Accuracy of the decision tree (Float64)]

       q: q-th quantile (Float64)
       nb_tests: number of tests (Int64)
       dtree: decision tree (sklearn.tree.DecisionTreeClassifier)
       nb_samples: number of samples (Int64)
       sample_size: size of sample (Int64)
       param: simulation parameters (Object simulation_parameters)
    """
    accuracy = 0
    for test in 1:nb_tests
        accuracy += test_industry_dtree(q,dtree,nb_samples,sample_size,param)
    end
    accuracy = accuracy/nb_tests
    return accuracy
end

function build_industry_dtrees(M,q,nb_training_samples,nb_tests,nb_test_samples,sample_size,params)
    """Function that builds decision tree for each component 
       Returns [Accuracy of the decision tree (Float64)]

       M: number of components (Int64)
       q: q-th quantile (Float64)
       nb_tests: number of tests (Int64)
       dtree: decision tree (sklearn.tree.DecisionTreeClassifier)
       nb_samples: number of samples (Int64)
       sample_size: size of sample (Int64)
       param: simulation parameters on each components (List of M Object simulation_parameters)
    """
    dtrees = []
    for m in 1:M
        dtree = train_industry_dtree(q,nb_training_samples,sample_size,params[m])
        val = accuracy_industry_dtree(q,nb_tests,dtree,nb_test_samples,sample_size,params[m])
        push!(dtrees,dtree)
        println("Score for component ",m," : ",val*100," %")
    end
    return dtrees
end
