function indicator_function(dtree,label,x)
    """Function that returns 1 if the decision tree predicts label
       Returns [binary indicator (Int64)]
        
       dtree: decision tree (sklearn.tree.DecisionTreeClassifier)
       label: discrete value (Int64)
       param: simulation parameters (Object simulation_parameters)
    """
    output = 0
    x = np.reshape(x,(1,-1))
    val = dtree.predict(x)
    
    if val[1] == label
        output = 1
    else
        output = 0
    end
    return output
end

#Compute conditional probabilities p(d|s)
function monte_carlo_simulation(dtree,model,state,label,n_samples=1000)
    """Function that compute discrete emission probabilities using monte-carlo estimator
       Returns [probability (Float64)]
        
       dtree: decision tree (sklearn.tree.DecisionTreeClassifier)
       model: HMM parameter (hmmlear.hmm.GaussianHMM)
       state: discrete state of the HMM (Int64)
       label: discrete label (Int64)
       n_samples: number of samples to draw (Int64)
    """
    mean = model.means_[state,:]
    covariance = model.covars_[state,:,:] 
    samples = scipy.stats.multivariate_normal.rvs(mean, covariance, n_samples)
    values = np.array([indicator_function(dtree,label,samples[i,:]) for i in 1:n_samples])
    return np.mean(values)
end
########################################################################################################

#Compute the table of conditional probabilities p(d|s)
function compute_cond_probas(dtree,model,nb_labels)
    """Function that computes discrete emission probabilities using monte-carlo estimator
       Returns [emission probability distribution (Array{Float64}: nb_states x nb_labels)]
        
       dtree: decision tree (sklearn.tree.DecisionTreeClassifier)
       model: HMM parameter (hmmlear.hmm.GaussianHMM)
       nb_labels: number of possible outputs of the decision tree (Int64)
    """
    nb_states = model.n_components
    table = zeros(nb_labels,nb_states)
    for state in 1:nb_states
        for label in 1:nb_labels
            table[label,state] = monte_carlo_simulation(dtree,model,state,label)
        end
    end

    #these lines ensure that table represents probability distribution
    for state in 1:nb_states
        table[state,state] += 0.1
        table[:,state] = table[:,state]./sum(table[:,state])
    end
    table = np.transpose(table)
    return table
end
#########################################################################################################


#########################################################################################################
#Build all emissions matrix
function build_emissions(dtrees,models)
    """Function that computes emission probabilities for each component
       Returns [list of M emission probabilities (Array{Float64}: nb_states x nb_labels)]
        
       dtrees: list of decision tree (List of M sklearn.tree.DecisionTreeClassifier)
       models: list of HMM parameter (List of M hmmlear.hmm.GaussianHMM)
    """
    emissions = []
    for (m,dtree) in enumerate(dtrees)
        println("Compute probabilities of component ",m)
        table = compute_cond_probas(dtree,models[m],models[m].n_components)
        push!(emissions,table)
    end
    return emissions
end