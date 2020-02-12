#Functions to train HMMs

function compute_BIC(nb_states,nb_samples,Lengths,ll,nb_features)
    """Function that computes the bayes information criterion 
       Returns [Bayes Information Criterion (Float64)]

       nb_states: number of states used in this test (Int64)
       nb_samples: number of samples (Int64)
       Lengths: size of each sample (List of nb_samples Int64) 
       ll: log-likelihood (Float64)
       nb_features: number of features d (Int64)
    """
    #Number of parameters of the statistical model
    k = 2*(nb_states-1)+1 + nb_states*(nb_features*nb_features+nb_features) + nb_states
    #Number of parameters of samples* number of steps per sample
    N = sum(Lengths)
    
    #Compute the BIC
    BIC = -2*ll + k*log(N)
    return BIC
end

function train_hmm(nb_states,nb_samples,sample_size,param)
    """Function that trains the model on samples
       Returns [HMM statistical model (hmmlearn.hmm.GaussianHMM)]

       nb_states: number of states used in this test (Int64)
       nb_samples: number of samples (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters (Object simulation_parameters)
    """
    #Draw randomly the initial crack depth
    init_depth = rand(param.dimension)

    #Draw samples
    Depths, Observations,Lengths = get_samples(init_depth,nb_samples,sample_size,param)
    
    #concatenate all samples of the list 
    seq = Observations[1]
    for i in 2:length(Observations)
        seq = cat(seq,Observations[i],dims=1)
    end
    Observations = seq
    #Observations = np.reshape(Observations,(-1,1))
    
    #Define the initial parameters
    p_trans = zeros(nb_states,nb_states)
    for state_1 in 1:nb_states
        for state_2 in state_1:nb_states
            if state_2 <= state_1+1
              p_trans[state_1,state_2] = rand()              
            end
        end
    end
    p_trans = p_trans./sum(p_trans,dims=2)
    
    #Define the initial covariance (needs to be positive definite in hmmlearn)
    covars = rand(nb_states,param.dimension,param.dimension)
    for s in 1:nb_states
        #covars[s,:,:] = rand(nb_states,Int.(dimensions[m]),Int.(dimensions[m]))
        covars[s,:,:] = transpose(covars[s,:,:])*covars[s,:,:]
        eig_covars = maximum(broadcast(abs,eigvals(covars[s,:,:])))
        covars[s,:,:] += (eig_covars+0.1)*Matrix(I,param.dimension,param.dimension)
    end
    
    model = hmm.GaussianHMM(n_components=nb_states, covariance_type="full",init_params="")
    model.transmat_ = p_trans
    model.fit(Observations,lengths=Int.(Lengths))
    
    return model 
end

function test_hmm(model,nb_samples,sample_size,param)
    """Function that computes the BIC of the learned HMM on samples
       Returns [BIC (Float64)]

       model: Gaussian HMM parameters (Object hmmlearn.hmm.GaussianHMM)
       nb_states: number of states used in this test (Int64)
       nb_samples: number of samples (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters (Object simulation_parameters)
    """

    #Draw Test samples
    init_depth = rand(param.dimension)
    Depths,Observations,Lengths = get_samples(init_depth,nb_samples,sample_size,param)
    
    #concatenate all samples of the list 
    seq = Observations[1]
    for i in 2:length(Observations)
        seq = cat(seq,Observations[i],dims=1)
    end
    Observations = seq
    
    nb_states = model.n_components
    ll = model.score(Observations,lengths=Int.(Lengths))
    #Compute the BIC
    BIC = compute_BIC(nb_states,nb_samples,Lengths,ll,param.dimension)
    
    return BIC 
end

function select_best_model(nb_states_max,nb_training_samples,nb_test_samples,sample_size,param)
    """Function that choose the best nummber of states in our HMM using BIC
       Returns [optimal number of states (Int64), Best HMM statistical model (hmmlearn.hmm.GaussianHMM)]

       nb_states_max: maximum number of states we allow in our Gaussian HMM (Int64)
       nb_training_samples: number of samples to train our model (Int64)
       nb_test_samples: number of samples to test our model (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters (Object simulation_parameters)
    """
    BIC_scores = zeros(nb_states_max)
    model_opt = 0
    nb_states_opt = 2
    min = 10e10
    rd_state = 1
    for nb_states in 4:nb_states_max
        
        println("Train and Tests with ",nb_states," states")
        model = 0
        i=1
        while (BIC_scores[nb_states] == 0) && (i < 10)
            println(i)
            i += 1
            try
                model = train_hmm(nb_states,nb_training_samples,sample_size,param)
                BIC_scores[nb_states] = test_hmm(model,nb_test_samples,sample_size,param)
            catch
                BIC_scores[nb_states] = 0
            end
        end
        if BIC_scores[nb_states] == 0
           BIC_scores[nb_states] = 10e10 
        end
        
        if BIC_scores[nb_states] < min
            model_opt = model
            min = BIC_scores[nb_states]
            nb_states_opt = nb_states
        end
    end
    BIC_scores[1] = BIC_scores[2]+1
    
    println("Optimal number of states : ", nb_states_opt)  
    return nb_states_opt,model_opt
end


function build_models(M,nb_states_max,nb_training_samples,nb_test_samples,sample_size,params)
    """Function that finds the best statistical model on each component
       Returns [optimal number of states (List of M Int64), Best HMM statistical model (Lis of M hmmlearn.hmm.GaussianHMM)]

       nb_states_max: maximum number of states we allow in our Gaussian HMM (Int64)
       nb_training_samples: number of samples to train our model (Int64)
       nb_test_samples: number of samples to test our model (Int64)
       sample_size: sample size to draw trajectories (Int64) 
       param: simulation parameters on each component  (List of M Object simulation_parameters)
    """

    models = []
    nb_states = []
    
    for m in 1:M
        println("Start learning component ", m)
        nb_states_opt,model_opt = select_best_model(nb_states_max,nb_training_samples,nb_test_samples,sample_size,params[m])
        append!(nb_states,nb_states_opt)
        push!(models,model_opt)
    end
    return nb_states, models
end
