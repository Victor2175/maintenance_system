function get_sample_data(init_depth,sample_size,param)
    """Function that draw a trajectory of features (X_t^m)_{t=1,...,sample_size} using parameters params
       Returns [trajectory from the dynamic (Array{Float64}), trajectory of features observed (Array{Float64}), failure time (Int64)]

       init_depth: the initial crack depth, i.e., initial value Y_1 (Float64)
       sample_size: the sample size that we need to draw. We take Large value to be sure that component ends before t=sample_size
       (Int64)
       param: simulation parameters (object simulation_parameters)
    """

    #Degradation level
    states = ones(sample_size)

    #Observations
    y = zeros(sample_size,param.dimension)

    #Measurement
    x = zeros(sample_size,param.dimension)
    #pritnln(MvNormal(zeros(dimension),sigma_xhi))
    
    states[1] = 1
    y[1,:] = init_depth
    x[1,:] = y[1,:] + rand(MvNormal(zeros(param.dimension),param.sigma_xhi),1)
    
    t_stop = 1
    
    for t in 2:sample_size
        #println(t)
        
        #Draw a new state from the conditional transition probability
        values = 1:param.nb_degrad_states
        probabilities = param.transmat[Int.(states[t-1]),:]
        d = Categorical(probabilities)
        states[t] = rand(d)
       
        #Draw the factor w in the differential equation
        w = rand(MvNormal(zeros(param.dimension), param.sigma_w))
        
        #Next measurement from the differential equation
        y[t,:] = y[t-1,:] + param.C*map(exp,w).*(param.beta_b*exp(param.gamma_e[Int.(states[t])])*map(sqrt,y[t-1,:])).^param.n*param.delta_t
        
        
        #Add a white noise for the observations
        x[t,:] = y[t,:] + rand(MvNormal(zeros(param.dimension),param.sigma_xhi),1)
        #println(y[t,:])
     
        #println([a >= 100 for a in x[t,:]])
        #We stop the simulation when 
        if sum([a >= 100 for a in y[t,:]]) >= 1
            t_stop = t
            break
        end
    end
    return y[1:t_stop,:], x[1:t_stop,:], t_stop
end

function get_samples(init_depth,nb_samples,sample_size,param)
    """Function that draw nb_samples trajectories of features (X_t^m)_{t=1,...,sample_size} using parameters params
       Returns [trajectories from the dynamic (List of nb_samples Array{Float64}: length[sample] x 1), trajectory of features observed (List of nb_samples Array{Float64}:length[sample] x 1), 
                failure time (List of nb_samples Int64)]

       init_depth: the initial crack depth, i.e., initial value Y_1 (Float64)
       nb_samples: number of trajectories (Int64)
       sample_size: the sample size that we need to draw. We take Large value to be sure that component ends before t=sample_size
       (Int64)
       param: simulation parameters (object simulation_parameters)
    """

    y = []
    x = []
    lengths = []

    for sample in 1:nb_samples
        depth, data, len = get_sample_data(init_depth,sample_size,param)
        push!(y, depth)
	    push!(x, data)
        push!(lengths, len)
    end
    return y,x,lengths
end


