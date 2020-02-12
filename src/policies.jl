include("model_decPOMDP.jl")


######################Function that returns action############################
function def_policy_MILP(T,M,K,pomdp,initial_obs,cuts)
    """ Function that runs MILP-based heuristic policy
        Returns [List of component to maintain (List of {Int64}, can be empty), Optimal Value (Float64), Computation time (Float64)]
        
        T: horizon (Int64)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        pomdp: define pomdp_parameters (object pomdp_parameters)
        initial_obs: initial observation on each component o^m (vector{Int64}: nbO[m])
        cuts: binary indicator (Int64)
    """
    #Set the action set
    #If a=2, then the equipment is maintained
    #Otherwise, we do nothing
    A = 1:2


    #Define the mathematical program
    model_milp = []


    #If cuts = 1, then use MILP with valid inequalities
    #Otherwise, then use MILP
    if cuts == 1
        model_milp = model_decPOMDP_cuts(T,M,K,pomdp.nbS,pomdp.nbO,A,pomdp.p_init,pomdp.p_trans,pomdp.p_emis,pomdp.p_cuts,pomdp.reward,initial_obs)
    else
        model_milp = model_decPOMDP(T,M,K,pomdp.nbS,pomdp.nbO,A,pomdp.p_init,pomdp.p_trans,pomdp.p_emis,pomdp.reward,initial_obs)
    end

    #Solve Problem
    optimize!(model_milp)

    action = []
    obj = 0.0
    time_policy = 0.0
    if termination_status(model_milp) == MOI.OPTIMAL
        τ_a = value.(model_milp[:τ_a])   
    
        #Get variable values τ_a
        tau = zeros(T,M+1,2)
        for m in 1:M+1    
            tau[1,m,1] = τ_a[1,m,1]
            tau[1,m,2] = τ_a[1,m,2]
            # println(τ_a[1,m,1])
        end
        #Get indices of τ_{1}^{t,m} sorted in the decreasing order
        indices = sortperm(tau[1,1:M,2],rev=true) 
        #Define the set of components to maintain
        for (idx,a) in enumerate(indices)
           if (tau[1,a,2] > 0) && (idx <= K) && (tau[1,a,2] >= tau[1,M+1,2])
              push!(action,a)
           end
        end

        #Get the objective value
        obj = objective_value(model_milp)

        #Get the computation time
        time_policy = solve_time(model_milp)

    elseif termination_status(model_milp) == MOI.TIME_LIMIT && has_values(model_milp)
        τ_a = value.(model_milp[:τ_a])
        obj = objective_value(model_milp)

        #Get variable values τ_a
        tau = zeros(T,M+1,2)
        for m in 1:M+1    
            tau[1,m,1] = τ_a[1,m,1]
            tau[1,m,2] = τ_a[1,m,2]
        end
        #Get indices of τ_{1}^{t,m} sorted in the decreasing order
        indices = sortperm(tau[1,1:M,2],rev=true) 
         
        #Define the set of components to maintain
        for (idx,a) in enumerate(indices)
           if (tau[1,a,2] > 0) && (idx <= K) && (tau[1,a,2] >= tau[1,M+1,2])
              push!(action,a)
           end
        end

        #Get the objective value
        obj = objective_value(model_milp)

        #Get the computation time
        time_policy = solve_time(model_milp)
    else
        println("The model was not solved correctly.")
    end

    return action, obj, time_policy
end

function def_policy_NLP(T,M,K,pomdp,initial_obs)
    """ Function that runs NLP-based heuristic policy.
        Returns [List of component to maintain (List of {Int64}, can be empty), Optimal Value (Float64), Computation time (Float64)]
        
        T: horizon (Int64)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        pomdp: define pomdp_parameters (object pomdp_parameters)
        initial_obs: initial observation on each component o^m (vector{Int64}: nbO[m])
    """
    #Set the action set
    #If a=2, then the equipment is maintained
    #Otherwise, we do nothing
    A = 1:2

    #Define the mathematical program
    model_nlp = model_decPOMDP_NLP(T,M,K,pomdp.nbS,pomdp.nbO,A,pomdp.p_init,pomdp.p_trans,pomdp.p_emis,pomdp.reward,initial_obs)

    #Solve mathematical program
    optimize!(model_nlp)

    action = []
    obj = 0.0
    time_policy = 0.0
    if termination_status(model_nlp) == MOI.OPTIMAL
        τ_a = value.(model_nlp[:τ_a])   
    
        #Get variable values τ_a
        tau = zeros(T,M+1,2)
        for m in 1:M+1    
            tau[1,m,1] = τ_a[1,m,1]
            tau[1,m,2] = τ_a[1,m,2]
        end

        #Get indices of τ_{1}^{t,m} sorted in the decreasing order
        indices = sortperm(tau[1,1:M,2],rev=true)

        #Define the set of components to maintain
        for (idx,a) in enumerate(indices)
           if (tau[1,a,2] > 0) && (idx <= K) && (tau[1,a,2] >= tau[1,M+1,2])
              push!(action,a)
           end
        end

        #Get the objective value
        obj = objective_value(model_nlp)

        #Get the computation time
        time_policy = solve_time(model_nlp)

    elseif termination_status(model_nlp) == MOI.TIME_LIMIT && has_values(model_nlp)
        τ_a = value.(model_nlp[:τ_a])
        obj = objective_value(model_nlp)

        #Get variable values τ_a
        tau = zeros(T,M+1,2)
        for m in 1:M+1    
            tau[1,m,1] = τ_a[1,m,1]
            tau[1,m,2] = τ_a[1,m,2]
            # println(τ_a[1,m,1])
        end
        #Get indices of τ_{1}^{t,m} sorted in the decreasing order
        indices = sortperm(tau[1,1:M,2],rev=true) 
         #Define the set of components to maintain
        
        for (idx,a) in enumerate(indices)
           if (tau[1,a,2] > 0) && (idx <= K) && (tau[1,a,2] >= tau[1,M+1,2])
              push!(action,a)
           end
        end

        #Get the objective value
        obj = objective_value(model_nlp)

        #Get the computation time
        time_policy = solve_time(model_nlp)
    else
        println("The model was not solved correctly.")
    end
    return action, obj, time_policy
end

function def_policy_LP(T,M,K,pomdp,initial_obs,cuts)
    """ Function that runs LP-based heuristic policy (that corresponds to the relaxation of our MILP).
        Returns [List of component to maintain (List of {Int64}, can be empty), Optimal Value (Float64), Computation time (Float64)]
        
        T: horizon (Int64)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        pomdp: define pomdp_parameters (object pomdp_parameters)
        initial_obs: initial observation on each component o^m (vector{Int64}: nbO[m])
    """
    #Set the action set
    #If a=2, then the equipment is maintained
    #Otherwise, we do nothing
    A = 1:2


    #Define the mathematical program
    model_lp = []

    #If cuts = 1, then use MILP with valid inequalities
    #Otherwise, then use MILP
    if cuts == 1
        model_lp = model_decPOMDP_cuts(T,M,K,pomdp.nbS,pomdp.nbO,A,pomdp.p_init,pomdp.p_trans,pomdp.p_emis,pomdp.p_cuts,pomdp.reward,initial_obs)
    else
        model_lp = model_decPOMDP(T,M,K,pomdp.nbS,pomdp.nbO,A,pomdp.p_init,pomdp.p_trans,pomdp.p_emis,pomdp.reward,initial_obs)
    end

    #Solve Problem
    model_lp = relax_model(model_lp)
    optimize!(model_lp)
    
    action = []
    obj = 0.0
    time_policy = 0.0
    if termination_status(model_lp) == MOI.OPTIMAL
        τ_a = value.(model_lp[:τ_a])   
    
        #Get variable values τ_a
        tau = zeros(T,M+1,2)
        for m in 1:M+1    
            tau[1,m,1] = τ_a[1,m,1]
            tau[1,m,2] = τ_a[1,m,2]
        end

        #Get indices of τ_{1}^{t,m} sorted in the decreasing order
        indices = sortperm(tau[1,1:M,2],rev=true)

        #Define the set of components to maintain
        for (idx,a) in enumerate(indices)
           if (tau[1,a,2] > 0) && (idx <= K) && (tau[1,a,2] >= tau[1,M+1,2])
              push!(action,a)
           end
        end

        #Get the objective value
        obj = objective_value(model_lp)

        #Get the computation time
        time_policy = solve_time(model_lp)

    elseif termination_status(model_lp) == MOI.TIME_LIMIT && has_values(model_lp)
        τ_a = value.(model_lp[:τ_a])
        obj = objective_value(model_lp)

        #Get variable values τ_a
        tau = zeros(T,M+1,2)
        for m in 1:M+1    
            tau[1,m,1] = τ_a[1,m,1]
            tau[1,m,2] = τ_a[1,m,2]
            # println(τ_a[1,m,1])
        end
        #Get indices of τ_{1}^{t,m} sorted in the decreasing order
        indices = sortperm(tau[1,1:M,2],rev=true) 
         #Define the set of components to maintain
        action = []
        for (idx,a) in enumerate(indices)
           if (tau[1,a,2] > 0) && (idx <= K) && (tau[1,a,2] >= tau[1,M+1,2])
              push!(action,a)
           end
        end

        #Get the objective value
        obj = objective_value(model_lp)

        #Get the computation time
        time_policy = solve_time(model_lp)
    else
        println("The model was not solved correctly.")
    end
    return action, obj, time_policy
end

function def_myopic_policy(models,M,K,pomdp)
    """ [Not perfectly defined]
        Function that runs myopic policy.
        Returns [List of component to maintain (List of {Int64}, can be empty)]
        
        models: list of statistical model on each component (list of M hmmlearn.model)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        pomdp: define POMDP parameters (Object pomdp_parameters) 
        reward: reward on each component r^m(s,a,s') (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
    """
    #Set the action set
    A=1:2
    nbA =length(A)

    #Define next reward on each component
    next_reward= []
    for m in 1:M
        push!(next_reward, sum(pomdp.p_init[m][ss]*(pomdp.p_trans[m][ss,:,:].*pomdp.reward[m][ss,:,:]) for ss in 1:pomdp.nbS[m]))
    end

    #Define objective value for each component and each action
    #Sum over the next state values
    objs = zeros(M,nbA)
    for m in 1:M, a in 1:nbA
        objs[m,a] = sum(next_reward[m][a,:])
    end

    #Sort the objective values in the decreasing order
    indices = sortperm(objs[1:M,2],rev=true)

    #Define the list of components to maintain
    action = []
    for (idx,a) in enumerate(indices)
        if (idx <= K) && (objs[a,2] >= sum(objs[m,1] for m in 1:M))
                push!(action,a)
        end
    end
    return action
end


function def_industry_policy(M,K,dtrees,features,failures,costs_f,costs_r)
    """ Function that runs the heuristic in the current practice.
        Returns [List of component to maintain (List of {Int64}, can be empty)]
    
        M: number of components (Int64)
        K: maintenance capacity (Int64) 
        dtrees: decision trees (or fault trees) (List of M sklearn.tree.DecisionTreeClassifier)
        features: vector of current continuous observations X_t^m on each component (List of M Array{Float64}: dimension[m] x 1)
        failures: vector of binary indicators indicating the presence of failures (Array{Float64}: M x 1)
        costs_f: vector of failure costs (Array{Float64}: M x 1)
        costs_r: vector of maintenance costs (Array{Float64}: M x 1)
    """
    #Define predictions of the fault trees given current observation (features)
    predictions = zeros(M)
    
    #We assume that if a failure happens, the fault tree detects it
    #Otherwise, it defines the standar prediction
    for m in 1:M
        if failures[m] ==1
            predictions[m] = 1
        else
            val = np.reshape(features[m],(1,-1))
            predictions[m] = dtrees[m].predict(val)[1]
        end
    end

    #Define the list of action
    action =[]
    indices = findall(x->x >= 1,Int.(predictions))
    if length(indices) == 0
	    action =[]
    elseif length(indices) <= K
	    for (idx, a) in enumerate(indices)
            push!(action, a)
        end
    else
        #Define the vector of costs difference
        Costs = zeros(M)
        for m in 1:M
            Costs[m] = costs_f[m] - costs_r[m]
        end
        #Sort the objective values in the decreasing order
        indices_costs = sortperm(Costs,rev=true)
        for (idx, m) in enumerate(indices_costs)
            if idx <= K
                cost_temp = Costs[Costs .== Costs[m]]
                ind_temp = findall(x-> x == Costs[m], Costs)
                if length(cost_temp) == 1
                    push!(action, m)
                else
                    #Draw randomly bewteen the equal number
                    temp = sample(Int.(ind_temp))
                    push!(action, temp)     
                    #Set a negative cost to be sure that you will not draw it twice
                    Costs[temp] = -1
                end
            end
        end
    end
    println(action)
    return action
end
