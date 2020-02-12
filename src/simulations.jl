include("policies.jl")

function init_belief(obs,m,p_init,params)
    """ Function that returns the initial belief state of component m
        Returns [Array{Float64}: nbS[m] x 1]
        
        obs: observation on component m o^m (Int64)
        m: index of the component (Int64)
        nbS: number of states on component m (List of M Int64) 
        p_init: initial state probability distribution on each component p^m(s) (List of M Array{Float64}: nbS[m] x 1) 
        p_emis: emission probability distribution on each component p^m(o|s) (List of M Array{Float64}: nbS[m] x nbO[m]) 
    """
    #Define the current belief state
    b_temp = zeros(params.nbS[m])
    for s in 1:params.nbS[m]
        b_temp[s] = params.p_emis[m][s,obs[m]]*p_init[m][s]
    end
    # println(p_init[m])
    #Check that the computation is correct
    if sum(b_temp) == 0
        b_temp = p_init[m]
    else
        b_temp = b_temp/sum(b_temp)
    end
    return b_temp
end

function update_belief(obs,action,b,m,params)
    """ Function that returns the current belief state of component m
        Returns [Array{Float64}: nbS[m] x 1]
        
        obs: observation on component m o^m (Int64)
        m: index of the component (Int64)
        params: POMDP parameters (Object pomdp_parameters)
    """
    #Define the current belief state
    b_temp = zeros(params.nbS[m])

    #If the component is not maintained, we update the blief state
    if m in action
        for s in 1:params.nbS[m]
	       b_temp[s] = params.p_emis[m][s,obs[m]]*sum(params.p_trans[m][ss,2,s]*b[m][ss] for ss in 1:params.nbS[m])
        end    
    else
        for s in 1:params.nbS[m]
           b_temp[s] = params.p_emis[m][s,obs[m]]*sum(params.p_trans[m][ss,1,s]*b[m][ss] for ss in 1:params.nbS[m])
        end
    end

    #Check that the state probability distribution is correctly defined 
    #If it is not the case
    if sum(b_temp) == 0
        #Then the belief state is the same
        b_temp = b[m]
    else
        b_temp = b_temp/sum(b_temp)
    end
    return b_temp
end


function simulation(T_sim,h,sample_size,M,K,sim_params,dtrees,policy,dtrees_industry,costs_f,costs_r,T,pomdp_params,continuous_obs,discrete_obs)
    """ Function that runs MILP-based heuristic policy
        Returns [Total cost of the simulation (Float64), Number of component maintained (Float64), Number of failures (Float64), Computation time (Float64)]
        
        
        T_sim: Number of steps in the simulation (Int64)
        h: interval time between two maintenance slots (Int64)
        sample_size: size of the sample of trajectories (Int64)
        M: Number of components (Int64) 
        K: maintenance capacity (Int64)
        sim_params: Obj simulation_parameters 
        dtrees: decision trees learned to discretize the observation on each component (List of M sklearn.tree.DecisionTreeClassifier)
        policy: a string specifying which policy we use (List of string ["Myopic","Industry","MILP","MILP_cuts", "NLP","LP", "LP_cuts"])
        dtrees_industry: decision trees or fault trees used in industry on each component (List of M sklearn.tree.DecisionTreeClassifier)
        T: rolling horizon for mathematical programs (Int64)
        pomdp_params: define POMDP parameters (object pomdp_parameters)
        features: initial continuous observation trajectorie on each component X_t^m (Array{Float64}: sample_size until failure x dimension[m])
        discrete_obs: initial discrete observation trajectories on each component O_t^m (Array{Int64}: sample_size until failure x 1)
    """
    #Define the parameter gamma_e
    
    #Initial cost
    cost = 0
    
    #Nb of failure
    fails = 0
    
    #Nb of replacement
    repls = 0 
   
    #Policy Time
    policy_time = 0
    
    #initial trajectories
    discrete_trajectories = discrete_obs
    
    #continuous trajectories 
    continuous_trajectories = continuous_obs
    
    #Define current observation
    action = []
    
    #time of decision 
    time_of_decision = zeros(M)
    #Store the initial probability distributions
    p_init =[]
    for m in 1:M
        push!(p_init,pomdp_params.p_init[m])
    end
    #Take belief state in memory
    b_memory = p_init

    #Horizon 
    for t in 1:T_sim
        
        #update time of decision
        time_of_decision += h*ones(M)
	
        #Check if an equipment dies during the period t-1,t
        #And select the current vector of observations
        obs = zeros(M)
        failures = zeros(M)
        for m in 1:M
            
            #If length(trajectory) < time, then failure occurs
            # println(length(discrete_trajectories[m]))
            if size(continuous_trajectories[m],1) < time_of_decision[m]
                
                #Here add the failure cost    
                cost += costs_f[m]
        
                #Failure happens, we replace the equipment between two maintenance slots
                println("Failure happened on component ",m)
                failures[m] = 1
                fails +=1
                
                ######Second Option : if a failure happens the equipment is replaced only whent the DM does it###
                time_of_decision[m] = time_of_decision[m] - h
                ###################################################################
                
            end
            #Initial decision time
            if time_of_decision[m] == 0
                time_of_decision[m] += 1
            end

            #Observation on trajectories
            # println(time_of_decision[m])
            obs[m] = discrete_trajectories[m][Int.(time_of_decision[m])]
	    end
        
        #Update the belief state
        belief_state = []
        for m in 1:M
            # b = zeros(pomdp_params.nbS[m])
            if failures[m] > 0 
                vect_fail = zeros(pomdp_params.nbS[m])
                vect_fail[pomdp_params.nbS[m]] = 1.0
                push!(belief_state, vect_fail)
            else
                push!(belief_state, update_belief(Int.(obs),action,b_memory,m,pomdp_params))
            end
        end
        b_memory = belief_state
        pomdp_params.p_init =  belief_state
        for m in 1:M
            println(belief_state[m])
        end
        #####################################################################################################
        
        ##############################Optimization block#####################################################
        #Given obs
        #Define the rolling horizon Delta
        Δ = 1
        if T_sim - t <= T
            Δ = T_sim - t +1
        else
            Δ = T
        end

        #Define the policy
        if policy == "Myopic"
	        action = def_myopic_policy(models,M,K,pomdp_params)	
	    elseif policy == "Industry"
            val = []
            for m in 1:M
                # println(continuous_trajectories[m][Int.(time_of_decision[m]),:])
                push!(val,continuous_trajectories[m][Int.(time_of_decision[m]),:])
            end
            # println(val)
            action = def_industry_policy(M,K,dtrees_industry,val,failures,costs_f,costs_r)
	    elseif policy == "MILP"
	        action,val, time = def_policy_MILP(Δ,M,K,pomdp_params,Int.(obs),0) 
	        policy_time +=time
	    elseif policy == "MILP_cuts"
	        action,val, time = def_policy_MILP(Δ,M,K,pomdp_params,Int.(obs),1) 	
	        policy_time +=time
	    elseif policy == "NLP"
            action,val, time = def_policy_NLP(Δ,M,K,pomdp_params,Int.(obs))
            policy_time +=time
	    elseif policy == "LP"  
            action,val, time = def_policy_LP(Δ,M,K,pomdp_params,Int.(obs),0)
            policy_time +=time
        elseif policy == "LP_cuts"
            action,val, time = def_policy_LP(Δ,M,K,pomdp_params,Int.(obs),1)
            policy_time +=time
	    end
 	            
        if length(action) == 0
            #The output of this block is an action a in {1,...,M+1}
            #####################################################################################################
            println("Time ", t, " : do nothing")
            
        else
	       #The output of this block is a vector of action 
 	        for (idx,a) in enumerate(action)
		        println("Time ", t, " : replace component ", a)
            	#Change the sequence of the equipment we remove
            	cost += costs_r[a]
                repls +=1
                y,x,Lengths = get_sample_data(rand(Int.(sim_params[a].dimension)),sample_size,sim_params[a])
               
                continuous_trajectories[a] = x
                println(size(continuous_trajectories[a],1))
                labels = dtrees[a].predict(x)
                discrete_trajectories[a] = labels.+1
                time_of_decision[a] = 0
                # failures[a] = 0
	        end
	    end
    end
    #update time of decision
    time_of_decision .+= h*ones(M)
    failures = zeros(M)
    for m in 1:M
        if size(continuous_trajectories[m],1) < time_of_decision[m]
                
            #Here add the failure cost    
            cost += costs_f[m]
            fails += 1
            
            #Failure happens, we replace the equipment between two maintenance slots
            println("Failure happened on component ",m)
               
            ######Second Option : if a failure happens the equipment is replaced only whent the DM does it###
            time_of_decision[m] = time_of_decision[m] - h
            #timebeforeDeath[m] = 1
            ###################################################################
                
        end
    end
    
    return cost, fails, repls, policy_time/T_sim
end
