function indicator_function(observation,o)
    if o == observation
        return 1
    else
        return 0
    end
end

function relax_model(model)
    for v in all_variables(model)
      if is_integer(v)
        unset_integer(v)
        # If applicable, also round the lower and upper bounds if they're not integer.
      elseif is_binary(v)
        unset_binary(v)
        if has_lower_bound(v) && and lower_bound(v) > 0
          set_lower_bound(v, 1)
        else
          set_lower_bound(v, 0)
        end
        if has_upper_bound(v) && and upper_bound(v) < 1
          set_upper_bound(v, 0)
        else
          set_upper_bound(v, 1)
        end
        set_upper_bound(v, 1)
      end
    end
  # If applicable, also handle semi-integer and semicontinuous.
    return model
end

function model_decPOMDP(T,M,K,nbS,nbO,A,p_init,p_trans,p_emis,reward,initial_obs)
    ## Mathematical Program
    """ Function that builds the model of our MILP approximation.
        Returns [JuMP.Model()]
        
        T: horizon (Int64)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        nbS: number of states on components (vector of integers: M x 1) 
        nbO: number of observations on each component (Array{Int64}: M x 1)
        A: action set (Array{Int64}: 2 x 1)
        p_init: initial state probability distribution on each component p^m(s) (List of M Array{Float64}: nbS[m] x 1) 
        p_trans: state transition probability distribution on each component p^m(s'|s,a) (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
        p_emis: emission probability distribution on each component p^m(o|s) (List of M Array{Float64}: nbS[m] x nbO[m]) 
        reward: reward on each component r^m(s,a,s') (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
        initial_obs: discrete observation on each component (Array{Int64}: M x 1)
    """
    
    model_milp = JuMP.direct_model(Gurobi.Optimizer(OutputFlag=0, TimeLimit=3600))
    # set_optimizer_attributes(model_milp, OutputFlag=>"0", TimeLimit=>"3600")
    
    @variables model_milp begin
        τ_soa[1:T,m=1:M,1:nbS[m],1:nbO[m],A]>=0
        τ_sa[1:T,m=1:M,1:nbS[m],A]>=0
        τ_s[1:T+1,m=1:M,1:nbS[m]]>=0
        τ_a[1:T,1:M+1,A] >=0
	    alpha[1:T,m=1:M,1:nbO[m]], Bin
    end
    
    @objective(model_milp,Max, sum(τ_sa[t,m,s,a]*p_trans[m][s,a,ss]*reward[m][s,a,ss] for t in 1:T, m in 1:M, s in 1:nbS[m], a in A, ss in 1:nbS[m]))

    for t in 1:T, m in 1:M, s in 1:nbS[m]
        @constraint(model_milp,sum(τ_sa[t,m,ss,aa]*p_trans[m][ss,aa,s] for ss in 1:nbS[m], aa in A) == τ_s[t+1,m,s])
    end
    
    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m]
        if t >=1
        	@constraint(model_milp,sum(τ_soa[t,m,s,o,aa] for aa in A) == p_emis[m][s,o]*τ_s[t,m,s])
        end
        # else 
        # 	@constraint(model_milp,sum(τ_soa[t,m,s,o,aa] for aa in A) == indicator_function(initial_obs[m],o)*τ_s[t,m,s])
        # end
    end
    
    for t in 1:T, m in 1:M, s in 1:nbS[m]
        @constraint(model_milp,sum(τ_sa[t,m,s,aa] for aa in A) == τ_s[t,m,s])
    end
    
    for t in 1:T, m in 1:M, s in 1:nbS[m], a in A
        @constraint(model_milp,sum(τ_soa[t,m,s,oo,a] for oo in 1:nbO[m]) == τ_sa[t,m,s,a])
    end
    
    for t in 1:T, m in 1:M, a in A
        @constraint(model_milp,sum(τ_sa[t,m,ss,a] for ss in 1:nbS[m]) == τ_a[t,m,a])
    end
    
    for m in 1:M, s in 1:nbS[m]
        @constraint(model_milp,τ_s[1,m,s] == p_init[m][s])
    end

    #for m in 1:M, s in 1:nbS[m]
    #	@constraint(model_relax,τ_s[T+1,m,s] == sum(τ_sa[T,m,ss,aa]*p_trans[m][ss,aa,s] for ss in 1:nbS[m], aa in A) + discount*sum(τ_sa[T+1,m,ss,aa]*p_trans[m][ss,aa,s] for ss in 1:nbS[m], aa in A))
    	#@constraint(model_relax,τ_s[T+1,m,s] == sum(τ_sa[T,m,ss,aa]*p_trans[m][ss,aa,s] for ss in 1:nbS[m], aa in A))
    #end

    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m]
    	# if t==1
     #        @constraint(model_milp, τ_soa[t,m,s,o,2] <= indicator_function(initial_obs[m],o)*alpha[t,m,o])
     #        @constraint(model_milp, τ_soa[t,m,s,o,2] >= indicator_function(initial_obs[m],o)*(τ_s[t,m,s] - (1 - alpha[t,m,o])))
     #        @constraint(model_milp, τ_soa[t,m,s,o,2] <= indicator_function(initial_obs[m],o)* τ_s[t,m,s])
     #    end

        if t >= 1
            @constraint(model_milp, τ_soa[t,m,s,o,2] <= alpha[t,m,o])
            @constraint(model_milp, τ_soa[t,m,s,o,2] >= p_emis[m][s,o]*(τ_s[t,m,s] - (1 - alpha[t,m,o])))
            @constraint(model_milp, τ_soa[t,m,s,o,2] <= p_emis[m][s,o]* τ_s[t,m,s])
        end
    end


    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m]
        # if t==1
        #     @constraint(model_milp, τ_soa[t,m,s,o,1] <= indicator_function(initial_obs[m],o)*(1-alpha[t,m,o]))
        #     @constraint(model_milp, τ_soa[t,m,s,o,1] >= indicator_function(initial_obs[m],o)*(τ_s[t,m,s] - alpha[t,m,o]))
        #     @constraint(model_milp, τ_soa[t,m,s,o,1] <= indicator_function(initial_obs[m],o)* τ_s[t,m,s])
        # end

        if t >= 1
            @constraint(model_milp, τ_soa[t,m,s,o,1] <= 1-alpha[t,m,o])
            @constraint(model_milp, τ_soa[t,m,s,o,1] >= p_emis[m][s,o]*(τ_s[t,m,s] - alpha[t,m,o]))
            @constraint(model_milp, τ_soa[t,m,s,o,1] <= p_emis[m][s,o]*τ_s[t,m,s])
        end
    end

    for t in 1:T
	   @constraint(model_milp, sum(τ_a[t,m,2] for m in 1:M+1) == K)
    end
    
    return model_milp
end

function model_decPOMDP_NLP(T,M,K,nbS,nbO,A,p_init,p_trans,p_emis,reward,initial_obs)
    ## Mathematical Program
    """ Function that builds the model of our NLP approximation.
        Returns [JuMP.Model()]
        
        T: horizon (Int64)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        nbS: number of states on components (vector of integers: M x 1) 
        nbO: number of observations on each component (Array{Int64}: M x 1)
        A: action set (Array{Int64}: 2 x 1)
        p_init: initial state probability distribution on each component p^m(s) (List of M Array{Float64}: nbS[m] x 1) 
        p_trans: state transition probability distribution on each component p^m(s'|s,a) (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
        p_emis: emission probability distribution on each component p^m(o|s) (List of M Array{Float64}: nbS[m] x nbO[m]) 
        reward: reward on each component r^m(s,a,s') (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
        initial_obs: discrete observation on each component (Array{Int64}: M x 1)
    """

    # SOLVER_nlp =  GurobiSolver(OutputFlag = 1,TimeLimit=3600, NonConvex = 2)
    model_nlp = JuMP.direct_model(Gurobi.Optimizer(OutputFlag=0, TimeLimit=3600,NonConvex=2))

    @variables model_nlp begin
        τ_soa[1:T,m=1:M,1:nbS[m],1:nbO[m],A]>=0
        τ_sa[1:T,m=1:M,1:nbS[m],A]>=0
        τ_s[1:T+1,m=1:M,1:nbS[m]]>=0
        τ_a[1:T,1:M+1,A] >=0
        alpha[1:T,m=1:M,1:nbO[m]] >=0
    end

    @objective(model_nlp,Max, sum(τ_sa[t,m,s,a]*p_trans[m][s,a,ss]*reward[m][s,a,ss] for t in 1:T, m in 1:M, s in 1:nbS[m], a in A, ss in 1:nbS[m]))

    for t in 1:T, m in 1:M, s in 1:nbS[m]
        @constraint(model_nlp,sum(τ_sa[t,m,ss,aa]*p_trans[m][ss,aa,s] for ss in 1:nbS[m], aa in A) == τ_s[t+1,m,s])
    end

    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m]
        if t >1
                @constraint(model_nlp,sum(τ_soa[t,m,s,o,aa] for aa in A) == p_emis[m][s,o]*τ_s[t,m,s])
        else
                @constraint(model_nlp,sum(τ_soa[t,m,s,o,aa] for aa in A) == indicator_function(initial_obs[m],o)*τ_s[t,m,s])
        end
    end

    for t in 1:T, m in 1:M, s in 1:nbS[m]
        @constraint(model_nlp,sum(τ_sa[t,m,s,aa] for aa in A) == τ_s[t,m,s])
    end

    for t in 1:T, m in 1:M, s in 1:nbS[m], a in A
        @constraint(model_nlp,sum(τ_soa[t,m,s,oo,a] for oo in 1:nbO[m]) == τ_sa[t,m,s,a])
    end

    for t in 1:T, m in 1:M, a in A
        @constraint(model_nlp,sum(τ_sa[t,m,ss,a] for ss in 1:nbS[m]) == τ_a[t,m,a])
    end

    for m in 1:M, s in 1:nbS[m]
        @constraint(model_nlp,τ_s[1,m,s] == p_init[m][s])
    end

    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m], a in 1:2
    	@constraint(model_nlp,sum(τ_soa[t,m,s,o,aa] for aa in A) == p_emis[m][s,o]*τ_s[t,m,s])
        #@constraint(model_relax,sum(τ_soa[t,m,s,o,aa] for aa in A) == indicator_function(initial_obs[m],o)*τ_s[t,m,s])
    end

    for t in 1:T
        @constraint(model_nlp, sum(τ_a[t,m,2] for m in 1:M+1) == K)
    end

    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m]
        if t==1
            @constraint(model_nlp, τ_soa[t,m,s,o,2] == indicator_function(initial_obs[m],o)*alpha[t,m,o]*τ_s[t,m,s])
        end
        if t > 1
            @constraint(model_nlp, τ_soa[t,m,s,o,2] == p_emis[m][s,o]*τ_s[t,m,s]*alpha[t,m,o])
        end
    end


    for t in 1:T, m in 1:M, s in 1:nbS[m], o in 1:nbO[m]
        if t==1
            @constraint(model_nlp, τ_soa[t,m,s,o,1] == indicator_function(initial_obs[m],o)*(1-alpha[t,m,o])*τ_s[t,m,s])
        end
        if t > 1
            @constraint(model_nlp, τ_soa[t,m,s,o,1] == p_emis[m][s,o]*τ_s[t,m,s]*(1-alpha[t,m,o]))
        end
    end


    return model_nlp
end



function model_decPOMDP_cuts(T,M,K,nbS,nbO,A,p_init,p_trans,p_emis,p_cuts,reward,initial_obs)
    """ Function that build the model of our MILP approximation with valid inequalities.
        Returns [JuMP.Model()]
        
        T: horizon (Int64)
        M: number of components (Int64)
        K: maintenance capacity (Int64)
        nbS: number of states on components (vector of integers: M x 1) 
        nbO: number of observations on each component (Array{Int64}: M x 1)
        A: action set (Array{Int64}: 2 x 1)
        p_init: initial state probability distribution on each component p^m(s) (List of M Array{Float64}: nbS[m] x 1) 
        p_trans: state transition probability distribution on each component p^m(s'|s,a) (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
        p_emis: emission probability distribution on each component p^m(o|s) (List of M Array{Float64}: nbS[m] x nbO[m])
        p_cuts: independence probability distribution on each component p^m(s'|s,a,o) (List of M Array{Float64}: nbS[m] x 2 x nbO[m] x nbS[m])
        reward: reward on each component r^m(s,a,s') (List of M Array{Float64}: nbS[m] x 2 x nbS[m])
        initial_obs: discrete observation on each component (Array{Int64}: M x 1)
    """

    model_milp = model_decPOMDP(T,M,K,nbS,nbO,A,p_init,p_trans,p_emis,reward,initial_obs)
    @variables model_milp begin
        τ_sasoa[1:T,m=1:M,1:nbS[m],A,1:nbS[m],1:nbO[m],A] >=0
    end
    
    ###########################################Local consistency constraints############################
    for t in 2:T, m in 1:M, s in 1:nbS[m], a in A, ss in 1:nbS[m], o in 1:nbO[m]
       @constraint(model_milp,sum(τ_sasoa[t,m,s,a,ss,o,aa] for aa in A) == p_emis[m][ss,o]*p_trans[m][s,a,ss]* model_milp[:τ_sa][t-1,m,s,a])
    end
    
    for t in 2:T, m in 1:M, o in 1:nbO[m], s in 1:nbS[m], a in A
       @constraint(model_milp,sum(τ_sasoa[t,m,ss,aa,s,o,a] for ss in 1:nbS[m], aa in A) ==model_milp[:τ_soa][t,m,s,o,a])
    end
    ####################################################################################################
   
    ##########################################New Valid constraints###########################################
    #τ_sasoa^t = p_(s' | sao)^t * sum(τ_sasoa^{t}, s_t) 
    for t in 2:T, m in 1:M, s in 1:nbS[m], a in A, oo in 1:nbO[m], ss in 1:nbS[m], aa in A
       @constraint(model_milp,τ_sasoa[t,m,s,a,ss,oo,aa] == p_cuts[m][s,a,ss,oo]*sum(τ_sasoa[t,m,s,a,sss,oo,aa] for sss in 1:nbS[m]))
    end
    ####################################################################################################

    return model_milp
end
