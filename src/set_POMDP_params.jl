mutable struct pomdp_parameters
    #Define number of components
    M

    #Define number of states on each component: List of M Int64
    nbS

    #Define number of observation on each component: List of M Int64
    nbO

    #Define the number of action: 2
    nbA

    #Define the initial probability distribution: List of M Array{Float64}
    p_init

    #Define the emission probability distribution: List of M Array{Float64} (nbS[m] x nbO[m])
    p_emis

    #Define the state transition probability distribution: List of M Array{Float64} (nbS[m] x 2 x nbS[m])
    p_trans

    #Define the independence probability distribution: List of M Array{Float64} (nbS[m] x 2 x nbO[m] x nbS[m])
    p_cuts

    #Define the rewards: List of M Array{Float64} (nbS[m] x 2 x nbS[m])
    reward
end

function def_reward(M,models,cost_f,cost_r)
    """Function that set the independence probability distributions
       Returns a list of M Array{Float64}: nbS[m] x 2 x nbO[m] x nbS[m]
        
       M: number of components (Int64)
       models: list of HMM parameters (list of M hmmlean.hmm.GaussianHMM)
       cost_f: failure costs (Array{Float64}: M x 1)
       cost_r: maintenance costs (Array{Float64}: M x 1)
    """
    reward = [] 
    for m in 1:M
        nb_states = Int.(models[m].n_components)
        r = zeros(nb_states,2,nb_states)
        for s in 1:nb_states, ss in 1:nb_states
            if ss == nb_states
                r[s,1,ss] = r[s,1,ss] - cost_f[m]
            end
            r[s,2,ss] = r[s,2,ss] - cost_r[m]
        end
        push!(reward,r)
    end
    return reward
end

function def_pcuts(M,nbS,nbO,p_trans,p_emis)
    """Function that set the independence probability distributions
       Returns a list of M Array{Float64}: nbS[m] x 2 x nbO[m] x nbS[m]
        
       M: number of components (Int64)
       nbS: number of states on components (Array{Int64}: M x 1) 
       nbO: number of observations on each component (Array{Int64}: M x 1)
       p_trans: state transition probability distribution on each component p^m(s'|s,a) (List of M Array{Float64}: nbS[m] x 2 x nbS[m]) 
       p_emis: emission probability distribution on each component p^m(o|s) (List of M Array{Float64}: nbS[m] x nbO[m]) 
    """
    p_cuts = []
    A = 1:2
    for m in 1:M
        proba = zeros(nbS[m],2,nbS[m],nbO[m])
        for s in 1:nbS[m], a in A, ss in 1:nbS[m], oo in 1:nbO[m]
            if sum(p_trans[m][s,a,sss]*p_emis[m][sss,oo] for sss in 1:nbS[m]) == 0
                proba[s,a,ss,oo] = 0
            else
                proba[s,a,ss,oo] = (p_trans[m][s,a,ss]*p_emis[m][ss,oo])/(sum(p_trans[m][s,a,sss]*p_emis[m][sss,oo] for sss in 1:nbS[m]))
            end
        end
        push!(p_cuts,proba)
    end
    return p_cuts
end


function set_pomdp_parameters(M,h,models,emissions,costs_f,costs_r)
    """Function that set POMDP parameters
       Returns a pomdp_parameters object  
        
       M: number of components (Int64)
       h: interval time between maintenance slots (Int64)
       models: HMM parameters (list of M scklearn.hmmlearn)
       emissions: emission probability distribution of the discrete observations (List of M Array{Float64}: nbS[m] x nbO[m]) 
    """
    nbS = zeros(M)
    nbO = zeros(M)
    nbA = 2 #if a=m, then replace equipment m, if a=M+1, then do nothing
    p_emis = []
    p_trans = []
    p_init =[]
    
    for m in 1:M
        nbS[m] = Int.(models[m].n_components)
        nbO[m] = Int.(models[m].n_components)
        push!(p_init,models[m].startprob_)
        push!(p_emis,emissions[m])
        
        transmat = zeros(Int.(nbS[m]),nbA,Int.(nbS[m]))
	    transmat[:,1,:] = (models[m].transmat_)^h
	    transmat[:,2,:] = ones(Int.(nbS[m]))*transpose(models[m].startprob_)
        push!(p_trans,transmat)
    end

    reward = def_reward(M,models,costs_f,costs_r)
    p_cuts = def_pcuts(M,Int.(nbS),Int.(nbO),p_trans,p_emis)
    params = pomdp_parameters(M,Int.(nbS),Int.(nbO),nbA,p_init,p_emis,p_trans,p_cuts,reward)
    return params
end


