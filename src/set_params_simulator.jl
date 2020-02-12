struct simulation_parameters
    #This object contains the simulation parameters for a single component

    #Dimension d of features
    dimension

    #Define number of state for the propagation rate γ 
    nb_degrad_states
    
    #Define the range of value for the propagation rate γ 
    gamma_e

    #Define state transition matrix for the propagation rate
    transmat

    #Define the time step of the simulation 
    delta_t

    #Define parameter n (material property) 
    n

    #Define variance of noise σ_{w}^2   
    sigma_w

    #Define parameter C (material property) 
    C

    #Define the base stress level 
    beta_b

    #Define the covariance matrix Σ_{ξ}  
    sigma_xhi
end


function set_params_simulator(M)
    """Function that sets the simulator parameters
       Returns a list of M object simulation_parameters

       M: number of components (Int64)
    """
    dimensions = zeros(M) 
    nb_degrad_states = zeros(M)
    gamma_es = []
    transmats = []
    Cs =zeros(M)
    ns = zeros(M)
    beta_bs = zeros(M)
    
    sigma_ws = []
    sigma_xhis = []
    
    for m in 1:M
        #Define the dimension d of features in \mathbb{R}^d
        dimensions[m] = Int.(rand(3:10))
        
        
        #Define number of degradation state
        #Draw \mathcal{X}_{Γ}      
        nb_degrad_states[m] = Int.(rand(10:20))
        
        #Define the range of value for the propagation rate
        #Set the maximum possible value of Γ
        gamma_e_max = 1.2
        if nb_degrad_states[m] > 1
            push!(gamma_es,0:(gamma_e_max/nb_degrad_states[m]):gamma_e_max)
        else 
            push!(gamma_es,[0])
        end
        
        #Define the transition matrix of propagation rates p_{Γ}
        #We draw it such that it does not propagate too fast
        transmat = zeros(Int.(nb_degrad_states[m]),Int.(nb_degrad_states[m]))
        #println(nb_degrad_states[m])
        for state_1 in 1:Int.(nb_degrad_states[m])
            for state_2 in state_1:Int.(nb_degrad_states[m])
                if state_1 == state_2
                    transmat[state_1,state_2] = 10*rand()
                else
                    transmat[state_1,state_2] = exp(-state_2+rand())
                end
            end
        end
        transmat = transmat./sum(transmat,dims=2)
        push!(transmats, transmat)
        
        #Define covariance matrice Σ_{w}
        #||| Σ_{w} ||| = 1.7
        sigma_w = rand(Int.(dimensions[m]),Int.(dimensions[m]))
        sigma_w = transpose(sigma_w)*sigma_w
        eig_val_w = maximum(eigvals(sigma_w))
        sigma_w = (1.7/eig_val_w)*sigma_w
        push!(sigma_ws,sigma_w)
        
        #Define covariance matrice Σ_{ξ}
        #||| Σ_{ξ} ||| = 100.0
        sigma_xhi = rand(Int.(dimensions[m]),Int.(dimensions[m]))
        sigma_xhi = transpose(sigma_xhi)*sigma_xhi
        sigma_xhi = (100.0/maximum(eigvals(sigma_xhi)))*sigma_xhi
        push!(sigma_xhis,sigma_xhi)
        
        #draw uniformly a number between 0 and 1
        u = rand()
        #C ∈ [0.001;0.01]
        Cs[m] = (0.01 - 0.001)*u + 0.001
        
        #draw uniformly a number between 0 and 1
        u = rand()
        #n ∈ [1.0;2.0]
        ns[m] = (2.0 - 1.0)*u + 1.0
    
        #draw uniformly a number between 0 and 1
        u = rand()
        #β_b ∈ [0.5;1.5]
        beta_bs[m] = (1.5 - 0.5)*u + 0.5
    end
    #Set the simulation time step
    delta_t=1.0

    #Critical value 
    d=100

    #Set the params
    params = []
    for m in 1:M
        push!(params,simulation_parameters(Int.(dimensions[m]),Int.(nb_degrad_states[m]),gamma_es[m],transmats[m],delta_t,ns[m],sigma_ws[m],Cs[m],beta_bs[m],sigma_xhis[m]))
    end
    return params
end