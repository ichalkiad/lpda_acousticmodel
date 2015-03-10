function compute_weight_mats{T,Q}(X::AbstractMatrix{T}, C::AbstractMatrix{Q}, k::Int64, Rint::Int64, Rpen::Int64, dist::String) 

#add inbounds after testing


    m, n = size(X)
    # r : inner products of samples or normalized samples
    r = Array(Float64, (n, n))
    # d : distances of neighbours for current sample
    d = Array(Float64, k, 1)
    # e : edges,indices of neighbours in X
    e = Array(Int, k, 1)
    Wint = Array(Float64,(n,n))
    Wpen = Array(Float64,(n,n))
    
    sa2 = sum(X.^2, 1)
    if (dist=="locality")
       At_mul_B!(r, X, X)
    elseif (dist=="correlation")
       #projection on unit hypersphere
       X_norm = Array(Float64,(m,n))
       for j = 1 : n
       	   X_norm[:,j] = X[:,j]./sqrt(sa2[j])
       end
       At_mul_B!(r, X_norm, X_norm)
    end

    #construct weight matrices
    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
	    @inbounds Wint[i,j] = Wint[j,i]
	    @inbounds Wpen[i,j] = Wpen[j,i]
        end
        @inbounds r[j,j] = 0
	@inbounds Wint[j,j] = 1
	@inbounds Wpen[j,j] = 1
        if (dist=="locality") 
	  #euclidean distance 
	  for i = j+1 : n
	    @inbounds v = sa2[i] + sa2[j] - 2 * r[i,j]
            @inbounds r[i,j] = isnan(v) ? NaN : max(v, 0.)
	  end   	
	end
	e[:] = sortperm(r[:,j])[2:k+1]
	if (dist=="locality")
           d[:] = r[e[:],j]
	elseif (dist=="correlation")
	   # 1-inner_product because of -d, l. 56,59
	   d[:] = 1-r[e[:],j]
	end    	
	for i = j+1 : n	      
	    idx = findfirst(e[:],i)
	    if (idx!=0)
               if C[i] == C[j]
	       	   Wint[i,j] = exp(-d[idx]/Rint)
		   Wpen[i,j] = 0
	       else
		   Wpen[i,j] = exp(-d[idx]/Rpen)
		   Wint[i,j] = 0
	       end
	    else
	       Wint[i,j] = 0
	       Wpen[i,j] = 0
	    end
        end	

    end

    return (Wint, Wpen)

end
