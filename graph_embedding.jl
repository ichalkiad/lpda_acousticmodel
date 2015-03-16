function compute_weight_mats{T,Q}(X::AbstractMatrix{T}, C::AbstractMatrix{Q}, kint::Int64,kpen::Int64, Rint::Float64, Rpen::Float64, dist::String)

#add inbounds after testing

    m, n = size(X)

    # r : inner products of samples or normalized samples
    r = Array(Float64, (n, n))

    neighb_sameClass = Array(Int,n,1)
    neighb_diffClass = Array(Int,n,1)
    temp = Array(Int64,n,1)

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
	#temp holds the indices of the neighbours in growing distance order
	temp[:] = sortperm(r[:,j])

	p = 1
	pp = 1
	#find kint same-class NNs and kpen diff-class NNs
	i = 1
	while (i <= n && (p<=kint+1 || pp<=kpen))
	    if (C[temp[i]]==C[j] && p<=kint+1)
	       neighb_sameClass[p] = temp[i]
	       p = p + 1
	    elseif (C[temp[i]]!=C[j] && pp<=kpen)
	       neighb_diffClass[pp] = temp[i]
	       pp = pp + 1
	    end
	    i = i + 1
	end
 
	# e : edges,indices of neighbours in X
	if (p-1 > kint)
 	   e_int = Array(Int, kint, 1)
	   e_int[:] = neighb_sameClass[2:kint+1]
	else
	   e_int = Array(Int, p-2, 1)
     	   e_int[:] = neighb_sameClass[2:p-1]
	end

	if (pp-1 > kpen)
	    e_pen = Array(Int, kpen, 1)
      	    e_pen[:] = neighb_diffClass[1:kpen]
	else
     	   e_pen = Array(Int, pp-1, 1)
           e_pen[:] = neighb_diffClass[1:pp-1]
	end

  	# d : distances of neighbours (of same and diff class) for current sample
	d_int = Array(Float64, size(e_int))
  	d_pen = Array(Float64, size(e_pen))

 	#compute weights
	if (dist=="locality")
     	   d_int[:] = r[e_int[:],j]
	   d_pen[:] = r[e_pen[:],j]
	elseif (dist=="correlation")
	   # 1-inner_product because of -d, l. 56,59
	   d_int[:] = 1-r[e_int[:],j]
	   d_pen[:] = 1-r[e_pen[:],j]
	end

  	for i = j+1 : n
	    idx_int = findfirst(e_int[:],i)
	    idx_pen = findfirst(e_pen[:],i)
	    if (idx_int!=0)
	       Wint[i,j] = exp(-d_int[idx_int]/Rint)
	       Wpen[i,j] = 0
	    elseif (idx_pen!=0)
	       Wpen[i,j] = exp(-d_pen[idx_pen]/Rpen)
	       Wint[i,j] = 0
	    else
	       Wint[i,j] = 0
	       Wpen[i,j] = 0
	    end
        end
    
    end

    return (Wint, Wpen)

end


