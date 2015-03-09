function weightMatNN{T,Q}(X::AbstractMatrix{T},C::AbstractMatrix{Q}, Rint::Int,Rpen::Int,k::Int=12)

    m, n = size(X)
    r = Array(T, (n, n))
    d = Array(T, k, 1)
    e = Array(Int, k, 1)
    Wint = Array(Float64,(n,n))
    Wpen = Array(Float64,(n,n))

    At_mul_B!(r, X, X)
    sa2 = sum(X.^2, 1)

    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
	    @inbounds Wint[i,j] = Wint[j,i]
	    @inbounds Wpen[i,j] = Wpen[j,i]
        end
        @inbounds r[j,j] = 0
	@inbounds Wint[j,j] = 1
	@inbounds Wpen[j,j] = 1
        for i = j+1 : n
            @inbounds v = sa2[i] + sa2[j] - 2 * r[i,j]
            @inbounds r[i,j] = isnan(v) ? NaN : max(v, 0.)
        end
	e[:] = sortperm(r[:,j])[2:k+1]
        d[:] = r[e[:],j]
	
	for i = j+1 : n	      #add inbounds after testing
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





#function find_nn{T,Q}(X::AbstractMatrix{T},C::AbstractMatrix{Q}, Rint::Int,Rpen::Int,k::Int=12)

#    m, n = size(X)
#    r = Array(T, (n, n))
#    d = Array(T, k, n)
#    e = Array(Int, k, n)

#    At_mul_B!(r, X, X)
#    sa2 = sum(X.^2, 1)

#    for j = 1 : n
#        for i = 1 : j-1
#            @inbounds r[i,j] = r[j,i]
#        end
#        @inbounds r[j,j] = 0
#        for i = j+1 : n
#            @inbounds v = sa2[i] + sa2[j] - 2 * r[i,j]
#            @inbounds r[i,j] = isnan(v) ? NaN : max(v, 0.)
#        end
#        e[:, j] = sortperm(r[:,j])[2:k+1]
#        d[:, j] = r[e[:, j],j]
#    end

#    Wint = Array(Float64,(n,n))
#    Wpen = Array(Float64,(n,n))
#    for j = 1:n
#    	for i = 1 : j-1
#            @inbounds Wint[i,j] = Wint[j,i]
#	    @inbounds Wpen[i,j] = Wpen[j,i]
#        end
#	@inbounds Wint[j,j] = 1
#	@inbounds Wpen[j,j] = 1
#	for i = j+1 : n	      #add inbounds after testing
#	    idx = findfirst(e[:,j],i)
#	    if (idx!=0)
#               if C[i] == C[j]
#	       	   Wint[i,j] = exp(-d[idx,j]/Rint)
#		   Wpen[i,j] = 0
#	       else
#		   Wpen[i,j] = exp(-d[idx,j]/Rpen)
#		   Wint[i,j] = 0
#	       end
#	    else
#	       Wint[i,j] = 0
#	       Wpen[i,j] = 0
#	    end
#       end
#    end	

#    return (d, e, Wint, Wpen)

#end