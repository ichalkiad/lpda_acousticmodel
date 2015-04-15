function compute_weight_mats!{T,Q}(Wint::SparseMatrixCSC{Float64,Int64},Wpen::SparseMatrixCSC{Float64,Int64},X::AbstractMatrix{T}, C::AbstractVector{Q}, kint::Int64,kpen::Int64, Rint::Float64, Rpen::Float64, dist::String)

#add inbounds after testing

    m, n = size(X)

    # r : distances of sample j from the rest
    r = zeros(n)

    neighb_sameClass = zeros(n)
    neighb_diffClass = zeros(n)
    temp = zeros(n)

    #construct weight matrices
    for j = 1 : n

	    if (dist=="locality")
	      #euclidean distance
	      for i = 1 : n
	        r[i] = norm(X[:,i]-X[:,j])^2
	      end
	   elseif (dist=="correlation")
	      #cosine correlation distance
	      for i = 1 : n
	         r[i] = dot(X[:,i],X[:,j])
        end
     end

	#display("Sample, distances")
	#display(j)
	#display(r)
	#readline(STDIN)


	#temp holds the indices of the neighbours in growing distance order
	temp[:] = sortperm(r)

  #display("neighbours indices in growing dist")
	#display(temp)
	#readline(STDIN)

	p = 1
	pp = 1
	#find kint same-class NNs and kpen diff-class NNs
	i = 1
	while (i <= n && (p<=kint || pp<=kpen))
	      #leave self out of neighhbours list
        if (temp[i] != j)
	      	 if (C[temp[i]]==C[j] && p<=kint)
	            neighb_sameClass[p] = temp[i]
	            p = p + 1
	      	 elseif (C[temp[i]]!=C[j] && pp<=kpen)
	            neighb_diffClass[pp] = temp[i]
	            pp = pp + 1
	      	 end
        end
        i = i + 1
  end

	#display("Neigh same class -- diff class")
	#display(neighb_sameClass)
	#display(neighb_diffClass)
	#readline(STDIN)

	# e : edges,indices of neighbours in X
	if (p-1 > kint)
 	   e_int = Array(Int64,kint,1)
	   e_int[:] = neighb_sameClass[1:kint]
	else
	   e_int = Array(Int64,p-1,1)
     e_int[:] = neighb_sameClass[1:p-1]
	end

	if (pp-1 > kpen)
	    e_pen = Array(Int64,kpen,1)
      e_pen[:] = neighb_diffClass[1:kpen]
	else
     	   e_pen = Array(Int64,pp-1,1)
         e_pen[:] = neighb_diffClass[1:pp-1]
	end

	#display("Indices nns same class - diff class")
	#display(e_int)
	#display(e_pen)
	#readline(STDIN)

  	#d : distances of neighbours (of same and diff class) for current sample
	d_int = zeros(size(e_int))
  	d_pen = zeros(size(e_pen))

 	#compute weights
	if (dist=="locality")
     d_int[:] = r[e_int[:]]
	   d_pen[:] = r[e_pen[:]]
	elseif (dist=="correlation")
	   # 1-inner_product because of -d, l. 109,110
	   d_int[:] = 1-r[e_int[:]]
	   d_pen[:] = 1-r[e_pen[:]]
  end

	#display("Dists nns same class - diff class")
	#display(d_int)
	#display(d_pen)
	#readline(STDIN)

	Wint_dense = Array(Float64,size(e_int))
	Wpen_dense = Array(Float64,size(e_pen))

  Wint_dense = exp(-d_int./Rint)
  Wpen_dense = exp(-d_pen./Rpen)

	#display("Wint Wpen dense")
	#display(Wint_dense)
	#display(Wpen_dense)
	#readline(STDIN)

	Wint_sp = sparsevec(vec(e_int),vec(Wint_dense),n)
	Wpen_sp = sparsevec(vec(e_pen),vec(Wpen_dense),n)

	#display("Wint Wpen sparse")
	#display(Wint_sp)
	#display(Wpen_sp)
	#readline(STDIN)

	#store in Wint/Wpen
	Wint[:,j] = Wint_sp
	Wpen[:,j] = Wpen_sp


    end

end


