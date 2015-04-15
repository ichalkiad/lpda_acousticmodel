function f_uv{T}(P::AbstractMatrix{T},x_u::AbstractVector{T},x_v::AbstractVector{T})

	 m,n = size(P)
	 xP = Array(T,(1,n))
	 At_mul_B!(xP,x_u,P)
	 Px = Array(T,(n,1))
	 At_mul_B!(Px,P,x_v)
	 fuv = xP*Px

	 return fuv

end

function F_P{T}(X::AbstractMatrix{T},P::AbstractMatrix{T},Wint::AbstractMatrix{Float64},Wpen::AbstractMatrix{Float64})

m,n = size(X)
fp = 0.0
i = 1
j = 1
while i <= n
      while j <= n
      	    if i!=j
	       fp = fp + (1-f_uv(P,X[:,i],X[:,j])/(sqrt(f_uv(P,X[:,i],X[:,i]))*sqrt(f_uv(P,X[:,j],X[:,j]))))*(Wpen[i,j]-Wint[i,j])
	    end
	    j = j + 1
      end
      i = i + 1
end
      return 2*fp

end


function gradF_P{T}(X::AbstractMatrix{T},P::AbstractMatrix{T},Wint::AbstractMatrix{Float64},Wpen::AbstractMatrix{Float64})

m,n = size(X)
gradfp = 0.0
i = 1
j = 1
while i <= n
      while j <= n
      	    if i!=j
	       s1 = f_uv(P,X[:,i],X[:,j]).*(X[:,i]*X[:,i]')./((sqrt(f_uv(P,X[:,i],X[:,i])).^3).*sqrt(f_uv(P,X[:,j],X[:,j])))
	       s2 = f_uv(P,X[:,i],X[:,j]).*(X[:,j]*X[:,j]')./(sqrt(f_uv(P,X[:,i],X[:,i])).*sqrt(f_uv(P,X[:,j],X[:,j])).^3)
	       s3 = (X[:,i]*X[:,j]' + X[:,j]*X[:,i]')./(sqrt(f_uv(P,X[:,i],X[:,i])).*sqrt(f_uv(P,X[:,j],X[:,j])))
	       gradfp = gradfp + (s1 + s2 - s3)*P*(Wpen[i,j]-Wint[i,j])
	    end
	    j = j + 1
      end
      i = i + 1
end
      return 2*gradfp

end

function cpda!{T,Q}(e_values::AbstractVector{T},e_vectors::AbstractMatrix{T},Data::AbstractMatrix{T},Labels::AbstractVector{Q},Wint::SparseMatrixCSC{Float64,Int64},Wpen::SparseMatrixCSC{Float64,Int64},dims::Int64)

	 m,n = size(Data)

   #formulate generalized eigen problem
	 D_diag = sum(Wint,1)
	 Di = diagm(vec(D_diag))

	 D_diag = sum(Wpen,1)
	 Dp = diagm(vec(D_diag))

   s1 = open("/home/yannis/Desktop/LPDA/temp.bin","w+")
   temp = mmap_array(Float64, (m,n), s1)

   s2 = open("/home/yannis/Desktop/LPDA/A.bin","w+")
   A = mmap_array(Float64, (m,m), s2)

   s3 = open("/home/yannis/Desktop/LPDA/B.bin","w+")
   B = mmap_array(Float64, (m,m), s3)

   display("Mmapped arrays")

	 A_mul_B!(temp,Data,Dp-Wpen)
	 A_mul_Bt!(A,temp,Data)

	 A_mul_B!(temp,Data,Di-Wint)
	 A_mul_Bt!(B,temp,Data)

	 #solve to get an initial estimate of projection matrix P
	 P = eigfact!(A,B)

   display("Got eigs")

   #keep desired number of real eigenvalues
   evals = P[:values]
   real_evals = Array((Float64,Int64),length(evals))

   k = 1
   for i = 1 : length(evals)
       (re,im) = (real(evals[i]),imag(evals[i]))
       if im == 0
          real_evals[k] = (re,i)
          k = k + 1
       end
   end

   sorted_evals = zeros(k-1)
   sorted_evals = sort(real_evals[1:k-1],by=x->x[1],rev=true)

   sorted_idx = zeros(k-1)
   eigen_v = zeros(k-1)
   for i = 1 : length(sorted_evals)
       sorted_idx[i] = sorted_evals[i][2]
       eigen_v[i] = sorted_evals[i][1]
   end

   if (k-1 < dims)
      print("Found only $(k-1) real eigenvalues.")
   end
   e_values[:] = eigen_v[1:min(k-1,dims)]

   #keep the corresponding eigenvectors
   e_vectors[:,:] = P[:vectors][:,sorted_idx[1:dims]]

   #P_init = e_vectors

   display("Got eigvecs")

	 #gradient descent to minimize F_P
	 #P_old = P_init
   P_old = e_vectors

	 #learning rate
	 a =  0.1
	 #tolerance in F_P increase
	 F_tol = 0.0001

	 P_new = P_old + a*gradF_P(Data,P_old,Wint,Wpen)
	 F_Pnew = F_P(Data,P_new,Wint,Wpen)
	 F_Pold = F_P(Data,P_old,Wint,Wpen)
	 DF = abs(F_Pnew-F_Pold)
	 while ((DF./abs(F_Pnew))[1] > F_tol)
	       P_new = P_old + a*gradF_P(Data,P_old,Wint,Wpen)
	       F_Pnew = F_P(Data,P_new,Wint,Wpen)
	       F_Pold = F_P(Data,P_old,Wint,Wpen)
	       DF = abs(F_Pnew-F_Pold)
         P_old = P_new
         #display(F_Pnew)
	 end
   #display(F_Pnew)
	 ProjMat = P_new


   e_vectors[:,:] = ProjMat

   close(s1)
   close(s2)
   close(s3)

   rm("/home/yannis/Desktop/LPDA/temp.bin")
   rm("/home/yannis/Desktop/LPDA/A.bin")
   rm("/home/yannis/Desktop/LPDA/B.bin")


end


