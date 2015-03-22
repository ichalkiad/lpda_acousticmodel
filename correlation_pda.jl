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


function cpda{T,Q}(Data::AbstractMatrix{T},Labels::AbstractMatrix{Q},dims::Int64)

	 m,n = size(Data)
	 println("Give kint, kpen, Rint, Rpen:")
	 kint = int(readline(STDIN))
	 kpen = int(readline(STDIN))
	 Rint = float(readline(STDIN))
	 Rpen = float(readline(STDIN))

   sa2 = sum(Data.^2, 1)
   #projection on unit hypersphere
   for j = 1 : n
       Data[:,j] = Data[:,j]./sqrt(sa2[j])
   end

	 #compute intrinsic and penalty matrices
	 (Wint,Wpen) = compute_weight_mats(Data,Labels,kint,kpen,Rint,Rpen,"correlation")

	 #formulate generalized eigen problem
	 D_diag = sum(Wint,1)
	 Di = diagm(vec(D_diag))

	 D_diag = sum(Wpen,1)
	 Dp = diagm(vec(D_diag))

	 temp = Array(Float64,(m,n))
	 A = Array(Float64,(m,m))
	 A_mul_B!(temp,Data,Dp-Wpen)
	 A_mul_Bt!(A,temp,Data)

	 B = Array(Float64,(m,m))
	 A_mul_B!(temp,Data,Di-Wint)
	 A_mul_Bt!(B,temp,Data)

	 #solve to get an initial estimate of projection matrix P
	 P = eigfact!(A,B)
	 sorted_idx = sortperm(vec(P[:values]),rev=true)

	 (eigvals,eigvecs) =  (P[:values][sorted_idx[1:dims]],P[:vectors][:,sorted_idx[1:dims]])

	 P_init = eigvecs

	 #gradient descent to minimize F_P
	 P_old = P_init

	 #learning rate
	 a =  0.01
	 #tolerance in F_P increase
	 F_tol = 0.000001

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
         display(F_Pnew)
	 end
	 ProjMat = P_new

   return eigvals,ProjMat

end



#dims=20
#X = rand(100,500)
#C = Array(Int64,(1,500))
#C[1:100] = 1
#C[101:250] = 2
#C[251:300] = 1
#C[301:end] = 2
#kint = 100
#kpen = 100
#Rint = 1.0
#Rpen = 1.0
#dist = "correlation"
#Data=X
#Labels=C
#P = eigvecs
#(Wint,Wpen) = compute_weight_mats(X,C,kint,kpen,Rint,Rpen,dist)
#(eigval,eigvec) = cpda(Data,Labels,dims)
#f_uv(eigvec,eigvec[:,1],eigvec[:,2])
#gradF_P(X,eigvec,Wint,Wpen)
