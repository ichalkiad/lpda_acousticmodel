function lpda{T,Q}(Data::AbstractMatrix{T},Labels::AbstractMatrix{Q},dims::Int64)

	 m,n = size(Data)
	 println("Give kint, kpen, Rint, Rpen:")
	 kint = int(readline(STDIN))
	 kpen = int(readline(STDIN))
	 Rint = float(readline(STDIN))
	 Rpen = float(readline(STDIN))

	 #compute intrinsic and penalty matrices
	 (Wint,Wpen) = compute_weight_mats(Data,Labels,kint,kpen,Rint,Rpen,"locality")

	 #formulate generalized eigen problem
	 D_diag = sum(Wint,2)
	 Di = diagm(vec(D_diag))

	 D_diag = sum(Wpen,2)
	 Dp = diagm(vec(D_diag))

	 temp = Array(Float64,(m,n))
	 A = Array(Float64,(m,m))
	 A_mul_B!(temp,Data,Dp-Wpen)
	 A_mul_Bt!(A,temp,Data)

	 B = Array(Float64,(m,m))
	 A_mul_B!(temp,Data,Di-Wint)
	 A_mul_Bt!(B,temp,Data)

	 #solve generalized eigen problem, contains eigenvalues and eigenvectors
	 P = eigfact!(A,B)
	 sorted_idx = sortperm(vec(P[:values]),rev=true)

	 return (P[:values][sorted_idx[1:dims]],P[:vectors][:,sorted_idx[1:dims]])

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
#dist = "locality"
#Data=X
#Labels=C

#(eigval,eigvec) = lpda(Data,Labels,dims)
#(Wint,Wpen) = compute_weight_mats(X,C,kint,kpen,Rint,Rpen,dist)
