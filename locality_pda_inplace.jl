function lpda!{T,Q}(e_values::AbstractVector{T},e_vectors::AbstractMatrix{T},Data::AbstractMatrix{T},Labels::AbstractVector{Q},Wint::SparseMatrixCSC{Float64,Int64},Wpen::SparseMatrixCSC{Float64,Int64},dims::Int64)

	 m,n = size(Data)

   #formulate generalized eigen problem
	 D_diag = sum(Wint,2)
	 Di = diagm(vec(D_diag))

	 D_diag = sum(Wpen,2)
	 Dp = diagm(vec(D_diag))

   s1 = open("/home/yannis/Desktop/LPDA/temp.bin","w+")
   temp = mmap_array(Float64, (m,n), s1)

   s2 = open("/home/yannis/Desktop/LPDA/A.bin","w+")
   A = mmap_array(Float64, (m,m), s2)

   s3 = open("/home/yannis/Desktop/LPDA/B.bin","w+")
   B = mmap_array(Float64, (m,m), s3)

   display("Mmapped arrays")

   #temp = Array(Float64,(m,n))
	 #A = Array(Float64,(m,m))
	 A_mul_B!(temp,Data,Dp-Wpen)
   write(s1,temp)
	 A_mul_Bt!(A,temp,Data)
   write(s2,A)

	 #B = Array(Float64,(m,m))
	 A_mul_B!(temp,Data,Di-Wint)
   write(s1,temp)
	 A_mul_Bt!(B,temp,Data)
   write(s3,B)

	 #solve generalized eigen problem, P contains eigenvalues and eigenvectors
	 P = eigfact!(A,B)

   display("Got eigs")

   #keep desired number of real eigenvalues
   evals = P[:values]
   real_evals = Array((Float64,Int64),length(evals))

   k = 1
   for i = 1 : length(evals)
       (re,im) = (real(evals[i]),imag(evals[i]))
       if im == 0
          real_evals[i] = (re,i)
          k = k + 1
       end
   end

   sorted_evals = zeros(k-1)
   sorted_evals = sort(real_evals[1:k-1],by=x->x[1],rev=true)

   sorted_idx = zeros(k-1)
   for i = 1 : length(sorted_evals)
       sorted_idx[i] = sorted_evals[i][2]
   end

   #keep the corresponding eigenvectors
   e_vectors[:,:] = P[:vectors][:,sorted_idx[1:dims]]

   display("Got eigvecs")

   close(s1)
   close(s2)
   close(s3)

   rm("/home/yannis/Desktop/LPDA/temp.bin")
   rm("/home/yannis/Desktop/LPDA/A.bin")
   rm("/home/yannis/Desktop/LPDA/B.bin")

end


