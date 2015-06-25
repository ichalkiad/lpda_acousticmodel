using MAT,MATLAB,HDF5,JLD,Clustering,KDTrees


Wint = load("/home/yannis/Desktop/Wint.jld")["Wint"]
Wpen = load("/home/yannis/Desktop/Wpen.jld")["Wpen"]
P = real(load("/home/yannis/Desktop/P.jld")["P"])


mfcc_sampled = h5read("/home/yannis/Desktop/KALDI_norm_var/mfcc_sampled1M_norm_mvK.h5", "sampled_mfcc")
labels_sampled = h5read("/home/yannis/Desktop/KALDI_norm_var/labels_sampled1M_norm_mvK.h5", "sampled_labels")

Data = mfcc_sampled
(m,n) = size(Data)


   #formulate generalized eigen problem
	 D_diag = sum(Wint,2)
	 Di = sparse(vec([1:900000]),vec([1:900000]),vec(D_diag))

	 D_diag = sum(Wpen,2)
   Dp = sparse(vec([1:900000]),vec([1:900000]),vec(D_diag))

   A = Data*(Dp-Wpen)*Data'
   B = Data*(Di-Wint)*Data'

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
   e_values = eigen_v[1:min(k-1,dims)]

   #keep the corresponding eigenvectors
   e_vectors = real(P[:vectors][:,sorted_idx[1:dims]])

   display("Got eigvecs")






