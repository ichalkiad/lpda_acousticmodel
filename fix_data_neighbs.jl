using HDF5,JLD,MATLAB

triphones_labels = load("/home/ichalkia/phones_priors.mat")["phones_priors"]
P = real(load("/home/ichalkia/P_LPDA.jld")["P"])
mfcc_sampled = h5read("/home/ichalkia/mfcc_sampled1M_norm_mvK.h5", "sampled_mfcc")
labels_sampled = h5read("/home/ichalkia/labels_sampled1M_norm_mvK.h5", "sampled_labels")
data = mfcc_sampled
(m,n) = size(data)

neighbours = int(h5read("/home/ichalkia/neighb.h5", "neighbours"))

X = P'*data;
X = X .+ minimum(X)
X = (X .-minimum(X)) / (maximum(X) - (minimum(X)))

data_neighb = zeros(size(P,2),28*size(mfcc_sampled,2))
labels_neighb = int(zeros(1,28*size(mfcc_sampled,2)))


k = 1
for i = 1:n
display(i)
  ns = neighbours[:,i]
  data_neighb[:,k:k+27] = X[:,ns]
  #labels_neighb[find(triphones_labels[:,1].==labels_sampled[i])[1],k] = 1
  for j = 1:length(ns)
      labels_neighb[k-1+j] = find(triphones_labels[:,1].==labels_sampled[ns[j]])[1]
  end
  k = k + 28

end

h5write("/home/ichalkia/data_neighbNEW.h5","data",data_neighb)
h5write("/home/ichalkia/labels_neighbNEW.h5","label",labels_neighb)
