using HDF5,JLD,MATLAB

triphones_labels = load("/media/data/ichalkia/dataset_mono2/phones_priors_mono2.mat")["phones_priors"]
#P = real(load("/media/data/ichalkia/dataset_mono_correct/P_LPDA_mono_rand.jld")["P"])
mfcc_sampled = h5read("/media/data/ichalkia/dataset_mono2/mono2_117D/mfcc_train117D.h5", "mfcc_sampled")
#labels_sampled = h5read("/media/data/ichalkia/dataset_mono2/labels_train.h5", "labels_sampled")
data = mfcc_sampled
(m,n) = size(data)

display((m,n))

neighbours = int(h5read("/media/data/ichalkia/dataset_mono2/neighbFULL_mono2.h5", "neighbours"))
display(size(neighbours))


data_neighb = zeros(size(mfcc_sampled,1),28*size(mfcc_sampled,2))
#labels_neighb = int(zeros(1,28*size(mfcc_sampled,2)))

display(size(data_neighb))
#display(size(labels_neighb))

k = 1
for i = 1:n
display(i)
  ns = neighbours[:,i]
  data_neighb[:,k:k+27] = mfcc_sampled[:,ns]
  #labels_neighb[find(triphones_labels[:,1].==labels_sampled[i])[1],k] = 1
#  for j = 1:length(ns)
#      labels_neighb[k-1+j] = find(triphones_labels[:,1].==labels_sampled[ns[j]])[1]
#  end
  k = k + 28

end

h5write("/media/data/ichalkia/dataset_mono2/mono2_117D/data_neighborhoods_117D.h5","data",data_neighb)
#h5write("/media/data/ichalkia/dataset_mono2/labels_neighborhoods_mono2.h5","label",labels_neighb)
