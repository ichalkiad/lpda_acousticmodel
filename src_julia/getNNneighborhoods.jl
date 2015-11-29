using HDF5,JLD

kint=14
kpen=14

mfcc_sampled = h5read("/media/data/ichalkia/knn13/mfcc13mono.h5", "data")
Wint = load("/media/data/ichalkia/knn13/WintLPDA14mono.jld")["Wint"]
Wpen = load("/media/data/ichalkia/knn13/WpenLPDA14mono.jld")["Wpen"]

display(size(Wint))
display(size(mfcc_sampled))


NNs = kint+kpen
Wdiff = Wint-Wpen

#s = open("NNneighborhoods.bin", "w+")
#NN = mmap_array(Float64, (NNs*size(mfcc_sampled,1),9*size(mfcc_sampled,2)), s)
#p = open("NNweights.bin", "w+")
#NNw = mmap_array(Float64, (size(mfcc_sampled,1)*NNs,1), p)


#display(size(NN))
#display(size(NNw))


Wij_diff = zeros(NNs*size(Wint,2))
neighbours = zeros(NNs)
m = 1
v = 1
q = 1 
for i=1:size(Wint,2)
#    display(i)
#    print("Getting neighbours of sample $(i)\n")
    neighbours = find(Wdiff[:,i].!=0)
    for j=1:length(neighbours)
#    	print("Getting context of neighbor $(j),i.e. sample $(neighbours[j])\n")
#        for k=neighbours[j]-4:neighbours[j]+4
#	    print("Neighbor of $(neighbours[j]): $(k)\n")
#    	    for l=1:size(mfcc_sampled,2)
#	    	NN[m,v] = mfcc_sampled[k,l]
#		v = v + 1
#	    end 
#	end
#	display(NN[m,:])
#	m = m + 1
#	v = 1
#	print("Weight of neighbor $(neighbours[j]) is $(Wdiff[neighbours[j],i])\n")
#	NNw[q] = Wdiff[neighbours[j],i]
	q = q + 1
    end
end
display(p)

#write(s,NN)
#close(s)
#write(p,NNw)
#close(p)
