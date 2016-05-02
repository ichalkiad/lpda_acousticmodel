using HDF5,JLD

#Set number of approx. nearset neighbors in each graph
kint=30
kpen=30

#Read in dataset and affinity matrices
mfcc_sampled = h5read("/media/data/ichalkia/knn13/mfcc13mono.h5", "data")
Wint = load("/media/data/ichalkia/knn13/WintLPDA30_phonesStates.jld")["Wint"]
Wpen = load("/media/data/ichalkia/knn13/WpenLPDA30_phonesStates.jld")["Wpen"]

display(size(Wint))
display(size(Wpen))
display(size(mfcc_sampled))


NNs = kint+kpen
Wdiff = Wint[1:801148,1:801148]-Wpen

#Output files containing neighborhoods of each sample (NN) and corresponding wewweights (NNw)
s = open("NNneighborhoodsLPDA30_phonesStates.bin", "w+")
NN = mmap_array(Float32, (NNs*size(mfcc_sampled,1),9*size(mfcc_sampled,2)), s)
p = open("NNweightsLPDA30_phonesStates.bin", "w+")
NNw = mmap_array(Float32, (size(mfcc_sampled,1)*NNs,1), p)


display(size(NN))
display(size(NNw))


Wij_diff = zeros(NNs*size(Wint,2))
neighbours = zeros(NNs)
m = 1
v = 1
q = 1 
less_NNs = 0
for i=1:size(Wpen,2)
    display(i)
#    print("Getting neighbours of sample $(i)\n")
    neighbours = find(Wdiff[:,i].!=0)
    if (length(neighbours)<kint+kpen)
        less_NNs = 1
    end

    for j=1:length(neighbours)
#    	print("Getting context of neighbor $(j),i.e. sample $(neighbours[j])\n")
        for k=neighbours[j]-4:neighbours[j]+4       
#	    print("Neighbor of $(neighbours[j]): $(k)\n")
	    if (k<1) || (k>801144)  ###No neighbours for samples 1-4 and last 4
		k = neighbours[j]
	    end
    	    for l=1:size(mfcc_sampled,2)
	    	NN[m,v] = mfcc_sampled[k,l]
		v = v + 1
	    end 
	end
#	display(NN[m,:])
	m = m + 1
	v = 1
#	print("Weight of neighbor $(neighbours[j]) is $(Wdiff[neighbours[j],i])\n")
	NNw[q] = Wdiff[neighbours[j],i]
	q = q + 1
    end
    if (less_NNs==1)
	for z=1:kint+kpen-length(neighbours)
	    NN[m,:] = NN[m-1,:]
	    NNw[q] = 0
	    m = m + 1
	    q = q + 1
	end
    end
    less_NNs = 0
end
display(p)
display(s)

write(s,NN)
close(s)
write(p,NNw)
close(p)
