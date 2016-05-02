%Print neighborhood of each sample and pipe to pfile_create (float-compatible version) to get manifold pfile

%Read in neighborhoods and corresponding weights
NN=open("/media/data/ichalkia/knn13/NNneighborhoodsLPDA30_phonesStates.bin","r")
NNw=open("/media/data/ichalkia/knn13/NNweightsLPDA30_phonesStates.bin","r")

k=0
l=0

%Define total number of neighbors
total_NNs = 60

for i=1:801148*total_NNs % number_of_samples*total_NNs

    features = read(NN,Float32,(1,117))
    weight = read(NNw,Float32,1)

    print(k)
    print(" ")
    print(l)
    for j=1:117
	print(" ")
	print(features[1,j])
    end
    print(" ")
    print(weight[1])
    print("\n")
   
    %Random creation of sentences/frames for the pfile
    l = l + 1
    if (l==total_NNs)
   	l = 0
  	k = k + 1
    end
    
end

close(NN)
close(NNw)




