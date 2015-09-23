using HDF5,JLD

Wint = load("/media/data/ichalkia/dataset_mono2/WintLPDA14_mono2.jld")["Wint"]
Wpen = load("/media/data/ichalkia/dataset_mono2/WpenLPDA14_mono2.jld")["Wpen"]

display(size(Wint,2))

Wdiff = Wint-Wpen

Wij_diff = zeros(28,size(Wint,2))
neighbours = zeros(28,size(Wint,2))
for i=1:size(Wint,2)
    neighbours[:,i] = find(Wdiff[:,i].!=0)
    Wij_diff[:,i] = full(Wdiff[find(Wdiff[:,i].!=0),i])  
    display(i)
end
h5write("/media/data/ichalkia/dataset_mono2/neighbFULL_mono2.h5","neighbours",neighbours)
h5write("/media/data/ichalkia/dataset_mono2/Wij_diff_mono2.h5","Wij_diff",Wij_diff)
