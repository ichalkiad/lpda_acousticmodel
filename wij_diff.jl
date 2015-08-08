using HDF5,JLD

Wint = load("/home/ichalkia/WintLPDA14.jld")["Wint"]
Wpen = load("/home/ichalkia/WpenLPDA14.jld")["Wpen"]

Wdiff = Wint-Wpen

Wij_diff = zeros(28,900000)
neighbours = zeros(28,900000)
for i=1:900000
    neighbours[:,i] = find(Wdiff[:,i].!=0)
    Wij_diff[:,i] = full(Wdiff[find(Wdiff[:,i].!=0),i])  
    display(i)
end
h5write("/home/ichalkia/neighb.h5","neighbours",neighbours)
h5write("/home/ichalkia/Wij_diff.h5","Wij_diff",Wij_diff)
