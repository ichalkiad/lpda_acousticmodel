using HDF5,JLD,MAT,MATLAB

(m,n) = load("/home/yannis/Desktop/LPDA/mfcc_si84_half_clean.txt_enerdims.jld")["sv_dims"]

l = open("/home/yannis/Desktop/LPDA/frameLabels.txt.bin")
labels = mmap_array(Float64, (n,1), l)
lab = vec(labels[1:5000])

#file = matopen("/home/yannis/Desktop/LPDA/phones_.mat", "w")
#write(file, "phones", labels)
#close(file)

s = open("/home/yannis/Desktop/LPDA/mfcc_si84_half_clean.txt_ener.bin")
mfcc = mmap_array(Float64, (m,n), s)

n=5000

Wint = speye(n,n)
Wpen = speye(n,n)

#compute_weight_mats!(Wint,Wpen,mfcc, labels[:], 200,200, 1000.0, 3000.0, "locality")

compute_weight_mats!(Wint,Wpen,mfcc[:,1:5000], lab, 100,100, 1000.0, 3000.0, "correlation")

for j = 1 : n
   @inbounds Wint[j,j] = 1
   @inbounds Wpen[j,j] = 1
end


save("/home/yannis/Desktop/LPDA/WintCor.bin","Wint",Wint)
save("/home/yannis/Desktop/LPDA/WpenCor.bin","Wpen",Wpen)

#Wint = load("/home/yannis/Desktop/LPDA/Wint.bin")["Wint"]
#Wpen = load("/home/yannis/Desktop/LPDA/Wpen.bin")["Wpen"]

dims = 3

e_values = zeros(dims)
e_vectors = Array(Float64, (m,dims))

#########
##LPDA##
########

#display("go to lpda")
#lpda!(e_values,e_vectors,mfcc[:,1:10000],lab,Wint,Wpen,dims)

########

########
##CPDA##
########

display("go to cpda")
cpda!(e_values,e_vectors,mfcc[:,1:5000],lab,Wint,Wpen,dims)

########


display("project")
#readline(STDIN)

y = e_vectors'*mfcc[:,1:5000]
save("/home/yannis/Desktop/LPDA/Ycpda.bin","y",y)

close(l)
close(s)

