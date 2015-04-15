#MatLab optimization
using MATLAB
using MAT
using HDF5,JLD

(m,n) = load("/home/yannis/Desktop/LPDA/mfcc_si84_half_clean.txt_enerdims.jld")["sv_dims"]

l = open("/home/yannis/Desktop/LPDA/frameLabels.txt.bin")
labels = mmap_array(Float64, (n,1), l)
lab = vec(labels[1:5000])

s = open("/home/yannis/Desktop/LPDA/mfcc_si84_half_clean.txt_ener.bin")
mfcc = mmap_array(Float64, (m,n), s)

n=5000
Data = mfcc[:,1:5000]
dims = 3
e_values = zeros(dims)
e_vectors = Array(Float64, (m,dims))

#Wint = speye(n,n)
#Wpen = speye(n,n)

#compute intrinsic and penalty matrices
#compute_weight_mats!(Wint,Wpen,mfcc[:,1:5000], lab, 100,100, 1000.0, 3000.0, "correlation")


#for j = 1 : n
  # @inbounds Wint[j,j] = 1
 #  @inbounds Wpen[j,j] = 1
#end

Wint = load("/home/yannis/Desktop/LPDA/WintCor.bin")["Wint"]
Wpen = load("/home/yannis/Desktop/LPDA/WpenCor.bin")["Wpen"]

#formulate generalized eigen problem
D_diag = sum(Wint,1)
Di = diagm(vec(D_diag))

D_diag = sum(Wpen,1)
Dp = diagm(vec(D_diag))

s1 = open("/home/yannis/Desktop/LPDA/temp.bin","w+")
temp = mmap_array(Float64, (m,n), s1)

s2 = open("/home/yannis/Desktop/LPDA/A.bin","w+")
A = mmap_array(Float64, (m,m), s2)

s3 = open("/home/yannis/Desktop/LPDA/B.bin","w+")
B = mmap_array(Float64, (m,m), s3)

display("Mmapped arrays")

A_mul_B!(temp,Data,Dp-Wpen)
A_mul_Bt!(A,temp,Data)

A_mul_B!(temp,Data,Di-Wint)
A_mul_Bt!(B,temp,Data)

#solve to get an initial estimate of projection matrix P
P = eigfact!(A,B)

rm("/home/yannis/Desktop/LPDA/temp.bin")
rm("/home/yannis/Desktop/LPDA/A.bin")
rm("/home/yannis/Desktop/LPDA/B.bin")

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

display(real_evals)
display(k-1)
#readline(STDIN)

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
e_values[:] = eigen_v[1:min(k-1,dims)]

#keep the corresponding eigenvectors
e_vectors[:,:] = P[:vectors][:,sorted_idx[1:dims]]

P_init = e_vectors

file = matopen("minimizef.mat", "w")
write(file, "Data", mfcc[:,1:n])
write(file,"Wint",Wint)
write(file,"Wpen",Wpen)
write(file,"P0",P_init)
close(file)

session = MSession()
eval_string(session,"[projection_mat,fval,exitflag,output_info] = optimize_f()")

proj_mx = get_mvariable(session, :projection_mat)
fval_mx = get_mvariable(session, :fval)
exitflag_mx = get_mvariable(session,:exitflag)
output_mx = get_mvariable(session,:output_info)

proj = jmatrix(proj_mx)
fval_FP = jscalar(fval_mx)
status = jscalar(exitflag_mx)
output = jvariable(output_mx)

y = proj'*mfcc[:,1:5000]
save("/home/yannis/Desktop/LPDA/YcpdaMATLAB.bin","y",y)

file2 = matopen("results.mat", "w")
write(file2,"P", proj)
write(file2,"fval",fval_FP)
write(file2,"status",status)
write(file2,"output",output)
close(file2)

close(l)
close(s)

close(session)

