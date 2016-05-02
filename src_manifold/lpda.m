%Quick run Locality Preserving Discriminant Analysis for visualization purposes
%Assume wpen,wint and 117-d dataset (data_x) is already in workspace

D_p=sum(wpen,2);D_p=diag(D_p);A=data_x'*(D_p-wpen)*data_x;
D_i=sum(wint,2);D_i=diag(D_i);B=data_x'*(D_i-wint)*data_x;
[V,D]=eig(A,B);

%Select 3 first real eigenvectors (e.g. 1,2,3) and save in lpda matrix
lpda = V(:,1:3);
save lpda.mat lpda -v7.3
