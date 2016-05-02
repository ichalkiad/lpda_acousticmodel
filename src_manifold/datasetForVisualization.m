%Size of visualization subset
size_of_subset = 3000;
%Vector containing labels to be included in visualization
vizlabels=labelsViz;
%Vector containing total dataset labels
labels=labelsPhHMM;
%Number of samples in each visualization class
class_samples = 1000;

%Initialization
data_x13=zeros(size_of_subset,13);
data_x117=zeros(size_of_subset,117);
labels_y=zeros(size_of_subset,1);
indices=[];

ind1=1;
ind2=class_samples;

for z=1:length(vizlabels)

	lab=vizlabels(z)

	labels_y(ind1:ind2)=lab;
        a=find(labels==lab);
        b=a(find(a>=5)); %starting point of samples, to be set according to how many neighboring frames we need
        c=b(find(b<=801143));%ending point of samples, to be set according to how many neighboring frames we need and the total number of samples in the dataset
        data_x13(ind1:ind2,:)=data(c(1:class_samples),:);
        indices = [indices; c(1:class_samples)];
        k=1;
        for i=ind1:ind2
		data_x117(i,1:13)=data(c(k)-4,:);
                data_x117(i,14:26)=data(c(k)-3,:);
                data_x117(i,27:39)=data(c(k)-2,:);
                data_x117(i,40:52)=data(c(k)-1,:);
                data_x117(i,53:65)=data(c(k),:);
                data_x117(i,66:78)=data(c(k)+1,:);
                data_x117(i,79:91)=data(c(k)+2,:);
                data_x117(i,92:104)=data(c(k)+3,:);
                data_x117(i,105:117)=data(c(k)+4,:);
                k = k + 1;
        end    

	ind1 = ind1 + class_samples;
        ind2 = ind2 + class_samples;

end

h5create('PhoneVisData13.h5','/data', [size_of_subset 13] );
h5write('PhoneVisData13.h5','/data', data_x13 );
h5create('PhoneVisLabels.h5','/labels', [size_of_subset 1] );
h5write('PhoneVisLabels.h5','/labels', labels_y );
save data117.mat data_x117 -v7.3
phones_priorsPhones5k=tabulate(labels_y);
a=find(phones_priorsPhones5k(:,2)==0);
phones_priorsPhones5k(a,:)=[];
save phones_priors_PhonesVis.mat phones_priorsPhones5k -v7.3

%Run getWint.jl,getWpen.jl,getNNneighbohoods_New.jl, lpda.m to get manifold matrices and projection graph

%Plot

x_lpda=lpda'*data_x117';
gscatter(x_lpda(1,:)',x_lpda(2,:)',labels_y);
scatter3(x_lpda(1,:)',x_lpda(2,:)',x_lpda(3,:)', 20);
