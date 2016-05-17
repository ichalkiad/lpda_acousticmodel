%Read in labels of training set
a=unique(labelsHMM,'rows');
frameLabels=labelsHMM;
labels = zeros(length(frameLabels),1);

%Read in labels of validation set
frameLabelsV = labelsValid;
labelsV = zeros(length(frameLabelsV),1);

for i=1:length(a)
    
    ind1 = find(frameLabels(:,1)==a(i,1)); 
    ind2 = find(frameLabels(:,2)==a(i,2));
    ind = intersect(ind1,ind2);
    labels(ind) = i;
    
    ind1 = find(frameLabelsV(:,1)==a(i,1)); 
    ind2 = find(frameLabelsV(:,2)==a(i,2));
    ind = intersect(ind1,ind2);
    labelsV(ind) = i;
    
end
% h5create('/home/yannis/Desktop/labelsPhoneState.h5','/labels',[801148 1]);
% h5write('/home/yannis/Desktop/labelsPhoneState.h5','/labels',labels);
