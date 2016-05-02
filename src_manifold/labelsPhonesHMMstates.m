%Map pairs of (phone,HMMstate) to labels

%Read labels.txt , which contains (phone,HMMstate) pairs
frameLabels = importdata('labels.txt');
a=unique(frameLabels,'rows');

labels = zeros(length(a),1);
for i=1:length(a)
    ind1 = find(frameLabels(:,1)==a(i,1)); 
    ind2 = find(frameLabels(:,2)==a(i,2));
    ind = intersect(ind1,ind2);
    labels(ind) = i;
end

%Export labels in hdf5 format
h5create('labelsPhoneState.h5','/labels',[size(labels,1) 1]);
h5write('labelsPhoneState.h5','/labels',labels);
