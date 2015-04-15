function [ f_p, grad_f ] = F_P( P )
    
found = 0;
work_cont = whos();
for i = 1:numel(work_cont)
  if strcmp(work_cont(i,1).name,'P0')
      found = found + 1;
  elseif strcmp(work_cont(i,1).name,'Data')
      found = found + 1;
  elseif strcmp(work_cont(i,1).name,'Wint')
      found = found + 1;
  elseif strcmp(work_cont(i,1).name,'Wpen')
      found = found + 1;
  end      
end

if (found~=4)
    load('minimizef.mat');
end

X = Data;
[m,n] = size(X);

fp = 0.0;
i = 1;
j = 1;
while i <= n
      while j <= n
      	    if (i~=j)
                fp = fp + (1-X(:,i)'*P*P'*X(:,j))/(sqrt(X(:,i)'*P*P'*X(:,i))*sqrt(X(:,j)'*P*P'*X(:,j)))*(Wpen(i,j)-Wint(i,j));
            end
            j = j + 1;
      end
      i = i + 1;
end
f_p = -2*fp;


gradfp = 0.0;
i = 1;
j = 1;
while i <= n
      while j <= n
      	    if i~=j
                s1 = (X(:,i)'*P*P'*X(:,j)).*(X(:,i)*X(:,i)')./((sqrt(X(:,i)'*P*P'*X(:,i)).^3).*sqrt(X(:,j)'*P*P'*X(:,j)));
                s2 = (X(:,i)'*P*P'*X(:,j)).*(X(:,j)*X(:,j)')./(sqrt(X(:,i)'*P*P'*X(:,i)).*(sqrt(X(:,j)'*P*P'*X(:,j)).^3));
                s3 = (X(:,i)*X(:,j)' + X(:,j)*X(:,i)')./(sqrt(X(:,i)'*P*P'*X(:,i)).*sqrt(X(:,j)'*P*P'*X(:,j)));
                gradfp = gradfp + (s1 + s2 - s3)*P*(Wpen(i,j)-Wint(i,j));
            end
            j = j + 1;
      end
      i = i + 1;
end
grad_f = -2*gradfp;


end

