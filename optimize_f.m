function [projection_mat,fval,exitflag,output_info] = optimize_f() 

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

objective = 'objective';
x_init = 'x0';
solver_alg = 'solver';
options = 'options';

objective_func = @cost_func
x0 = P0
solver = 'fminunc'
%options_list = optimoptions(solver);
options_list = optimoptions('fminunc','Algorithm','trust-region','Diagnostics','on','FunValCheck','on','GradObj','on');

optim_problem = struct(objective,objective_func,x_init,x0,solver_alg,solver,options,options_list);

[projection_mat,fval,exitflag,output_info] = fminunc(optim_problem)


end