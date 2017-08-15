function [weights, momentum, magnitude] = UpdateWeights(weights, grads, momentum, magnitude, alpha, sw)
  num_layers = length(weights);
  
  empty = true;
  for l = 1:num_layers
    if (length(momentum{l,1})~=0 || length(momentum{l,2})~=0)
      empty = false;
      break;
    end
  end
  
  if (empty == true)    
    for l = 1:num_layers
      momentum{l,1} = grads{l,1};
      momentum{l,2} = grads{l,2};
      
      magnitude{l,1} = sqrt(grads{l,1}.^2);
      magnitude{l,2} = sqrt(grads{l,2}.^2); 
    end    
  else
    for l = 1:num_layers  
    
      % RMSProp
      magnitude{l,1} = sqrt(0.7*magnitude{l,1}.^2 + 0.3*grads{l,1}.^2);
      magnitude{l,2} = sqrt(0.7*magnitude{l,2}.^2 + 0.3*grads{l,2}.^2); 
  
      grads{l,1} = bsxfun(@rdivide, grads{l,1}, magnitude{l,1}+10^-9); 
      grads{l,2} = bsxfun(@rdivide, grads{l,2}, magnitude{l,2}+10^-9);
      
      % Momentum
      momentum{l,1} = 0.9*momentum{l,1} + 0.1*grads{l,1};
      momentum{l,2} = 0.9*momentum{l,2} + 0.1*grads{l,2}; 
    end      
  end

  for l = 1:num_layers
      weights{l,2} = weights{l,2} - alpha*momentum{l,2};
      weights{l,1} = weights{l,1} - alpha*momentum{l,1};
  end
endfunction