function numgrad = computeNumericalGradient(J, weights)
  % Computes the gradient using "finite differences" 
  % and gives us a numerical estimate of the gradient.

  num_layers = length(weights)
  numgrad = cell(num_layers, 2);
  for l = 1:num_layers
    numgrad{l,1} = zeros(size(weights{l,1}));
    numgrad{l,2} = zeros(size(weights{l,1})); 
  end

  e = 1e-4;
  for l = 1:num_layers
    numRows = size(weights{l,1},1);
    numCols = size(weights{l,1},2);
    
    for row = 1:numRows
      for col = 1:numCols    
        for s = 1:2
          save = weights{l,s}(row, col);
          % set perturbation vector
          weights{l,s}(row, col) = save - e;
          loss1 = J(weights);
          
          % set perturbation vector
          weights{l,s}(row, col) = save + e;
          loss2 = J(weights);         
        
          weights{l,s}(row, col) = save;
        
          % Compute Numerical Gradient
          numgrad{l,s}(row, col) = (loss2 - loss1) / (2*e);
        end      
      end
    end
  end 
end