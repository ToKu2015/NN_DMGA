function [J grads] = costFunction(weights, X, y, lambda)
  m = size(X, 1);
  input_layer_size = size(weights{1,1},2);
  num_labels = size(weights{end,1},1);
  num_layers = length(weights);

  J = 0;
  grads = cell(num_layers, 2);
  for l = 1:num_layers
    grads{l,1} = zeros(size(weights{l,1}));
    grads{l,2} = zeros(size(weights{l,1}));
  end 

  % Feedforward the neural network and return the cost in the variable J.
  a_s = cell(num_layers+1,1);
  a_s{1} = X;
  clear X;

  for l = 1:num_layers
    temp = a_s{l}.^2*(weights{l,2}.^2)'-2*a_s{l}*(weights{l,2}.^2.*weights{l,1})';
    temp = bsxfun(@plus, temp, sum(weights{l,2}.^2.*weights{l,1}.^2,2)');    
    
    if (l == num_layers)
        a_s{l+1} = -0.5*temp; %last layer
    else
        a_s{l+1} = exp(-0.5*temp);
    end
  end
  
  % softmax
  max_ = max(a_s{num_layers+1}, [], 2);
  a_s{num_layers+1} = exp(bsxfun(@minus, a_s{num_layers+1}, max_));
  a_s{num_layers+1} = bsxfun(@rdivide, a_s{num_layers+1}, sum(a_s{num_layers+1}, 2));

  % cost function
  idx_true = find(y == 1);
  sum_aux = sum(log(a_s{num_layers+1}(idx_true)));

  J = (-1/m)*sum_aux;

  % backpropagation algorithm to compute the gradients of all layers
  delta_s = cell(num_layers+1,1);
  delta_s{num_layers+1} = a_s{num_layers+1} - y;   
  
  for l = num_layers:-1:1
    if (l == num_layers)
      delta_times_a = delta_s{l+1}; % last layer
    else
      delta_times_a = delta_s{l+1}.*a_s{l+1};
    end
    
    Theta_square = weights{l,2}.^2;
    Theta_square_times_Centroid = Theta_square.*weights{l,1};  
    delta_s{l} = -a_s{l}.*(delta_times_a*Theta_square)+delta_times_a*Theta_square_times_Centroid;
   
    aux1 = (1/m)*(delta_times_a'*a_s{l});
    aux2 = mean(delta_times_a, 1)';
    grads{l,1} = aux1.*Theta_square - bsxfun(@times, Theta_square_times_Centroid, aux2);

    grads{l,2} = -(1/m)*(delta_times_a'*a_s{l}.^2).*weights{l,2} + 2*aux1.*weights{l,2}.*weights{l,1};
    grads{l,2} = grads{l,2} - bsxfun(@times, weights{l,2}.*weights{l,1}.^2, aux2); 
  end

  % regularization with the cost function and gradients
  for l = 1:num_layers
    J = J + lambda/(2*m)*sum(sum(weights{l,2}.^2)) + lambda/(2*m)*sum(sum(weights{l,2}.^2));
    grads{l,1} = grads{l,1} + lambda/m*weights{l,1};
    grads{l,2} = grads{l,2} + lambda/m*weights{l,2};
  end

end