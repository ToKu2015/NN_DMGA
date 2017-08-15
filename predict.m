function [p] = predict(weights, X, y, number, train = true)
  X = X(1:number,:);
  y = y(1:number,:);

  m = size(X, 1);
  input_layer_size = size(weights{1,1},2);
  num_labels = size(weights{end,1},1);
  num_layers = length(weights);

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
  [~, pred] = max(a_s{num_layers+1}, [], 2);
  [~,y] = max(y, [], 2);
  
  if train == true
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
  else
    fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y)) * 100);
  end
  fflush(stdout);
  
  p = a_s{num_layers+1};
end