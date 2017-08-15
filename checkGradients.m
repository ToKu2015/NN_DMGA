function checkGradients(lambda)
  if ~exist('lambda', 'var') || isempty(lambda)
      lambda = 0;
  end

  input_layer_size = 3;
  num_labels = 3;
  m = 50;

  layers = [input_layer_size, 10, 10, 8, 4, 3, num_labels];

  num_layers = length(layers) - 1;
  weights = cell(num_layers, 2);
  for l = 1:num_layers
    L_in = layers(l);
    L_out = layers(l+1);
    
    weights{l,1} = 0.5 + 0.01*randn(L_out, L_in);
    weights{l,2} = 1.0 + 0.01*randn(L_out, L_in);
  end

  X = zeros(m, input_layer_size);
  X = reshape(sin(1:numel(X)), size(X)) / sqrt(input_layer_size);
  y_temp  = 1 + mod(1:m, num_labels)';

  idx = sub2ind([m, num_labels], 1:m, y_temp');
  y = zeros(m, num_labels);
  y(idx) = 1;

  % Handle for cost function
  costFunc = @(p) costFunction(p, X, y, lambda);
                    
  [cost, grad] = costFunc(weights);
  numgrad = computeNumericalGradient(costFunc, weights);

  numgrad_ = [];
  grad_ = [];
  for l = 1:num_layers
    numgrad_ = [numgrad_; numgrad{l,1}(:); numgrad{l,2}(:)];
    grad_ = [grad_; grad{l,1}(:); grad{l,2}(:)];
  end

  diff = norm(numgrad_-grad_)/norm(numgrad_+grad_);

  fprintf(['Relative Difference: %g\n'], diff);
end