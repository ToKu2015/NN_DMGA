function [weights] = InitializeWeights(layers)
  fprintf('\nInitializing Neural Network Parameters\n')
  fflush(stdout);

  num_layers = length(layers) - 1
  weights = cell(num_layers, 2);
  numParameters = 0;
  for l = 1:num_layers
    L_in = layers(l);
    L_out = layers(l+1);
    
    weights{l,1} = 0.5+0.3*rand(L_out, L_in);
    factor = sqrt(2*log(1/0.1)/L_in); 
    weights{l,2} = factor*ones(L_out, L_in)+0.4*factor*randn(L_out, L_in); 
    
    numParameters = numParameters + 2*L_in*L_out;
  end
  
  numParameters
endfunction