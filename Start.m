%% Machine Learning - Neural Networks Based on Distance Measures and Gaussian Activation functions

%% Initialization
clear ;
close all;
clc;

%% Parameters 
input_layer_size  = 28*28;
num_labels = 10;

path_to_MNIST = 'add the path to the MNIST data here';
 
layers = [input_layer_size, 100, 100, num_labels];

% initialize the weights of all layers
[weights] = InitializeWeights(layers);

%  Check gradients by running checkNNGradients
fprintf('\nChecking Backpropagation\n')
fflush(stdout);
checkGradients(0.0);
    
% Training distributed memory neural networks 
fprintf('\nTraining Neural Network\n')
fflush(stdout);

% Weight regularization parameter
lambda = 0.001; 
batch_size = 100;
alpha = 0.01;
decay = 0.01;

num_layers = length(layers) - 1;
momentum = cell(num_layers, 2);
magnitude = cell(num_layers, 2);

avg = 2.3;
num_epochs = 50;
for epoch = 1:num_epochs
  X = loadMNISTImages(strcat(path_to_MNIST,'train-images.idx3-ubyte'))';
  y_temp = loadMNISTLabels(strcat(path_to_MNIST,'train-labels.idx1-ubyte'))+1;
  
  m = size(X, 1);
  idx = sub2ind([m, num_labels], 1:m, y_temp');
  y = zeros(m, num_labels);
  y(idx) = 1;
  clear y_temp;

  if (epoch ~= 0)
    id = randperm(m);
    X = X(id,:);
    y = y(id,:);
  end

  for i = 0 : (floor(m/batch_size)-1)
    X_ = X(i*batch_size+1:i*batch_size+batch_size,:);
    y_ = y(i*batch_size+1:i*batch_size+batch_size,:);
    
    [J, grads] = costFunction(weights, X_, y_, lambda);    
    [weights, momentum, magnitude] = UpdateWeights(weights, grads, momentum, magnitude, alpha, mod(floor(i/2),2)); 
    
    avg = 0.99*avg + 0.01*J;
    disp([epoch, i, avg]);
    fflush(stdout);
  end  
  
  alpha = alpha * 1/(1 + decay * epoch)
  
  % check accuracy on test set
  X = loadMNISTImages(strcat(path_to_MNIST,'t10k-images.idx3-ubyte'))';
  y_temp = loadMNISTLabels(strcat(path_to_MNIST,'t10k-labels.idx1-ubyte'))+1;
  
  m = size(X, 1);
  idx = sub2ind([m, num_labels], 1:m, y_temp');
  y = zeros(m, num_labels);
  y(idx) = 1;
  clear y_temp;

  pred = predict(weights, X, y, 10000, false);
  
  clear X,y;
end
 
X = loadMNISTImages(strcat(path_to_MNIST,'t10k-images.idx3-ubyte'))';
y_temp = loadMNISTLabels(strcat(path_to_MNIST,'t10k-labels.idx1-ubyte'))+1;

m = size(X, 1);
idx = sub2ind([m, num_labels], 1:m, y_temp');
y = zeros(m, num_labels);
y(idx) = 1;
clear y_temp;

pred = predict(weights, X, y, 10000, false);