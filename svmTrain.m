function [model] = svmTrain(X, Y, C,kernelFunction, ...
                            tol, max_passes)

% Data parameters
m = size(X, 1);
n = size(X, 2);

% Map 0 to -1
Y(Y==0) = -1;

% Variables
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

%Precompute kernel matrix
if strcmp(func2str(kernelFunction), 'linearKernel')
    K = X*X';
elseif strfind(func2str(kernelFunction), 'gaussianKernel')
    X2 = sum(X.^2, 2);
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
    K = kernelFunction(1, 0) .^ K;
end


% Train
fprintf('\nTraining ...');
dots = 12;
while passes < max_passes,
            
    num_changed_alphas = 0;
    for i = 1:m,
        
        E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
        
        if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0)),
            
            j = ceil(m * rand());
            while j == i,
                j = ceil(m * rand());
            end

            E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            if (Y(i) == Y(j)),
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
           
            if (L == H),
                continue;
            end
            
            
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0),
                continue;
            end
            
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            
            if (abs(alphas(j) - alpha_j_old) < tol),
                alphas(j) = alpha_j_old;
                continue;
            end
            
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';
             
            if (0 < alphas(i) && alphas(i) < C),
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C),
                b = b2;
            else
                b = (b1+b2)/2;
            end
            num_changed_alphas = num_changed_alphas + 1;
        end
        
    end
    
    if (num_changed_alphas == 0),
        passes = passes + 1;
    else
        passes = 0;
    end

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    
    drawnow('update');
end
fprintf(' Done! \n\n');

% Save the model
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = kernelFunction;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';

end
