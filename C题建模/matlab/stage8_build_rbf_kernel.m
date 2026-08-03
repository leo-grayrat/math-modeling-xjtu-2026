function [kernel, info, processed] = stage8_build_rbf_kernel( ...
    features, trainingRows, options)
%STAGE8_BUILD_RBF_KERNEL 只用训练对象确定预处理参数和 RBF 核宽度。

arguments
    features double
    trainingRows (:, 1) logical
    options.Standardize (1, 1) logical = true
end

assert(size(features, 1) == numel(trainingRows), ...
    "stage8:KernelRowCount", "特征行数与训练对象标记不一致。");
assert(any(trainingRows), "stage8:EmptyKernelTraining", ...
    "建核时至少需要一个训练对象。");

processed = features;
if options.Standardize
    for columnNumber = 1:size(processed, 2)
        observed = processed(trainingRows, columnNumber);
        observed = observed(isfinite(observed));
        if isempty(observed)
            fillValue = 0;
        else
            fillValue = median(observed);
        end
        missing = ~isfinite(processed(:, columnNumber));
        processed(missing, columnNumber) = fillValue;
    end

    center = mean(processed(trainingRows, :), 1);
    scale = std(processed(trainingRows, :), 0, 1);
    scale(scale < 1e-12 | ~isfinite(scale)) = 1;
    processed = (processed - center) ./ scale;
    if ~isempty(processed)
        processed = processed / sqrt(size(processed, 2));
    end
end

trainingFeatures = processed(trainingRows, :);
trainingDistance = pdist2(trainingFeatures, trainingFeatures, ...
    "squaredeuclidean");
upperValues = trainingDistance(triu(true(size(trainingDistance)), 1));
positiveValues = upperValues(upperValues > 1e-12 & isfinite(upperValues));
if isempty(positiveValues)
    bandwidth = 1;
else
    bandwidth = median(positiveValues);
end

squaredDistance = pdist2(processed, processed, "squaredeuclidean");
kernel = exp(-squaredDistance / (2 * bandwidth));
kernel = (kernel + kernel') / 2;
kernel(1:size(kernel, 1) + 1:end) = 1;

info.bandwidth = bandwidth;
info.trainingObjects = sum(trainingRows);
info.featureCount = size(features, 2);
info.standardized = options.Standardize;
end
