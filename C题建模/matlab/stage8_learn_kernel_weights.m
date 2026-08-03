function result = stage8_learn_kernel_weights( ...
    kernels, trainingObjects, referenceSimilarity, ridge)
%STAGE8_LEARN_KERNEL_WEIGHTS 学习非负且和为一的中心化核对齐权重。

arguments
    kernels (1, :) cell
    trainingObjects (:, 1) double {mustBeInteger, mustBePositive}
    referenceSimilarity double
    ridge (1, 1) double {mustBeNonnegative} = 0.05
end

nObjects = numel(trainingObjects);
nKernels = numel(kernels);
assert(isequal(size(referenceSimilarity), [nObjects, nObjects]), ...
    "stage8:ReferenceSize", "参照相似度与训练对象数不一致。");
assert(nKernels >= 2, "stage8:KernelCount", ...
    "学习权重至少需要两个候选核。");

reference = centerAndNormalize(referenceSimilarity);
centeredKernels = cell(1, nKernels);
for kernelNumber = 1:nKernels
    candidate = kernels{kernelNumber};
    assert(size(candidate, 1) == size(candidate, 2), ...
        "stage8:KernelSquare", "候选核必须为方阵。");
    assert(max(trainingObjects) <= size(candidate, 1), ...
        "stage8:KernelObjectRange", "训练对象索引超出候选核范围。");
    centeredKernels{kernelNumber} = centerAndNormalize( ...
        candidate(trainingObjects, trainingObjects));
end

gram = zeros(nKernels);
alignment = zeros(nKernels, 1);
for left = 1:nKernels
    alignment(left) = sum(centeredKernels{left} .* reference, "all");
    for right = left:nKernels
        value = sum(centeredKernels{left} .* ...
            centeredKernels{right}, "all");
        gram(left, right) = value;
        gram(right, left) = value;
    end
end

hessian = 2 * (gram + ridge * eye(nKernels));
hessian = (hessian + hessian') / 2;
linearTerm = -2 * alignment;
options = optimoptions("quadprog", "Display", "off", ...
    "Algorithm", "interior-point-convex");
[weights, objective, exitflag] = quadprog(hessian, linearTerm, ...
    [], [], ones(1, nKernels), 1, zeros(nKernels, 1), [], ...
    ones(nKernels, 1) / nKernels, options);
assert(exitflag > 0 && all(isfinite(weights)), ...
    "stage8:WeightSolver", "多核权重二次规划没有得到可靠解。");

weights(weights < 1e-12) = 0;
weights = weights / sum(weights);
result.weights = weights;
result.alignment = alignment;
result.objective = objective;
result.exitflag = exitflag;
result.ridge = ridge;
end

function centered = centerAndNormalize(values)
values = (values + values') / 2;
rowMean = mean(values, 2);
centered = values - rowMean - rowMean' + mean(values, "all");
scale = norm(centered, "fro");
assert(scale > 1e-12 && isfinite(scale), ...
    "stage8:DegenerateKernel", "中心化后的核或参照相似度没有有效变化。");
centered = centered / scale;
end
