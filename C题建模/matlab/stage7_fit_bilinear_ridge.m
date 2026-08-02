function model = stage7_fit_bilinear_ridge( ...
    cellFeatures, geneFeatures, ownExpression, cellIndex, geneIndex, ...
    target, lambda, options)
%STAGE7_FIT_BILINEAR_RIDGE 用隐式矩阵乘法拟合双线性岭回归。
% 函数不显式建立“样本数×全部交互项”大矩阵，避免内存爆炸。

arguments
    cellFeatures double
    geneFeatures double
    ownExpression double
    cellIndex (:, 1) double {mustBeInteger, mustBePositive}
    geneIndex (:, 1) double {mustBeInteger, mustBePositive}
    target (:, 1) double
    lambda (1, 1) double {mustBePositive}
    options.Tolerance (1, 1) double {mustBePositive} = 1e-4
    options.MaxIterations (1, 1) double {mustBeInteger, mustBePositive} = 60
    options.InitialPoint double = []
end

nCells = size(cellFeatures, 1);
nGenes = size(geneFeatures, 1);
nCellFeatures = size(cellFeatures, 2);
nGeneFeatures = size(geneFeatures, 2);
nInteractions = nCellFeatures * nGeneFeatures;

assert(size(ownExpression, 1) == nCells && ...
    size(ownExpression, 2) == nGenes, ...
    "stage7:OwnExpressionSize", "自身表达矩阵尺寸与细胞、基因数不一致。");
assert(numel(cellIndex) == numel(geneIndex) && ...
    numel(cellIndex) == numel(target), ...
    "stage7:RowCount", "样本索引和目标的行数不一致。");
assert(all(cellIndex <= nCells) && all(geneIndex <= nGenes), ...
    "stage7:IndexRange", "样本索引超出特征矩阵范围。");
assert(all(isfinite(target)), "stage7:FiniteTarget", ...
    "训练目标中存在 NaN 或 Inf。");

linearIndex = sub2ind([nCells, nGenes], cellIndex, geneIndex);
assert(numel(unique(linearIndex)) == numel(linearIndex), ...
    "stage7:DuplicatePairs", "训练数据中存在重复的细胞—基因组合。");

observed = false(nCells, nGenes);
observed(linearIndex) = true;
targetMatrix = zeros(nCells, nGenes);
targetMatrix(linearIndex) = target;
nObserved = numel(target);

rightHandMatrix = (cellFeatures' * targetMatrix) * geneFeatures / nObserved;
rightHandOwn = sum(target .* ownExpression(linearIndex)) / nObserved;
rightHandSide = [rightHandMatrix(:); rightHandOwn];

cellSecondMoment = mean(cellFeatures(cellIndex, :) .^ 2, 1)';
geneSecondMoment = mean(geneFeatures(geneIndex, :) .^ 2, 1);
diagonalInteraction = cellSecondMoment * geneSecondMoment + lambda;
diagonalOwn = mean(ownExpression(linearIndex) .^ 2) + lambda;
preconditionerDiagonal = [diagonalInteraction(:); diagonalOwn];
preconditionerDiagonal(preconditionerDiagonal < 1e-10) = 1e-10;

normalOperator = @(parameter) applyNormalOperator(parameter, ...
    cellFeatures, geneFeatures, ownExpression, observed, nObserved, ...
    lambda, nCellFeatures, nGeneFeatures);
preconditioner = @(value) value ./ preconditionerDiagonal;

if isempty(options.InitialPoint)
    initialPoint = zeros(nInteractions + 1, 1);
else
    initialPoint = options.InitialPoint(:);
    assert(numel(initialPoint) == nInteractions + 1, ...
        "stage7:InitialPointSize", "PCG 初值尺寸不正确。");
end

startTime = tic;
[parameter, flag, relativeResidual, iterations] = pcg( ...
    normalOperator, rightHandSide, options.Tolerance, ...
    options.MaxIterations, preconditioner, [], initialPoint);
elapsedSeconds = toc(startTime);

model.interaction = reshape(parameter(1:nInteractions), ...
    nCellFeatures, nGeneFeatures);
model.ownExpressionCoefficient = parameter(end);
model.lambda = lambda;
model.solverFlag = flag;
model.relativeResidual = relativeResidual;
model.iterations = iterations;
model.elapsedSeconds = elapsedSeconds;
model.parameterVector = parameter;
end

function result = applyNormalOperator(parameter, cellFeatures, ...
    geneFeatures, ownExpression, observed, nObserved, lambda, ...
    nCellFeatures, nGeneFeatures)
% 这里只计算正规方程对一个向量的作用，不建立大设计矩阵。
nInteractions = nCellFeatures * nGeneFeatures;
interaction = reshape(parameter(1:nInteractions), ...
    nCellFeatures, nGeneFeatures);
ownCoefficient = parameter(end);

fitted = (cellFeatures * interaction) * geneFeatures' + ...
    ownCoefficient * ownExpression;
fitted(~observed) = 0;

gradientInteraction = (cellFeatures' * fitted) * geneFeatures / nObserved;
gradientOwn = sum(fitted(observed) .* ownExpression(observed)) / nObserved;
result = [gradientInteraction(:); gradientOwn] + lambda * parameter;
end
