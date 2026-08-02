function model = stage7_fit_two_step_kernel( ...
    cellKernel, geneKernel, cellIndex, geneIndex, target, lambda)
%STAGE7_FIT_TWO_STEP_KERNEL 拟合双边两步核岭回归。
% 训练矩形中少量未观测组合先用训练残差的细胞、基因加性均值填补。

arguments
    cellKernel double
    geneKernel double
    cellIndex (:, 1) double {mustBeInteger, mustBePositive}
    geneIndex (:, 1) double {mustBeInteger, mustBePositive}
    target (:, 1) double
    lambda (1, 1) double {mustBePositive}
end

assert(size(cellKernel, 1) == size(cellKernel, 2), ...
    "stage7:CellKernelSize", "细胞核矩阵必须为方阵。");
assert(size(geneKernel, 1) == size(geneKernel, 2), ...
    "stage7:GeneKernelSize", "基因核矩阵必须为方阵。");
assert(numel(cellIndex) == numel(geneIndex) && ...
    numel(cellIndex) == numel(target), ...
    "stage7:KernelRowCount", "核回归索引与目标行数不一致。");

trainingCells = unique(cellIndex, "sorted");
trainingGenes = unique(geneIndex, "sorted");
[~, localCell] = ismember(cellIndex, trainingCells);
[~, localGene] = ismember(geneIndex, trainingGenes);
linearIndex = sub2ind([numel(trainingCells), numel(trainingGenes)], ...
    localCell, localGene);
assert(numel(unique(linearIndex)) == numel(linearIndex), ...
    "stage7:KernelDuplicatePairs", "核回归训练组合存在重复行。");

observed = false(numel(trainingCells), numel(trainingGenes));
observed(linearIndex) = true;
residualMatrix = nan(numel(trainingCells), numel(trainingGenes));
residualMatrix(linearIndex) = target;
residualMatrix = fillMissingAdditive(residualMatrix, observed);

cellTrainKernel = cellKernel(trainingCells, trainingCells);
geneTrainKernel = geneKernel(trainingGenes, trainingGenes);
cellMap = cellKernel(:, trainingCells) / ...
    (cellTrainKernel + lambda * eye(numel(trainingCells)));
geneMap = geneKernel(:, trainingGenes) / ...
    (geneTrainKernel + lambda * eye(numel(trainingGenes)));

model.predictionMatrix = (cellMap * residualMatrix) * geneMap';
model.lambda = lambda;
model.trainingCells = trainingCells;
model.trainingGenes = trainingGenes;
model.observedFraction = mean(observed, "all");
end

function filled = fillMissingAdditive(values, observed)
% 用仅来自训练行的加性均值填补训练矩形缺口。
globalMean = mean(values(observed));
cellMean = zeros(size(values, 1), 1);
geneMean = zeros(1, size(values, 2));
for cellNumber = 1:size(values, 1)
    rowObserved = observed(cellNumber, :);
    cellMean(cellNumber) = mean(values(cellNumber, rowObserved));
end
for geneNumber = 1:size(values, 2)
    columnObserved = observed(:, geneNumber);
    geneMean(geneNumber) = mean(values(columnObserved, geneNumber));
end
filled = values;
additive = cellMean + geneMean - globalMean;
filled(~observed) = additive(~observed);
end
