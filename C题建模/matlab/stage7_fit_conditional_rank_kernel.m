function model = stage7_fit_conditional_rank_kernel( ...
    cellKernel, geneKernel, cellIndex, geneIndex, target, lambda, eta)
%STAGE7_FIT_CONDITIONAL_RANK_KERNEL 联合拟合连续残差和同细胞条件排序。
% eta 只提高细胞内部基因差值误差的权重；eta=0 时退回阶段7两步核岭。

arguments
    cellKernel double
    geneKernel double
    cellIndex (:, 1) double {mustBeInteger, mustBePositive}
    geneIndex (:, 1) double {mustBeInteger, mustBePositive}
    target (:, 1) double
    lambda (1, 1) double {mustBePositive}
    eta (1, 1) double {mustBeNonnegative}
end

assert(size(cellKernel, 1) == size(cellKernel, 2), ...
    "rankrls:CellKernelSize", "细胞核矩阵必须为方阵。");
assert(size(geneKernel, 1) == size(geneKernel, 2), ...
    "rankrls:GeneKernelSize", "基因核矩阵必须为方阵。");

[residualMatrix, observed, trainingCells, trainingGenes] = ...
    stage7_build_residual_matrix(cellIndex, geneIndex, target);
cellTrainKernel = cellKernel(trainingCells, trainingCells);
geneTrainKernel = geneKernel(trainingGenes, trainingGenes);
cellMap = cellKernel(:, trainingCells) / ...
    (cellTrainKernel + lambda * eye(numel(trainingCells)));

nGenes = numel(trainingGenes);
centering = eye(nGenes) - ones(nGenes) / nGenes;
lossWeight = eye(nGenes) + eta * centering;
geneSystem = lossWeight * geneTrainKernel + lambda * eye(nGenes);
geneCoefficientMap = geneSystem \ lossWeight;
geneMap = geneKernel(:, trainingGenes) * geneCoefficientMap;

model.predictionMatrix = (cellMap * residualMatrix) * geneMap';
model.lambda = lambda;
model.eta = eta;
model.trainingCells = trainingCells;
model.trainingGenes = trainingGenes;
model.observedFraction = mean(observed, "all");
model.trainingResidualMatrix = residualMatrix;
model.cellTrainingKernel = cellTrainKernel;
model.geneMap = geneMap;
model.centeredLossMultiplier = 1 + eta;
end

