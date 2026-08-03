function model = stage7_fit_two_step_kernel( ...
    cellKernel, geneKernel, cellIndex, geneIndex, target, lambda)
%STAGE7_FIT_TWO_STEP_KERNEL 拟合双边两步核岭回归。
% 训练矩形中的未观测组合先用训练残差的细胞、基因加性均值填补。

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

[residualMatrix, observed, trainingCells, trainingGenes] = ...
    stage7_build_residual_matrix(cellIndex, geneIndex, target);

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
% 阶段9需要在不重新拟合模型的前提下预测反事实细胞状态。
model.trainingResidualMatrix = residualMatrix;
model.cellTrainingKernel = cellTrainKernel;
model.geneMap = geneMap;
end
