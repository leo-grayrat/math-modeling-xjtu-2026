function [residualMatrix, observed, trainingCells, trainingGenes] = ...
    stage7_build_residual_matrix(cellIndex, geneIndex, target)
%STAGE7_BUILD_RESIDUAL_MATRIX 把训练标签整理为细胞乘基因的残差矩阵。
% 少量未观测组合只使用训练行的细胞均值和基因均值进行加性填补。

arguments
    cellIndex (:, 1) double {mustBeInteger, mustBePositive}
    geneIndex (:, 1) double {mustBeInteger, mustBePositive}
    target (:, 1) double
end

assert(numel(cellIndex) == numel(geneIndex) && ...
    numel(cellIndex) == numel(target), ...
    "stage7:ResidualRowCount", "残差矩阵索引与目标行数不一致。");

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

globalMean = mean(residualMatrix(observed));
cellMean = zeros(size(residualMatrix, 1), 1);
geneMean = zeros(1, size(residualMatrix, 2));
for cellNumber = 1:size(residualMatrix, 1)
    rowObserved = observed(cellNumber, :);
    cellMean(cellNumber) = mean(residualMatrix(cellNumber, rowObserved));
end
for geneNumber = 1:size(residualMatrix, 2)
    columnObserved = observed(:, geneNumber);
    geneMean(geneNumber) = mean(residualMatrix(columnObserved, geneNumber));
end
additive = cellMean + geneMean - globalMean;
residualMatrix(~observed) = additive(~observed);
end

