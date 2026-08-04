function [baseByGene, info] = stage7_fit_fold_gene_base( ...
    geneFeatures, trainingGeneIndex, trainingTarget, splitSeed)
%STAGE7_FIT_FOLD_GENE_BASE 只用当前折训练标签重建混合岭回归底座。
% 已见基因使用本折训练均值；未见基因只能根据基因资料回归预测。

arguments
    geneFeatures double
    trainingGeneIndex (:, 1) double {mustBeInteger, mustBePositive}
    trainingTarget (:, 1) double
    splitSeed (1, 1) double {mustBeInteger, mustBeNonnegative}
end

assert(numel(trainingGeneIndex) == numel(trainingTarget), ...
    "stage7:BaseRowCount", "基础分的训练索引与标签行数不一致。");
assert(~isempty(trainingTarget) && all(isfinite(trainingTarget)), ...
    "stage7:BaseTarget", "基础分训练标签为空或包含非有限值。");
assert(all(trainingGeneIndex <= size(geneFeatures, 1)), ...
    "stage7:BaseGeneRange", "基础分训练基因索引超出特征范围。");

[geneGroup, trainingGenes] = findgroups(trainingGeneIndex);
trainingGeneMean = splitapply(@mean, trainingTarget, geneGroup);
assert(numel(trainingGenes) >= 3, "stage7:TooFewBaseGenes", ...
    "基础分至少需要三个训练基因。");

trainingFeatures = geneFeatures(trainingGenes, :);
nTrainingGenes = numel(trainingGenes);

rng(splitSeed, "twister");
nValidationGenes = max(1, round(0.20 * nTrainingGenes));
nValidationGenes = min(nValidationGenes, nTrainingGenes - 1);
validationGene = false(nTrainingGenes, 1);
validationGene(randperm(nTrainingGenes, nValidationGenes)) = true;
fittingGene = ~validationGene;

lambdaGrid = logspace(-3, 3, 25);
[lambda, validationMse] = selectMetadataLambda(trainingFeatures, ...
    trainingGeneMean, fittingGene, validationGene, lambdaGrid);
[beta, intercept] = fitCenteredRidge( ...
    trainingFeatures, trainingGeneMean, lambda);
baseByGene = intercept + geneFeatures * beta;

% 本折见过的基因直接用训练均值；留出基因保留元数据预测值。
baseByGene(trainingGenes) = trainingGeneMean;
assert(all(isfinite(baseByGene)), "stage7:FiniteFoldBase", ...
    "折内基础分出现 NaN 或 Inf。");

info.lambda = lambda;
info.validationMse = validationMse;
info.trainingGenes = nTrainingGenes;
info.trainingRows = numel(trainingTarget);
end

function [bestLambda, bestMse] = selectMetadataLambda( ...
    features, target, fittingRows, validationRows, lambdaGrid)
bestLambda = lambdaGrid(1);
bestMse = inf;
for lambda = lambdaGrid
    [beta, intercept] = fitCenteredRidge( ...
        features(fittingRows, :), target(fittingRows), lambda);
    prediction = intercept + features(validationRows, :) * beta;
    mse = mean((prediction - target(validationRows)) .^ 2);
    if mse < bestMse
        bestMse = mse;
        bestLambda = lambda;
    end
end
end

function [beta, intercept] = fitCenteredRidge(features, target, lambda)
featureCenter = mean(features, 1);
targetCenter = mean(target);
centeredFeatures = features - featureCenter;
centeredTarget = target - targetCenter;
penalty = lambda * eye(size(features, 2));
beta = (centeredFeatures' * centeredFeatures + penalty) \ ...
    (centeredFeatures' * centeredTarget);
intercept = targetCenter - featureCenter * beta;
end
