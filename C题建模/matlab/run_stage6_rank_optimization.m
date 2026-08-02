function run_stage6_rank_optimization(seed)
% 阶段6：在阶段4数值预测上加入细胞内排名目标。
% 排名模型只用内层留出基因调参，外层验证标签仅用于最后评价。

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
end

rng(seed, "twister");

matlabDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(matlabDir);
repoRoot = fileparts(projectDir);
dataRoot = fullfile(repoRoot, "2026年校赛题目", "C", "数据文件");
processedDir = fullfile(projectDir, "data_processed");
resultsDir = fullfile(projectDir, "results");
scoringDir = fullfile(resultsDir, "stage6_scoring");
if ~isfolder(scoringDir)
    mkdir(scoringDir);
end

labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");
geneMetadata = readtable(fullfile(dataRoot, "metadata", "gene_metadata.csv"), ...
    TextType="string", VariableNamingRule="preserve");
geneBaseScores = readtable(fullfile(resultsDir, "stage4_gene_base_scores.csv"), ...
    TextType="string", VariableNamingRule="preserve");
stage4Summary = readtable(fullfile(resultsDir, "stage4_gene_model_summary.csv"), ...
    TextType="string", VariableNamingRule="preserve");

[isValidation, validationGroup, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed);
checkPersistedSplit(processedDir, heldCells, heldGenes);

training = labels(~isValidation, :);
validation = labels(isValidation, :);
validationGroup = validationGroup(isValidation);
trainGenes = unique(training.perturbation_gene, "stable");
allGenes = geneBaseScores.perturbation_gene;
[knownTrainGenes, trainGeneLocation] = ismember(trainGenes, allGenes);
assert(all(knownTrainGenes), "阶段4基础分缺少部分训练基因。");

% 继续使用阶段4已经确定的元数据处理方式和数值岭回归惩罚参数。
[rawFeatures, metadataKnown] = extractGeneFeatures(geneMetadata, allGenes);
trainFeatureRows = ismember(allGenes, trainGenes);
X = fitAndEncodeFeatures(rawFeatures, trainFeatureRows, metadataKnown);
stage4Lambda = stage4Summary.ridge_lambda(1);

% 内层按基因留出，模拟正式测试中从未见过的新基因。
rng(seed + 1000, "twister");
nInner = max(1, round(0.20 * numel(trainGenes)));
innerValidationMask = false(numel(trainGenes), 1);
innerValidationMask(randperm(numel(trainGenes), nInner)) = true;
innerTrainingGenes = trainGenes(~innerValidationMask);
innerValidationGenes = trainGenes(innerValidationMask);
innerTrainingRows = ismember(training.perturbation_gene, innerTrainingGenes);
innerValidationRows = ismember(training.perturbation_gene, innerValidationGenes);

innerTrainingData = training(innerTrainingRows, :);
innerValidationData = training(innerValidationRows, :);
[innerTrainMean, innerTrainPercentile, innerTrainTop] = ...
    buildGeneTargets(innerTrainingData, innerTrainingGenes);
[~, innerValidationPercentile, innerValidationTop] = ...
    buildGeneTargets(innerValidationData, innerValidationGenes);

[~, innerTrainFeatureLocation] = ismember(innerTrainingGenes, allGenes);
[~, innerValidationFeatureLocation] = ismember(innerValidationGenes, allGenes);
XInnerTrain = X(innerTrainFeatureLocation, :);
XInnerValidation = X(innerValidationFeatureLocation, :);

% 数值基线与排名目标分开训练，再由内层官方口径决定混合比例。
[numericBeta, numericIntercept] = ...
    fitRidge(XInnerTrain, innerTrainMean, stage4Lambda);
innerNumericPrediction = numericIntercept + XInnerValidation * numericBeta;

lambdaGrid = logspace(-3, 3, 25);
[percentileLambda, percentileInnerMse] = selectRidgeLambda( ...
    XInnerTrain, innerTrainPercentile, XInnerValidation, ...
    innerValidationPercentile, lambdaGrid);
[topLambda, topInnerMse] = selectRidgeLambda( ...
    XInnerTrain, innerTrainTop, XInnerValidation, ...
    innerValidationTop, lambdaGrid);

[percentileBeta, percentileIntercept] = ...
    fitRidge(XInnerTrain, innerTrainPercentile, percentileLambda);
[topBeta, topIntercept] = ...
    fitRidge(XInnerTrain, innerTrainTop, topLambda);
innerPercentilePrediction = percentileIntercept + ...
    XInnerValidation * percentileBeta;
innerTopPrediction = topIntercept + XInnerValidation * topBeta;

% 官方排序部分中，Spearman占30%，NDCG与Precision合计占55%。
spearmanWeight = 0.30 / 0.85;
topWeight = 0.55 / 0.85;
innerRankPrediction = calibrateRankScore(innerTrainMean, ...
    innerTrainPercentile, innerTrainTop, innerPercentilePrediction, ...
    innerTopPrediction, spearmanWeight, topWeight);

innerBaseRows = mapGeneValues(innerValidationData.perturbation_gene, ...
    innerValidationGenes, innerNumericPrediction);
innerRankRows = mapGeneValues(innerValidationData.perturbation_gene, ...
    innerValidationGenes, innerRankPrediction);
alphaGrid = (0:0.05:1)';
alphaSummary = evaluateAlphaGrid(alphaGrid, innerBaseRows, innerRankRows, ...
    innerValidationData);
[~, bestAlphaRow] = max(alphaSummary.final_score);
selectedAlpha = alphaSummary.alpha(bestAlphaRow);

% 外层训练使用全部可见基因；已见基因保留实测排名，新基因才使用元数据回归。
[trainGeneMean, trainPercentile, trainTop] = ...
    buildGeneTargets(training, trainGenes);
XTrain = X(trainGeneLocation, :);
[percentileBeta, percentileIntercept] = ...
    fitRidge(XTrain, trainPercentile, percentileLambda);
[topBeta, topIntercept] = fitRidge(XTrain, trainTop, topLambda);
percentileModel = percentileIntercept + X * percentileBeta;
topModel = topIntercept + X * topBeta;

percentileHybrid = percentileModel;
topHybrid = topModel;
percentileHybrid(trainGeneLocation) = trainPercentile;
topHybrid(trainGeneLocation) = trainTop;
rankCalibrated = calibrateRankScore(trainGeneMean, trainPercentile, ...
    trainTop, percentileHybrid, topHybrid, spearmanWeight, topWeight);

stage4Base = geneBaseScores.hybrid_ridge_base;
rankOptimizedGene = (1 - selectedAlpha) * stage4Base + ...
    selectedAlpha * rankCalibrated;
baseValidation = mapGeneValues(validation.perturbation_gene, ...
    allGenes, stage4Base);
rankValidation = mapGeneValues(validation.perturbation_gene, ...
    allGenes, rankOptimizedGene);

modelNames = ["stage4_hybrid_base"; "rank_optimized"];
predictions = [baseValidation, rankValidation];
for modelIndex = 1:numel(modelNames)
    submission = validation(:, ["cell_line_id", "perturbation_gene"]);
    submission.label = predictions(:, modelIndex);
    writetable(submission, fullfile(scoringDir, ...
        "prediction_" + modelNames(modelIndex) + ".csv"));
end

groupSummary = buildGroupSummary(modelNames, predictions, validation, ...
    validationGroup);
writetable(groupSummary, fullfile(resultsDir, ...
    "stage6_rank_optimization_summary.csv"));
writetable(alphaSummary, fullfile(resultsDir, ...
    "stage6_inner_alpha_selection.csv"));

empiricalPercentile = nan(numel(allGenes), 1);
empiricalTop = nan(numel(allGenes), 1);
empiricalPercentile(trainGeneLocation) = trainPercentile;
empiricalTop(trainGeneLocation) = trainTop;
geneTargetTable = table(allGenes, trainFeatureRows, empiricalPercentile, ...
    empiricalTop, percentileModel, topModel, percentileHybrid, topHybrid, ...
    rankCalibrated, stage4Base, rankOptimizedGene, ...
    'VariableNames', {'perturbation_gene', 'is_training_gene', ...
    'empirical_percentile', 'empirical_top_score', ...
    'metadata_percentile_prediction', 'metadata_top_prediction', ...
    'hybrid_percentile', 'hybrid_top_score', 'rank_calibrated_score', ...
    'stage4_base_score', 'rank_optimized_score'});
writetable(geneTargetTable, fullfile(resultsDir, ...
    "stage6_gene_rank_targets.csv"));

parameter = ["stage4_ridge_lambda"; "percentile_ridge_lambda"; ...
    "top_ridge_lambda"; "percentile_inner_mse"; "top_inner_mse"; ...
    "spearman_proxy_weight"; "top_proxy_weight"; "selected_alpha"; ...
    "inner_baseline_score"; "inner_selected_score"; "training_genes"; ...
    "inner_validation_genes"];
value = [stage4Lambda; percentileLambda; topLambda; percentileInnerMse; ...
    topInnerMse; spearmanWeight; topWeight; selectedAlpha; ...
    alphaSummary.final_score(alphaSummary.alpha == 0); ...
    alphaSummary.final_score(bestAlphaRow); numel(trainGenes); ...
    numel(innerValidationGenes)];
parameterTable = table(parameter, value);
writetable(parameterTable, fullfile(resultsDir, ...
    "stage6_model_parameters.csv"));

fprintf("阶段6完成：内层选择 alpha=%.2f，基线 %.6f，混合后 %.6f。\n", ...
    selectedAlpha, alphaSummary.final_score(alphaSummary.alpha == 0), ...
    alphaSummary.final_score(bestAlphaRow));
fprintf("结果已写入 %s\n", resultsDir);
end

function [isValidation, group, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed)
rng(seed, "twister");
trainCells = sort(unique(labels.cell_line_id));
trainGenes = sort(unique(labels.perturbation_gene));
nHeldCells = max(1, round(0.11 * numel(trainCells)));
nHeldGenes = max(1, round(0.15 * numel(trainGenes)));
heldCells = trainCells(randperm(numel(trainCells), nHeldCells));
heldGenes = trainGenes(randperm(numel(trainGenes), nHeldGenes));
isHeldCell = ismember(labels.cell_line_id, heldCells);
isHeldGene = ismember(labels.perturbation_gene, heldGenes);
oldOldCandidates = find(~isHeldCell & ~isHeldGene);
nHeldOldOldRows = max(1, round(0.08 * numel(oldOldCandidates)));
heldOldOldRows = oldOldCandidates(randperm(numel(oldOldCandidates), ...
    nHeldOldOldRows));
isOldOldValidation = false(height(labels), 1);
isOldOldValidation(heldOldOldRows) = true;
isValidation = isHeldCell | isHeldGene | isOldOldValidation;
group = strings(height(labels), 1);
group(~isHeldCell & ~isHeldGene) = "old_cell_old_gene";
group(~isHeldCell & isHeldGene) = "old_cell_new_gene";
group(isHeldCell & ~isHeldGene) = "new_cell_old_gene";
group(isHeldCell & isHeldGene) = "new_cell_new_gene";
end

function checkPersistedSplit(processedDir, heldCells, heldGenes)
savedCells = readtable(fullfile(processedDir, "stage1_heldout_cells.csv"), ...
    TextType="string");
savedGenes = readtable(fullfile(processedDir, "stage1_heldout_genes.csv"), ...
    TextType="string");
assert(isequal(sort(heldCells), sort(savedCells.cell_line_id)), ...
    "留出细胞名单与阶段1不一致。");
assert(isequal(sort(heldGenes), sort(savedGenes.perturbation_gene)), ...
    "留出基因名单与阶段1不一致。");
end

function [geneMean, percentileTarget, topTarget] = ...
    buildGeneTargets(rows, geneNames)
rowPercentile = zeros(height(rows), 1);
rowTopScore = zeros(height(rows), 1);
[cellGroup, ~] = findgroups(rows.cell_line_id);
for cellIndex = 1:max(cellGroup)
    mask = cellGroup == cellIndex;
    n = sum(mask);
    rankValue = tiedrank(-rows.label(mask));
    if n > 1
        rowPercentile(mask) = 1 - (rankValue - 1) / (n - 1);
    else
        rowPercentile(mask) = 0.5;
    end
    rowTopScore(mask) = (double(rankValue <= 5) + ...
        double(rankValue <= 10) + double(rankValue <= 15)) / 3;
end

[geneGroup, groupedGenes] = findgroups(rows.perturbation_gene);
groupMean = splitapply(@mean, rows.label, geneGroup);
groupPercentile = splitapply(@mean, rowPercentile, geneGroup);
groupTop = splitapply(@mean, rowTopScore, geneGroup);
[known, location] = ismember(geneNames, groupedGenes);
assert(all(known), "排名目标缺少部分基因。");
geneMean = groupMean(location);
percentileTarget = groupPercentile(location);
topTarget = groupTop(location);
end

function [raw, metadataKnown] = extractGeneFeatures(metadata, geneNames)
[metadataKnown, location] = ismember(geneNames, metadata.gene_symbol);
knownLocations = location(metadataKnown);
assert(numel(unique(knownLocations)) == numel(knownLocations), ...
    "基因元数据中存在重复符号。");

numeric = nan(numel(geneNames), 7);
if any(metadataKnown)
    rows = knownLocations;
    numeric(metadataKnown, 1) = metadata.targetp_score(rows);
    numeric(metadataKnown, 2) = metadata.coexpression_gnf_n50_score(rows);
    numeric(metadataKnown, 3) = metadata.pgc_induction_score(rows);
    numeric(metadataKnown, 4) = tokenCount(metadata.curation_evidence(rows), ",");
    numeric(metadataKnown, 5) = tokenCount(metadata.pathways(rows), "|");
    numeric(metadataKnown, 6) = tokenCount(metadata.synonyms(rows), "|");
    numeric(metadataKnown, 7) = strlength(metadata.description(rows));
end

categories = strings(numel(geneNames), 5);
categoryFields = ["sub_mito_location", "mito_domain_score", ...
    "yeast_mito_homolog_score", "rickettsia_homolog_score", "msms_score"];
for featureIndex = 1:numel(categoryFields)
    values = strings(numel(geneNames), 1);
    if any(metadataKnown)
        values(metadataKnown) = ...
            metadata.(categoryFields(featureIndex))(knownLocations);
    end
    values(ismissing(values) | strlength(strtrim(values)) == 0) = "<missing>";
    categories(:, featureIndex) = values;
end

raw.numeric = numeric;
raw.categories = categories;
end

function counts = tokenCount(values, delimiter)
counts = zeros(numel(values), 1);
for valueIndex = 1:numel(values)
    value = values(valueIndex);
    if ismissing(value) || strlength(strtrim(value)) == 0 || value == "-"
        counts(valueIndex) = 0;
    else
        counts(valueIndex) = numel(strsplit(value, delimiter));
    end
end
end

function X = fitAndEncodeFeatures(raw, trainRows, metadataKnown)
Xnumeric = raw.numeric;
for featureIndex = 1:size(Xnumeric, 2)
    observed = Xnumeric(trainRows, featureIndex);
    observed = observed(~isnan(observed));
    if isempty(observed)
        fillValue = 0;
    else
        fillValue = median(observed);
    end
    missing = isnan(Xnumeric(:, featureIndex));
    Xnumeric(missing, featureIndex) = fillValue;
end
means = mean(Xnumeric(trainRows, :), 1);
scales = std(Xnumeric(trainRows, :), 0, 1);
scales(scales < 1e-12 | ~isfinite(scales)) = 1;
X = (Xnumeric - means) ./ scales;

for featureIndex = 1:size(raw.categories, 2)
    values = raw.categories(:, featureIndex);
    levels = unique(values(trainRows), "stable");
    % 第一类交给截距表示，其余类别分别建立0-1列。
    for levelIndex = 2:numel(levels)
        X(:, end + 1) = double(values == levels(levelIndex));
    end
end
X(:, end + 1) = double(~metadataKnown);
end

function [lambda, bestMse] = selectRidgeLambda( ...
    XTrain, yTrain, XValidation, yValidation, lambdas)
bestMse = inf;
lambda = lambdas(1);
for value = lambdas
    [beta, intercept] = fitRidge(XTrain, yTrain, value);
    prediction = intercept + XValidation * beta;
    mse = mean((prediction - yValidation).^2);
    if mse < bestMse
        bestMse = mse;
        lambda = value;
    end
end
end

function [beta, intercept] = fitRidge(X, y, lambda)
centerX = mean(X, 1);
centerY = mean(y);
Xc = X - centerX;
yc = y - centerY;
penalty = lambda * eye(size(X, 2));
beta = (Xc' * Xc + penalty) \ (Xc' * yc);
intercept = centerY - centerX * beta;
end

function calibrated = calibrateRankScore(numericTrain, percentileTrain, ...
    topTrain, percentilePrediction, topPrediction, percentileWeight, topWeight)
percentileScale = std(percentileTrain);
topScale = std(topTrain);
if percentileScale < 1e-12
    percentileScale = 1;
end
if topScale < 1e-12
    topScale = 1;
end
percentileTrainZ = (percentileTrain - mean(percentileTrain)) / percentileScale;
topTrainZ = (topTrain - mean(topTrain)) / topScale;
percentilePredictionZ = ...
    (percentilePrediction - mean(percentileTrain)) / percentileScale;
topPredictionZ = (topPrediction - mean(topTrain)) / topScale;

combinedTrain = percentileWeight * percentileTrainZ + topWeight * topTrainZ;
combinedPrediction = percentileWeight * percentilePredictionZ + ...
    topWeight * topPredictionZ;
combinedScale = std(combinedTrain);
if combinedScale < 1e-12
    combinedScale = 1;
end
combinedPrediction = ...
    (combinedPrediction - mean(combinedTrain)) / combinedScale;

numericScale = std(numericTrain);
if numericScale < 1e-12
    numericScale = 1;
end
calibrated = mean(numericTrain) + numericScale * combinedPrediction;
end

function values = mapGeneValues(geneNames, knownGenes, knownValues)
[known, location] = ismember(geneNames, knownGenes);
assert(all(known), "预测数据中存在没有对应分数的基因。");
values = knownValues(location);
end

function summary = evaluateAlphaGrid(alphaGrid, basePrediction, ...
    rankPrediction, validation)
nAlpha = numel(alphaGrid);
metrics = zeros(nAlpha, 6);
for alphaIndex = 1:nAlpha
    alpha = alphaGrid(alphaIndex);
    prediction = (1 - alpha) * basePrediction + alpha * rankPrediction;
    result = calculateMetrics(validation, prediction);
    metrics(alphaIndex, :) = [result.final_score, result.rho_macro, ...
        result.ndcg_score, result.precision_score, result.rmse, ...
        result.rmse_score];
end
summary = array2table([alphaGrid, metrics], 'VariableNames', ...
    {'alpha', 'final_score', 'rho_macro', 'ndcg_score', ...
    'precision_score', 'rmse', 'rmse_score'});
end

function summary = buildGroupSummary(modelNames, predictions, ...
    validation, validationGroup)
groupNames = ["all"; "old_cell_old_gene"; "old_cell_new_gene"; ...
    "new_cell_old_gene"; "new_cell_new_gene"];
modelColumn = strings(0, 1);
groupColumn = strings(0, 1);
metricValues = zeros(0, 8);
for modelIndex = 1:numel(modelNames)
    for groupIndex = 1:numel(groupNames)
        groupName = groupNames(groupIndex);
        if groupName == "all"
            mask = true(height(validation), 1);
        else
            mask = validationGroup == groupName;
        end
        result = calculateMetrics(validation(mask, :), ...
            predictions(mask, modelIndex));
        modelColumn(end + 1, 1) = modelNames(modelIndex);
        groupColumn(end + 1, 1) = groupName;
        metricValues(end + 1, :) = [sum(mask), result.final_score, ...
            result.rho_macro, result.ndcg_score, result.precision_score, ...
            result.rmse, result.rmse_score, result.spearman_score];
    end
end
summary = array2table(metricValues, 'VariableNames', ...
    {'rows', 'final_score', 'rho_macro', 'ndcg_score', ...
    'precision_score', 'rmse', 'rmse_score', 'spearman_score'});
summary.model = modelColumn;
summary.group = groupColumn;
summary = movevars(summary, ["model", "group"], "Before", 1);
end

function result = calculateMetrics(answer, prediction)
[cellGroup, ~] = findgroups(answer.cell_line_id);
nCells = max(cellGroup);
spearmanValues = zeros(nCells, 1);
precisionValues = zeros(nCells, 3);
ndcgValues = zeros(nCells, 3);
kValues = [5, 10, 15];

for cellIndex = 1:nCells
    mask = cellGroup == cellIndex;
    truth = answer.label(mask);
    predicted = prediction(mask);
    genes = answer.perturbation_gene(mask);
    n = numel(truth);
    originalOrder = (1:n)';
    predictedRank = tiedrank(-predicted);
    trueRank = tiedrank(-truth);
    spearmanValues(cellIndex) = pearsonOrZero(predictedRank, trueRank);
    [~, predictedOrder] = sortrows([-predicted, originalOrder], [1, 2]);
    [~, trueOrder] = sortrows([-truth, originalOrder], [1, 2]);

    for kIndex = 1:numel(kValues)
        k = kValues(kIndex);
        take = min(k, n);
        predictedTop = predictedOrder(1:take);
        trueTopGenes = genes(trueOrder(1:take));
        precisionValues(cellIndex, kIndex) = ...
            sum(ismember(genes(predictedTop), trueTopGenes)) / k;

        relevance = max(k - trueRank(predictedTop) + 1, 0);
        relevance(trueRank(predictedTop) > k) = 0;
        discounts = log2((2:(take + 1))');
        dcg = sum((2 .^ relevance - 1) ./ discounts);
        idealRelevance = (k:-1:1)';
        idealDiscounts = log2((2:(k + 1))');
        idcg = sum((2 .^ idealRelevance - 1) ./ idealDiscounts);
        ndcgValues(cellIndex, kIndex) = dcg / idcg;
    end
end

result.rho_macro = mean(spearmanValues);
result.spearman_score = (result.rho_macro + 1) / 2;
result.precision_score = mean(precisionValues, "all");
result.ndcg_score = mean(ndcgValues, "all");
result.rmse = sqrt(mean((prediction - answer.label).^2));
labelScale = std(answer.label, 1);
result.rmse_score = 1 / (1 + result.rmse / (labelScale + 1e-12));
result.final_score = 100 * (0.30 * result.spearman_score + ...
    0.30 * result.ndcg_score + 0.25 * result.precision_score + ...
    0.15 * result.rmse_score);
end

function value = pearsonOrZero(x, y)
x = x - mean(x);
y = y - mean(y);
denominator = sqrt(sum(x .^ 2) * sum(y .^ 2));
if denominator <= 1e-12
    value = 0;
else
    value = sum(x .* y) / denominator;
end
end
