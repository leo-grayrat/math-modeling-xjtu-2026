function run_stage8_weighted_multikernel(seed, runMode)
%RUN_STAGE8_WEIGHTED_MULTIKERNEL 比较阶段7总核与有权重多核。
% smoke 只跑一组真实划分；full 先跑三组，达线后才增加两组。

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
    runMode (1, 1) string {mustBeMember(runMode, ["smoke", "full"])} = "full"
end

rng(seed, "twister");
matlabDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(matlabDir);
repoRoot = fileparts(projectDir);
resultsDir = fullfile(projectDir, "results");
data = stage8_prepare_multikernel_data(repoRoot, projectDir, seed);

kernelLambda = 0.1;
alignmentRidge = 0.05;
allMetrics = table;
allWeights = table;
allBandwidths = table;

if runMode == "smoke"
    [allMetrics, allWeights, allBandwidths] = evaluateFold( ...
        data, 1, seed, kernelLambda, alignmentRidge, false);
    writetable(allMetrics, fullfile(resultsDir, "stage8_smoke_metrics.csv"));
    writetable(allWeights, fullfile(resultsDir, "stage8_smoke_weights.csv"));
    writetable(allBandwidths, fullfile(resultsDir, ...
        "stage8_smoke_bandwidths.csv"));
    fprintf("阶段8真实数据冒烟测试完成，分数和权重均已写入 results。\n");
    return
end

for foldNumber = 1:3
    [metrics, weights, bandwidths] = evaluateFold(data, foldNumber, ...
        seed, kernelLambda, alignmentRidge, true);
    allMetrics = [allMetrics; metrics]; %#ok<AGROW>
    allWeights = [allWeights; weights]; %#ok<AGROW>
    allBandwidths = [allBandwidths; bandwidths]; %#ok<AGROW>
end

preliminaryGate = summarizeGate(allMetrics, allWeights, 3, "前三组");
gateSummary = preliminaryGate;
if preliminaryGate.pass_score_gate
    fprintf("前三组达到扩展门槛，继续运行第4、5组新划分。\n");
    for foldNumber = 4:5
        [metrics, weights, bandwidths] = evaluateFold(data, foldNumber, ...
            seed, kernelLambda, alignmentRidge, false);
        allMetrics = [allMetrics; metrics]; %#ok<AGROW>
        allWeights = [allWeights; weights]; %#ok<AGROW>
        allBandwidths = [allBandwidths; bandwidths]; %#ok<AGROW>
    end
    finalGate = summarizeGate(allMetrics, allWeights, 5, "五组终检");
    gateSummary = [gateSummary; finalGate];
else
    fprintf("前三组没有达到扩展门槛，按停止线保留阶段7基线。\n");
end

writetable(allMetrics, fullfile(resultsDir, "stage8_fold_metrics.csv"));
writetable(allWeights, fullfile(resultsDir, "stage8_kernel_weights.csv"));
writetable(allBandwidths, fullfile(resultsDir, ...
    "stage8_kernel_bandwidths.csv"));
writetable(gateSummary, fullfile(resultsDir, "stage8_gate_decision.csv"));
end

function [metrics, weights, bandwidths] = evaluateFold( ...
    data, foldNumber, seed, kernelLambda, alignmentRidge, includeAblations)
[trainingRows, validationRows, validationGroup] = makeInnerSplit( ...
    data, seed + 3000 + foldNumber);
trainingCells = unique(data.rowCellIndex(trainingRows), "sorted");
trainingGenes = unique(data.rowGeneIndex(trainingRows), "sorted");
cellTrainingMask = false(numel(data.cellNames), 1);
geneTrainingMask = false(numel(data.geneNames), 1);
cellTrainingMask(trainingCells) = true;
geneTrainingMask(trainingGenes) = true;

[cellBlocks, geneBlocks, baseGeneFeatures] = buildFoldFeatures( ...
    data, cellTrainingMask, geneTrainingMask);
[cellKernels, cellTotalKernel, cellKernelInfo] = buildKernelSet( ...
    cellBlocks, cellTrainingMask);
[geneKernels, geneTotalKernel, geneKernelInfo] = buildKernelSet( ...
    geneBlocks, geneTrainingMask);

[baseByGene, baseInfo] = stage7_fit_fold_gene_base( ...
    baseGeneFeatures, data.rowGeneIndex(trainingRows), ...
    data.labels.label(trainingRows), seed + 5000 + foldNumber);
baseByRow = baseByGene(data.rowGeneIndex);
trainingResidual = data.labels.label(trainingRows) - baseByRow(trainingRows);
[cellReference, geneReference] = buildResidualReferences( ...
    data.rowCellIndex(trainingRows), data.rowGeneIndex(trainingRows), ...
    trainingResidual);

cellWeightResult = stage8_learn_kernel_weights(cellKernels, ...
    trainingCells, cellReference, alignmentRidge);
geneWeightResult = stage8_learn_kernel_weights(geneKernels, ...
    trainingGenes, geneReference, alignmentRidge);
cellEqualKernel = combineKernels(cellKernels, ...
    ones(numel(cellKernels), 1) / numel(cellKernels));
geneEqualKernel = combineKernels(geneKernels, ...
    ones(numel(geneKernels), 1) / numel(geneKernels));
cellWeightedKernel = combineKernels(cellKernels, cellWeightResult.weights);
geneWeightedKernel = combineKernels(geneKernels, geneWeightResult.weights);

validation = data.labels(validationRows, :);
baselinePrediction = baseByRow(validationRows);
baselineMetrics = predictAndScore(cellTotalKernel, geneTotalKernel, ...
    data, trainingRows, validationRows, trainingResidual, ...
    baselinePrediction, validation, kernelLambda);
metrics = metricRow(foldNumber, "stage7_total_kernel", "primary", "all", ...
    baseInfo.lambda, baselineMetrics, baselineMetrics, ...
    validation, validationGroup, baselineMetrics.prediction);

equalMetrics = predictAndScore(cellEqualKernel, geneEqualKernel, ...
    data, trainingRows, validationRows, trainingResidual, ...
    baselinePrediction, validation, kernelLambda);
metrics = [metrics; metricRow(foldNumber, "equal_multikernel", ...
    "primary", "all", baseInfo.lambda, equalMetrics, baselineMetrics, ...
    validation, validationGroup, equalMetrics.prediction)];

weightedMetrics = predictAndScore(cellWeightedKernel, geneWeightedKernel, ...
    data, trainingRows, validationRows, trainingResidual, ...
    baselinePrediction, validation, kernelLambda);
metrics = [metrics; metricRow(foldNumber, "weighted_multikernel", ...
    "primary", "all", baseInfo.lambda, weightedMetrics, baselineMetrics, ...
    validation, validationGroup, weightedMetrics.prediction)];

if includeAblations
    metrics = [metrics; runAblations("cell", data.cellBlockNames, ...
        cellKernels, geneEqualKernel, data, foldNumber, trainingRows, ...
        validationRows, trainingResidual, baselinePrediction, validation, ...
        validationGroup, kernelLambda, baseInfo.lambda, baselineMetrics)];
    metrics = [metrics; runAblations("gene", data.geneBlockNames, ...
        geneKernels, cellEqualKernel, data, foldNumber, trainingRows, ...
        validationRows, trainingResidual, baselinePrediction, validation, ...
        validationGroup, kernelLambda, baseInfo.lambda, baselineMetrics)];
end

weights = weightRows(foldNumber, "cell", data.cellBlockNames, ...
    cellWeightResult);
weights = [weights; weightRows(foldNumber, "gene", ...
    data.geneBlockNames, geneWeightResult)];
bandwidths = bandwidthRows(foldNumber, "cell", ...
    data.cellBlockNames, cellKernelInfo);
bandwidths = [bandwidths; bandwidthRows(foldNumber, "gene", ...
    data.geneBlockNames, geneKernelInfo)];
fprintf("阶段8第%d组完成：阶段7核 %.6f，有权重多核 %.6f。\n", ...
    foldNumber, baselineMetrics.final_score, weightedMetrics.final_score);
end

function rows = runAblations(side, blockNames, variedKernels, fixedKernel, ...
    data, foldNumber, trainingRows, validationRows, trainingResidual, ...
    baselinePrediction, validation, validationGroup, kernelLambda, ...
    baseLambda, baselineMetrics)
nBlocks = numel(variedKernels);
rows = table;
for blockNumber = 1:nBlocks
    if side == "cell"
        cellKernel = variedKernels{blockNumber};
        geneKernel = fixedKernel;
    else
        cellKernel = fixedKernel;
        geneKernel = variedKernels{blockNumber};
    end
    result = predictAndScore(cellKernel, geneKernel, data, trainingRows, ...
        validationRows, trainingResidual, baselinePrediction, validation, ...
        kernelLambda);
    rows = [rows; metricRow(foldNumber, side + "_single", ...
        "single_block", blockNames(blockNumber), baseLambda, result, ...
        baselineMetrics, validation, validationGroup, result.prediction)]; %#ok<AGROW>

    kept = setdiff(1:nBlocks, blockNumber);
    leaveOneOutKernel = combineKernels(variedKernels(kept), ...
        ones(numel(kept), 1) / numel(kept));
    if side == "cell"
        cellKernel = leaveOneOutKernel;
    else
        geneKernel = leaveOneOutKernel;
    end
    result = predictAndScore(cellKernel, geneKernel, data, trainingRows, ...
        validationRows, trainingResidual, baselinePrediction, validation, ...
        kernelLambda);
    rows = [rows; metricRow(foldNumber, side + "_drop", ...
        "leave_one_out", blockNames(blockNumber), baseLambda, result, ...
        baselineMetrics, validation, validationGroup, result.prediction)]; %#ok<AGROW>
end
end

function result = predictAndScore(cellKernel, geneKernel, data, ...
    trainingRows, validationRows, trainingResidual, baselinePrediction, ...
    validation, kernelLambda)
model = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
    data.rowCellIndex(trainingRows), data.rowGeneIndex(trainingRows), ...
    trainingResidual, kernelLambda);
linearIndex = sub2ind(size(model.predictionMatrix), ...
    data.rowCellIndex(validationRows), data.rowGeneIndex(validationRows));
result.prediction = baselinePrediction + model.predictionMatrix(linearIndex);
score = stage7_calculate_metrics(validation, result.prediction);
result.final_score = score.final_score;
result.rho_macro = score.rho_macro;
result.ndcg_score = score.ndcg_score;
result.precision_score = score.precision_score;
result.rmse = score.rmse;
result.rmse_score = score.rmse_score;
end

function row = metricRow(foldNumber, route, comparison, block, ...
    baseLambda, result, baseline, validation, validationGroup, prediction)
newGene = validationGroup == "old_cell_new_gene" | ...
    validationGroup == "new_cell_new_gene";
if any(newGene)
    newGeneMetrics = stage7_calculate_metrics( ...
        validation(newGene, :), prediction(newGene));
    newGeneScore = newGeneMetrics.final_score;
    baselineNewGene = stage7_calculate_metrics(validation(newGene, :), ...
        baseline.prediction(newGene));
    baselineNewGeneScore = baselineNewGene.final_score;
else
    newGeneScore = NaN;
    baselineNewGeneScore = NaN;
end
row = table(foldNumber, string(route), string(comparison), string(block), ...
    0.1, baseLambda, result.final_score, result.rho_macro, ...
    result.ndcg_score, result.precision_score, result.rmse, ...
    result.rmse_score, baseline.final_score, ...
    result.final_score - baseline.final_score, newGeneScore, ...
    baselineNewGeneScore, newGeneScore - baselineNewGeneScore, ...
    'VariableNames', {'fold', 'route', 'comparison', 'block', ...
    'kernel_lambda', 'base_lambda', 'final_score', 'rho_macro', ...
    'ndcg_score', 'precision_score', 'rmse', 'rmse_score', ...
    'baseline_score', 'improvement', 'new_gene_score', ...
    'baseline_new_gene_score', 'new_gene_change'});
row.validation_rows = height(validation);
end

function [cellBlocks, geneBlocks, baseFeatures] = buildFoldFeatures( ...
    data, cellTrainingMask, geneTrainingMask)
cellCategory = encodeCategories(data.cellMetadata, ...
    ["OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"], ...
    cellTrainingMask);
geneCategory = encodeCategories(data.geneMetadata, ...
    ["curated_gene_list", "sub_mito_location", "mito_domain_score", ...
    "yeast_mito_homolog_score", "rickettsia_homolog_score", ...
    "msms_score"], geneTrainingMask);
geneEvidence = encodeDelimitedTokens( ...
    data.geneMetadata.curation_evidence, ",", geneTrainingMask);
cellBlocks = {data.cellExpression, data.cellPathway, ...
    cellCategory, data.cellModules};
geneBlocks = {data.geneNumeric, geneCategory, data.genePathway, ...
    geneEvidence, data.geneExpressionProfile};
baseFeatures = buildBaseGeneFeatures(data.geneMetadata, ...
    data.geneNumeric, geneTrainingMask);
end

function block = encodeCategories(source, fieldNames, trainingRows)
block = zeros(height(source), 0);
for fieldNumber = 1:numel(fieldNames)
    values = string(source.(fieldNames(fieldNumber)));
    values(ismissing(values) | strlength(strtrim(values)) == 0) = "<missing>";
    levels = unique(values(trainingRows), "sorted");
    for levelNumber = 2:numel(levels)
        block(:, end + 1) = double(values == levels(levelNumber)); %#ok<AGROW>
    end
end
end

function block = encodeDelimitedTokens(values, delimiter, trainingRows)
tokenRows = cell(numel(values), 1);
vocabulary = strings(0, 1);
for rowNumber = 1:numel(values)
    if ismissing(values(rowNumber)) || strlength(strtrim(values(rowNumber))) == 0
        tokenRows{rowNumber} = strings(0, 1);
    else
        tokens = lower(strtrim(split(values(rowNumber), delimiter)));
        tokens = tokens(strlength(tokens) > 0);
        tokenRows{rowNumber} = unique(tokens);
        if trainingRows(rowNumber)
            vocabulary = [vocabulary; tokens]; %#ok<AGROW>
        end
    end
end
vocabulary = unique(vocabulary, "sorted");
block = zeros(numel(values), numel(vocabulary));
for rowNumber = 1:numel(values)
    [known, location] = ismember(tokenRows{rowNumber}, vocabulary);
    block(rowNumber, location(known)) = 1;
end
end

function features = buildBaseGeneFeatures(geneRows, numericBlock, trainingRows)
for columnNumber = 1:size(numericBlock, 2)
    observed = numericBlock(trainingRows, columnNumber);
    observed = observed(isfinite(observed));
    if isempty(observed)
        fillValue = 0;
    else
        fillValue = median(observed);
    end
    numericBlock(~isfinite(numericBlock(:, columnNumber)), columnNumber) = ...
        fillValue;
end
center = mean(numericBlock(trainingRows, :), 1);
scale = std(numericBlock(trainingRows, :), 0, 1);
scale(scale < 1e-12 | ~isfinite(scale)) = 1;
features = (numericBlock - center) ./ scale;
categoryFields = ["sub_mito_location", "mito_domain_score", ...
    "yeast_mito_homolog_score", "rickettsia_homolog_score", "msms_score"];
features = [features, encodeCategories(geneRows, categoryFields, trainingRows)];
features(:, end + 1) = 0;
end

function [kernels, totalKernel, info] = buildKernelSet(blocks, trainingRows)
kernels = cell(1, numel(blocks));
processed = cell(1, numel(blocks));
info = repmat(struct("bandwidth", NaN, "trainingObjects", NaN, ...
    "featureCount", NaN, "standardized", true), numel(blocks), 1);
for blockNumber = 1:numel(blocks)
    [kernels{blockNumber}, info(blockNumber), processed{blockNumber}] = ...
        stage8_build_rbf_kernel(blocks{blockNumber}, trainingRows);
end
[totalKernel, ~] = stage8_build_rbf_kernel( ...
    cat(2, processed{:}), trainingRows, Standardize=false);
end

function [cellReference, geneReference] = buildResidualReferences( ...
    cellIndex, geneIndex, residual)
trainingCells = unique(cellIndex, "sorted");
trainingGenes = unique(geneIndex, "sorted");
[~, localCell] = ismember(cellIndex, trainingCells);
[~, localGene] = ismember(geneIndex, trainingGenes);
linearIndex = sub2ind([numel(trainingCells), numel(trainingGenes)], ...
    localCell, localGene);
assert(numel(unique(linearIndex)) == numel(linearIndex), ...
    "stage8:DuplicateTrainingPairs", "训练残差中存在重复细胞—基因组合。");
observed = false(numel(trainingCells), numel(trainingGenes));
observed(linearIndex) = true;
residualMatrix = nan(size(observed));
residualMatrix(linearIndex) = residual;
residualMatrix = fillMissingAdditive(residualMatrix, observed);
cellReference = residualMatrix * residualMatrix';
geneReference = residualMatrix' * residualMatrix;
end

function filled = fillMissingAdditive(values, observed)
globalMean = mean(values(observed));
cellMean = zeros(size(values, 1), 1);
geneMean = zeros(1, size(values, 2));
for cellNumber = 1:size(values, 1)
    cellMean(cellNumber) = mean(values(cellNumber, observed(cellNumber, :)));
end
for geneNumber = 1:size(values, 2)
    geneMean(geneNumber) = mean(values(observed(:, geneNumber), geneNumber));
end
filled = values;
additive = cellMean + geneMean - globalMean;
filled(~observed) = additive(~observed);
end

function kernel = combineKernels(kernels, weights)
assert(numel(kernels) == numel(weights), "stage8:CombineKernelCount", ...
    "核数量与权重数量不一致。");
kernel = zeros(size(kernels{1}));
for kernelNumber = 1:numel(kernels)
    kernel = kernel + weights(kernelNumber) * kernels{kernelNumber};
end
kernel = (kernel + kernel') / 2;
end

function rows = weightRows(foldNumber, side, blockNames, result)
n = numel(blockNames);
rows = table(repmat(foldNumber, n, 1), repmat(string(side), n, 1), ...
    blockNames(:), result.weights(:), result.alignment(:), ...
    repmat(result.ridge, n, 1), repmat(result.exitflag, n, 1), ...
    'VariableNames', {'fold', 'side', 'block', 'weight', ...
    'raw_alignment', 'alignment_ridge', 'solver_exitflag'});
end

function rows = bandwidthRows(foldNumber, side, blockNames, info)
n = numel(blockNames);
rows = table(repmat(foldNumber, n, 1), repmat(string(side), n, 1), ...
    blockNames(:), reshape([info.featureCount], [], 1), ...
    reshape([info.bandwidth], [], 1), ...
    reshape([info.trainingObjects], [], 1), ...
    'VariableNames', {'fold', 'side', 'block', 'feature_count', ...
    'rbf_bandwidth', 'training_objects'});
end

function gate = summarizeGate(metrics, weights, expectedFolds, stage)
baseline = sortrows(metrics(metrics.route == "stage7_total_kernel", :), "fold");
weighted = sortrows(metrics(metrics.route == "weighted_multikernel", :), "fold");
assert(height(baseline) == expectedFolds && height(weighted) == expectedFolds, ...
    "stage8:GateFoldCount", "停止线所需的主路线折数不完整。");
improvement = weighted.final_score - baseline.final_score;
newGeneChange = weighted.new_gene_score - baseline.new_gene_score;
spearmanChange = weighted.rho_macro - baseline.rho_macro;
[meanDeviation, cellTopConsistency, geneTopConsistency] = ...
    summarizeWeightStability(weights);
if expectedFolds == 3
    requiredWins = 2;
else
    requiredWins = ceil(0.75 * expectedFolds);
end
scoreGate = sum(improvement > 0) >= requiredWins && ...
    mean(improvement) >= 0.10;
if expectedFolds < 5
    stabilityGate = true;
    adopt = false;
else
    stabilityGate = meanDeviation >= 0.10 && ...
        cellTopConsistency >= 0.60 && geneTopConsistency >= 0.60;
    adopt = scoreGate && mean(newGeneChange) >= 0 && ...
        mean(spearmanChange) >= -0.003 && stabilityGate;
end
gate = table(string(stage), expectedFolds, sum(improvement > 0), ...
    mean(improvement), mean(newGeneChange), mean(spearmanChange), ...
    meanDeviation, cellTopConsistency, geneTopConsistency, ...
    scoreGate, stabilityGate, adopt, ...
    'VariableNames', {'stage', 'folds', 'folds_improved', ...
    'mean_improvement', 'mean_new_gene_change', 'mean_spearman_change', ...
    'mean_weight_l1_from_equal', 'cell_top_consistency', ...
    'gene_top_consistency', 'pass_score_gate', ...
    'pass_stability_gate', 'adopt_weighted_multikernel'});
end

function [meanDeviation, cellConsistency, geneConsistency] = ...
    summarizeWeightStability(weights)
folds = unique(weights.fold);
sides = ["cell", "gene"];
deviation = zeros(numel(folds) * numel(sides), 1);
cursor = 0;
cellTop = strings(numel(folds), 1);
geneTop = strings(numel(folds), 1);
for sideNumber = 1:numel(sides)
    for foldNumber = 1:numel(folds)
        subset = weights(weights.fold == folds(foldNumber) & ...
            weights.side == sides(sideNumber), :);
        cursor = cursor + 1;
        deviation(cursor) = sum(abs(subset.weight - 1 / height(subset)));
        [~, best] = max(subset.weight);
        if sides(sideNumber) == "cell"
            cellTop(foldNumber) = subset.block(best);
        else
            geneTop(foldNumber) = subset.block(best);
        end
    end
end
meanDeviation = mean(deviation);
cellConsistency = topConsistency(cellTop);
geneConsistency = topConsistency(geneTop);
end

function consistency = topConsistency(values)
levels = unique(values);
counts = zeros(numel(levels), 1);
for levelNumber = 1:numel(levels)
    counts(levelNumber) = sum(values == levels(levelNumber));
end
consistency = max(counts) / numel(values);
end

function [isTraining, isValidation, group] = makeInnerSplit(data, splitSeed)
rng(splitSeed, "twister");
outerTraining = ~data.isOuterValidation;
availableCells = unique(data.labels.cell_line_id(outerTraining));
availableGenes = unique(data.labels.perturbation_gene(outerTraining));
nHeldCells = max(1, round(0.10 * numel(availableCells)));
nHeldGenes = max(1, round(0.15 * numel(availableGenes)));
heldCells = availableCells(randperm(numel(availableCells), nHeldCells));
heldGenes = availableGenes(randperm(numel(availableGenes), nHeldGenes));
heldCellRow = ismember(data.labels.cell_line_id, heldCells);
heldGeneRow = ismember(data.labels.perturbation_gene, heldGenes);
oldOldCandidates = find(outerTraining & ~heldCellRow & ~heldGeneRow);
nHeldRows = max(1, round(0.05 * numel(oldOldCandidates)));
heldRows = oldOldCandidates(randperm(numel(oldOldCandidates), nHeldRows));
randomHeldRow = false(height(data.labels), 1);
randomHeldRow(heldRows) = true;
isValidation = outerTraining & (heldCellRow | heldGeneRow | randomHeldRow);
isTraining = outerTraining & ~isValidation;
group = strings(height(data.labels), 1);
group(~heldCellRow & ~heldGeneRow) = "old_cell_old_gene";
group(~heldCellRow & heldGeneRow) = "old_cell_new_gene";
group(heldCellRow & ~heldGeneRow) = "new_cell_old_gene";
group(heldCellRow & heldGeneRow) = "new_cell_new_gene";
group = group(isValidation);
end
