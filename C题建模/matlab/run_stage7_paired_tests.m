function run_stage7_paired_tests(seed, nBootstrap, nPermutation)
%RUN_STAGE7_PAIRED_TESTS 以细胞为配对单位检验阶段7改进是否稳定。

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
    nBootstrap (1, 1) double {mustBeInteger, mustBePositive} = 10000
    nPermutation (1, 1) double {mustBeInteger, mustBePositive} = 10000
end

scriptDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(scriptDir);
resultsDir = fullfile(projectDir, "results");
scoringDir = fullfile(resultsDir, "stage7_scoring");

answer = readtable(fullfile(scoringDir, "stage7_validation_answer.csv"), ...
    TextType="string", Delimiter=",");
baseline = readtable(fullfile(scoringDir, ...
    "prediction_stage4_hybrid_base.csv"), TextType="string", Delimiter=",");
winner = readtable(fullfile(scoringDir, ...
    "prediction_two_step_kernel_ridge.csv"), TextType="string", Delimiter=",");
groups = readtable(fullfile(scoringDir, ...
    "stage7_validation_groups.csv"), TextType="string", Delimiter=",");

assert(height(answer) == height(baseline) && height(answer) == height(winner), ...
    "stage7:PairedRowCount", "真值、底座和阶段7预测的行数不一致。");
assert(isequal(answer(:, 1:2), baseline(:, 1:2)) && ...
    isequal(answer(:, 1:2), winner(:, 1:2)) && ...
    isequal(answer(:, 1:2), groups(:, 1:2)), ...
    "stage7:PairedRowOrder", "四份评分文件的细胞—基因行顺序不一致。");

[cellGroup, cellIds] = findgroups(answer.cell_line_id);
nCells = numel(cellIds);
cellStats = calculateCellStatistics(answer, baseline.label, winner.label, ...
    cellGroup, cellIds);

baselineAggregate = aggregateStatistics(cellStats, (1:nCells)', false);
winnerAggregate = aggregateStatistics(cellStats, (1:nCells)', true);
observedImprovement = improvementVector(baselineAggregate, winnerAggregate);

% 这里用已有评分函数复核聚合写法，避免统计检验改变官方指标口径。
directBaseline = stage7_calculate_metrics(answer, baseline.label);
directWinner = stage7_calculate_metrics(answer, winner.label);
assert(abs(directBaseline.final_score - baselineAggregate(1)) < 1e-9 && ...
    abs(directWinner.final_score - winnerAggregate(1)) < 1e-9, ...
    "stage7:PairedMetricMismatch", "细胞聚合结果没有复现阶段7官方总分。");

rng(seed, "twister");
bootstrapImprovement = zeros(nBootstrap, 5);
for repeat = 1:nBootstrap
    sampledCells = randi(nCells, nCells, 1);
    sampledBaseline = aggregateStatistics(cellStats, sampledCells, false);
    sampledWinner = aggregateStatistics(cellStats, sampledCells, true);
    bootstrapImprovement(repeat, :) = ...
        improvementVector(sampledBaseline, sampledWinner);
end

permutationImprovement = zeros(nPermutation, 5);
for repeat = 1:nPermutation
    swapMask = rand(nCells, 1) < 0.5;
    [permutedBaseline, permutedWinner] = ...
        permuteCellStatistics(cellStats, swapMask);
    permutationImprovement(repeat, :) = ...
        improvementVector(permutedBaseline, permutedWinner);
end

metricNames = ["official_final_score"; "rho_macro"; "ndcg_score"; ...
    "precision_score"; "rmse_reduction"];
baselineValues = [baselineAggregate(1:4), baselineAggregate(5)]';
winnerValues = [winnerAggregate(1:4), winnerAggregate(5)]';
ciLower = prctile(bootstrapImprovement, 2.5, 1)';
ciUpper = prctile(bootstrapImprovement, 97.5, 1)';
probabilityPositive = mean(bootstrapImprovement > 0, 1)';
pOneSided = (1 + sum(permutationImprovement >= observedImprovement, 1))' ...
    / (nPermutation + 1);
pTwoSided = (1 + sum(abs(permutationImprovement) >= ...
    abs(observedImprovement), 1))' / (nPermutation + 1);
interpretation = buildInterpretation(metricNames, observedImprovement', ...
    ciLower, ciUpper);

summary = table(metricNames, baselineValues, winnerValues, ...
    observedImprovement', ciLower, ciUpper, probabilityPositive, ...
    pOneSided, pTwoSided, repmat(seed, 5, 1), ...
    repmat(nBootstrap, 5, 1), repmat(nPermutation, 5, 1), interpretation, ...
    VariableNames=["metric", "baseline_value", "winner_value", ...
    "observed_improvement", "bootstrap_ci_lower", "bootstrap_ci_upper", ...
    "bootstrap_probability_positive", "permutation_p_one_sided", ...
    "permutation_p_two_sided", "seed", "bootstrap_repeats", ...
    "permutation_repeats", "interpretation"]);
writetable(summary, fullfile(resultsDir, ...
    "stage7_paired_test_summary.csv"));

cellChanges = buildCellChangeTable(cellStats);
writetable(cellChanges, fullfile(resultsDir, ...
    "stage7_cell_metric_changes.csv"));

fprintf("阶段7配对检验完成：%d 个细胞，Bootstrap=%d，置换=%d。\n", ...
    nCells, nBootstrap, nPermutation);
fprintf("官方总分改进 %.6f，95%% Bootstrap 区间 [%.6f, %.6f]，单侧置换 p=%.6g。\n", ...
    observedImprovement(1), ciLower(1), ciUpper(1), pOneSided(1));
end

function stats = calculateCellStatistics(answer, baseline, winner, ...
    cellGroup, cellIds)
nCells = numel(cellIds);
stats.cell_id = cellIds;
stats.n_rows = zeros(nCells, 1);
stats.truth_sum = zeros(nCells, 1);
stats.truth_square_sum = zeros(nCells, 1);
stats.baseline_rho = zeros(nCells, 1);
stats.winner_rho = zeros(nCells, 1);
stats.baseline_ndcg = zeros(nCells, 1);
stats.winner_ndcg = zeros(nCells, 1);
stats.baseline_precision = zeros(nCells, 1);
stats.winner_precision = zeros(nCells, 1);
stats.baseline_sse = zeros(nCells, 1);
stats.winner_sse = zeros(nCells, 1);

for cellNumber = 1:nCells
    mask = cellGroup == cellNumber;
    truth = answer.label(mask);
    genes = answer.perturbation_gene(mask);
    basePrediction = baseline(mask);
    winnerPrediction = winner(mask);
    stats.n_rows(cellNumber) = numel(truth);
    stats.truth_sum(cellNumber) = sum(truth);
    stats.truth_square_sum(cellNumber) = sum(truth .^ 2);
    [stats.baseline_rho(cellNumber), stats.baseline_ndcg(cellNumber), ...
        stats.baseline_precision(cellNumber)] = ...
        calculateCellRanking(truth, basePrediction, genes);
    [stats.winner_rho(cellNumber), stats.winner_ndcg(cellNumber), ...
        stats.winner_precision(cellNumber)] = ...
        calculateCellRanking(truth, winnerPrediction, genes);
    stats.baseline_sse(cellNumber) = sum((basePrediction - truth) .^ 2);
    stats.winner_sse(cellNumber) = sum((winnerPrediction - truth) .^ 2);
end
end

function [rho, ndcg, precision] = calculateCellRanking(truth, prediction, genes)
kValues = [5, 10, 15];
nRows = numel(truth);
originalOrder = (1:nRows)';
predictedRank = tiedrank(-prediction);
trueRank = tiedrank(-truth);
rho = pearsonOrZero(predictedRank, trueRank);
[~, predictedOrder] = sortrows([-prediction, originalOrder], [1, 2]);
[~, trueOrder] = sortrows([-truth, originalOrder], [1, 2]);
precisionValues = zeros(1, numel(kValues));
ndcgValues = zeros(1, numel(kValues));

for kNumber = 1:numel(kValues)
    k = kValues(kNumber);
    take = min(k, nRows);
    predictedTop = predictedOrder(1:take);
    trueTopGenes = genes(trueOrder(1:take));
    precisionValues(kNumber) = ...
        sum(ismember(genes(predictedTop), trueTopGenes)) / k;
    relevance = max(k - trueRank(predictedTop) + 1, 0);
    relevance(trueRank(predictedTop) > k) = 0;
    discounts = log2((2:(take + 1))');
    idealRelevance = (k:-1:1)';
    idealDiscounts = log2((2:(k + 1))');
    ndcgValues(kNumber) = sum((2 .^ relevance - 1) ./ discounts) / ...
        sum((2 .^ idealRelevance - 1) ./ idealDiscounts);
end

precision = mean(precisionValues);
ndcg = mean(ndcgValues);
end

function values = aggregateStatistics(stats, selectedCells, useWinner)
if useWinner
    rho = stats.winner_rho(selectedCells);
    ndcg = stats.winner_ndcg(selectedCells);
    precision = stats.winner_precision(selectedCells);
    sse = stats.winner_sse(selectedCells);
else
    rho = stats.baseline_rho(selectedCells);
    ndcg = stats.baseline_ndcg(selectedCells);
    precision = stats.baseline_precision(selectedCells);
    sse = stats.baseline_sse(selectedCells);
end
values = aggregateArrays(stats, selectedCells, rho, ndcg, precision, sse);
end

function [baselineValues, winnerValues] = permuteCellStatistics(stats, swapMask)
nCells = numel(stats.n_rows);
selectedCells = (1:nCells)';
baselineRho = stats.baseline_rho;
winnerRho = stats.winner_rho;
baselineNdcg = stats.baseline_ndcg;
winnerNdcg = stats.winner_ndcg;
baselinePrecision = stats.baseline_precision;
winnerPrecision = stats.winner_precision;
baselineSse = stats.baseline_sse;
winnerSse = stats.winner_sse;

[baselineRho(swapMask), winnerRho(swapMask)] = ...
    deal(winnerRho(swapMask), baselineRho(swapMask));
[baselineNdcg(swapMask), winnerNdcg(swapMask)] = ...
    deal(winnerNdcg(swapMask), baselineNdcg(swapMask));
[baselinePrecision(swapMask), winnerPrecision(swapMask)] = ...
    deal(winnerPrecision(swapMask), baselinePrecision(swapMask));
[baselineSse(swapMask), winnerSse(swapMask)] = ...
    deal(winnerSse(swapMask), baselineSse(swapMask));

baselineValues = aggregateArrays(stats, selectedCells, baselineRho, ...
    baselineNdcg, baselinePrecision, baselineSse);
winnerValues = aggregateArrays(stats, selectedCells, winnerRho, ...
    winnerNdcg, winnerPrecision, winnerSse);
end

function values = aggregateArrays(stats, selectedCells, rho, ndcg, precision, sse)
nRows = sum(stats.n_rows(selectedCells));
truthSum = sum(stats.truth_sum(selectedCells));
truthSquareSum = sum(stats.truth_square_sum(selectedCells));
rhoMacro = mean(rho);
ndcgScore = mean(ndcg);
precisionScore = mean(precision);
rmse = sqrt(sum(sse) / nRows);
truthMean = truthSum / nRows;
labelVariance = max(truthSquareSum / nRows - truthMean ^ 2, 0);
labelScale = sqrt(labelVariance);
rmseScore = 1 / (1 + rmse / (labelScale + 1e-12));
finalScore = 100 * (0.30 * (rhoMacro + 1) / 2 + ...
    0.30 * ndcgScore + 0.25 * precisionScore + 0.15 * rmseScore);
values = [finalScore, rhoMacro, ndcgScore, precisionScore, rmse];
end

function improvement = improvementVector(baseline, winner)
improvement = winner - baseline;
% RMSE 越小越好，因此最后一列改写为“底座减新模型”。
improvement(5) = baseline(5) - winner(5);
end

function text = buildInterpretation(metricNames, observed, lower, upper)
text = strings(numel(metricNames), 1);
for index = 1:numel(metricNames)
    if lower(index) > 0
        text(index) = "细胞重采样后的95%区间仍为正，改进较稳定";
    elseif upper(index) < 0
        text(index) = "细胞重采样后的95%区间仍为负，该指标存在稳定下降";
    elseif observed(index) > 0
        text(index) = "平均值有所提高，但95%区间跨过0";
    else
        text(index) = "平均值有所下降，且95%区间跨过0";
    end
end
end

function output = buildCellChangeTable(stats)
baselineRmse = sqrt(stats.baseline_sse ./ stats.n_rows);
winnerRmse = sqrt(stats.winner_sse ./ stats.n_rows);
output = table(stats.cell_id, stats.n_rows, stats.baseline_rho, ...
    stats.winner_rho, stats.winner_rho - stats.baseline_rho, ...
    stats.baseline_ndcg, stats.winner_ndcg, ...
    stats.winner_ndcg - stats.baseline_ndcg, ...
    stats.baseline_precision, stats.winner_precision, ...
    stats.winner_precision - stats.baseline_precision, ...
    baselineRmse, winnerRmse, baselineRmse - winnerRmse, ...
    VariableNames=["cell_line_id", "n_pairs", "baseline_rho", ...
    "winner_rho", "delta_rho", "baseline_ndcg", "winner_ndcg", ...
    "delta_ndcg", "baseline_precision", "winner_precision", ...
    "delta_precision", "baseline_rmse", "winner_rmse", ...
    "rmse_reduction"]);
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
