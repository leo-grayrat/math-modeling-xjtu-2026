function run_stage7_full_information_regression(seed, runMode)
%RUN_STAGE7_FULL_INFORMATION_REGRESSION 比较两条全信息双边回归路线。
% smoke 只用真实数据的小矩形跑通两个模型；full 执行三组内层筛选。

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
    runMode (1, 1) string {mustBeMember(runMode, ["smoke", "full"])} = "full"
end

rng(seed, "twister");
matlabDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(matlabDir);
repoRoot = fileparts(projectDir);
resultsDir = fullfile(projectDir, "results");
scoringDir = fullfile(resultsDir, "stage7_scoring");
if ~isfolder(scoringDir)
    mkdir(scoringDir);
end

data = prepareFullInformation(repoRoot, projectDir, seed);
writeFeatureSummary(data, resultsDir);

if runMode == "smoke"
    runSmokeSlice(data, resultsDir, seed);
    return
end

fprintf("正在建立细胞和基因全信息核矩阵。\n");
cellKernel = buildRbfKernel(data.cellKernelFeatures);
geneKernel = buildRbfKernel(data.geneKernelFeatures);

bilinearLambdas = [0.1, 0.01, 0.001];
kernelLambdas = [0.1, 1, 10];
innerSummary = table;
solverSummary = table;

for foldNumber = 1:3
    [innerTraining, innerValidation] = makeInnerSplit( ...
        data, seed + 3000 + foldNumber);
    [foldBaseByGene, foldBaseInfo] = stage7_fit_fold_gene_base( ...
        data.baseGeneFeatures, data.rowGeneIndex(innerTraining), ...
        data.labels.label(innerTraining), seed + 5000 + foldNumber);
    foldBaseByRow = foldBaseByGene(data.rowGeneIndex);
    validationTable = data.labels(innerValidation, :);
    baselinePrediction = foldBaseByRow(innerValidation);
    baselineMetrics = stage7_calculate_metrics( ...
        validationTable, baselinePrediction);
    innerSummary = [innerSummary; metricRow(foldNumber, ...
        "fold_hybrid_ridge_base", foldBaseInfo.lambda, baselineMetrics, ...
        baselineMetrics.final_score)]; %#ok<AGROW>
    foldTrainingResidual = data.labels.label(innerTraining) - ...
        foldBaseByRow(innerTraining);

    initialPoint = [];
    for lambda = bilinearLambdas
        model = stage7_fit_bilinear_ridge( ...
            data.cellFeatures, data.geneFeatures, data.ownExpression, ...
            data.rowCellIndex(innerTraining), ...
            data.rowGeneIndex(innerTraining), ...
            foldTrainingResidual, lambda, ...
            Tolerance=2e-4, MaxIterations=300, ...
            InitialPoint=initialPoint);
        assert(model.solverFlag == 0, "stage7:BilinearInnerSolver", ...
            "双线性岭回归未在内层第 %d 组收敛，lambda=%.6g。", ...
            foldNumber, lambda);
        initialPoint = model.parameterVector;
        residualPrediction = stage7_predict_bilinear_ridge(model, ...
            data.cellFeatures, data.geneFeatures, data.ownExpression, ...
            data.rowCellIndex(innerValidation), ...
            data.rowGeneIndex(innerValidation));
        prediction = baselinePrediction + residualPrediction;
        metrics = stage7_calculate_metrics(validationTable, prediction);
        innerSummary = [innerSummary; metricRow(foldNumber, ...
            "bilinear_ridge", lambda, metrics, ...
            baselineMetrics.final_score)]; %#ok<AGROW>
        solverSummary = [solverSummary; bilinearSolverRow( ...
            foldNumber, "inner", lambda, model)]; %#ok<AGROW>
    end

    for lambda = kernelLambdas
        model = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
            data.rowCellIndex(innerTraining), ...
            data.rowGeneIndex(innerTraining), ...
            foldTrainingResidual, lambda);
        linearIndex = sub2ind(size(model.predictionMatrix), ...
            data.rowCellIndex(innerValidation), ...
            data.rowGeneIndex(innerValidation));
        prediction = baselinePrediction + model.predictionMatrix(linearIndex);
        metrics = stage7_calculate_metrics(validationTable, prediction);
        innerSummary = [innerSummary; metricRow(foldNumber, ...
            "two_step_kernel_ridge", lambda, metrics, ...
            baselineMetrics.final_score)]; %#ok<AGROW>
        solverSummary = [solverSummary; kernelSolverRow( ...
            foldNumber, "inner", lambda, model)]; %#ok<AGROW>
    end

    fprintf("内层第 %d 组完成，基线分数 %.6f。\n", ...
        foldNumber, baselineMetrics.final_score);
end

writetable(innerSummary, fullfile(resultsDir, ...
    "stage7_inner_screening.csv"));
[routeSummary, selection] = selectInnerWinner(innerSummary);
writetable(routeSummary, fullfile(resultsDir, ...
    "stage7_route_selection.csv"));

if ~selection.outerEligible
    writetable(selection, fullfile(resultsDir, ...
        "stage7_selected_model.csv"));
    writetable(solverSummary, fullfile(resultsDir, ...
        "stage7_solver_summary.csv"));
    fprintf("两条路线均未在至少两组内层划分中超过基线，不读取外层标签。\n");
    return
end

outerTraining = ~data.isOuterValidation;
outerValidation = data.isOuterValidation;
[outerBaseByGene, ~] = stage7_fit_fold_gene_base( ...
    data.baseGeneFeatures, data.rowGeneIndex(outerTraining), ...
    data.labels.label(outerTraining), seed + 1000);
outerBaseByRow = outerBaseByGene(data.rowGeneIndex);
outerTrainingResidual = data.labels.label(outerTraining) - ...
    outerBaseByRow(outerTraining);
if selection.route == "bilinear_ridge"
    finalModel = stage7_fit_bilinear_ridge( ...
        data.cellFeatures, data.geneFeatures, data.ownExpression, ...
        data.rowCellIndex(outerTraining), data.rowGeneIndex(outerTraining), ...
        outerTrainingResidual, selection.lambda, ...
        Tolerance=1e-4, MaxIterations=300);
    assert(finalModel.solverFlag == 0, "stage7:BilinearOuterSolver", ...
        "双线性岭回归未在外层训练中收敛，lambda=%.6g。", ...
        selection.lambda);
    residualPrediction = stage7_predict_bilinear_ridge(finalModel, ...
        data.cellFeatures, data.geneFeatures, data.ownExpression, ...
        data.rowCellIndex(outerValidation), ...
        data.rowGeneIndex(outerValidation));
    solverSummary = [solverSummary; bilinearSolverRow( ...
        0, "outer", selection.lambda, finalModel)];
else
    finalModel = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
        data.rowCellIndex(outerTraining), data.rowGeneIndex(outerTraining), ...
        outerTrainingResidual, selection.lambda);
    linearIndex = sub2ind(size(finalModel.predictionMatrix), ...
        data.rowCellIndex(outerValidation), ...
        data.rowGeneIndex(outerValidation));
    residualPrediction = finalModel.predictionMatrix(linearIndex);
    solverSummary = [solverSummary; kernelSolverRow( ...
        0, "outer", selection.lambda, finalModel)];
end

baselinePrediction = outerBaseByRow(outerValidation);
winnerPrediction = baselinePrediction + residualPrediction;
% 只有路线已经通过内层门槛后，才读取外层验证标签用于最终评分文件。
validation = data.labels(outerValidation, :);
validationGroup = data.outerGroup(outerValidation);
writeScoringFiles(scoringDir, validation, validationGroup, ...
    baselinePrediction, winnerPrediction, selection.route);

selection.outerPredictionWritten = true;
writetable(selection, fullfile(resultsDir, "stage7_selected_model.csv"));
writetable(solverSummary, fullfile(resultsDir, "stage7_solver_summary.csv"));
fprintf("内层胜出路线为 %s，lambda=%.6g。外层预测已写入 %s。\n", ...
    selection.route, selection.lambda, scoringDir);
end

function data = prepareFullInformation(repoRoot, projectDir, seed)
dataRoot = fullfile(repoRoot, "2026年校赛题目", "C", "数据文件");
processedDir = fullfile(projectDir, "data_processed");
resultsDir = fullfile(projectDir, "results");

labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");
expression = readtable(fullfile(dataRoot, "features", ...
    "cell_expression_zscore.csv"), TextType="string", ...
    VariableNamingRule="preserve");
pathways = readtable(fullfile(dataRoot, "features", ...
    "cell_pathway_scores_zscore.csv"), TextType="string", ...
    VariableNamingRule="preserve");
cellMetadata = readtable(fullfile(dataRoot, "metadata", ...
    "cell_line_metadata.csv"), TextType="string", ...
    VariableNamingRule="preserve");
geneMetadata = readtable(fullfile(dataRoot, "metadata", ...
    "gene_metadata.csv"), TextType="string", ...
    VariableNamingRule="preserve");
moduleScores = readtable(fullfile(resultsDir, ...
    "stage3_cell_module_scores.csv"), TextType="string", ...
    VariableNamingRule="preserve");

cellNames = sort(unique(labels.cell_line_id));
geneNames = sort(unique(labels.perturbation_gene));
[knownRowCells, rowCellIndex] = ismember(labels.cell_line_id, cellNames);
[knownRowGenes, rowGeneIndex] = ismember(labels.perturbation_gene, geneNames);
assert(all(knownRowCells) && all(knownRowGenes), ...
    "stage7:LabelMapping", "标签行无法映射到细胞或基因名单。");

[isOuterValidation, outerGroup, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed);
checkPersistedSplit(processedDir, heldCells, heldGenes);
outerTraining = ~isOuterValidation;
trainingCells = unique(labels.cell_line_id(outerTraining));
trainingGenes = unique(labels.perturbation_gene(outerTraining));
cellTrainingMask = ismember(cellNames, trainingCells);
geneTrainingMask = ismember(geneNames, trainingGenes);

[knownExpressionCells, expressionCellIndex] = ...
    ismember(cellNames, expression.cell_line_id);
[knownPathwayCells, pathwayCellIndex] = ...
    ismember(cellNames, pathways.cell_line_id);
[knownCellMetadata, cellMetadataIndex] = ...
    ismember(cellNames, cellMetadata.cell_line_id);
[knownModules, moduleIndex] = ismember(cellNames, moduleScores.cell_line_id);
assert(all(knownExpressionCells & knownPathwayCells & ...
    knownCellMetadata & knownModules), ...
    "stage7:CellCoverage", "部分标签细胞缺少完整特征。");

expressionGeneNames = string(expression.Properties.VariableNames(2:end));
[knownExpressionGenes, expressionGeneIndex] = ...
    ismember(geneNames, expressionGeneNames);
assert(all(knownExpressionGenes), ...
    "stage7:ExpressionGeneCoverage", ...
    "部分待敲基因没有对应的细胞表达列。");

expressionBlock = expression{expressionCellIndex, 2:end};
pathwayBlock = pathways{pathwayCellIndex, 2:end};
moduleBlock = moduleScores{moduleIndex, 2:end};
cellCategoryBlock = encodeCategories(cellMetadata(cellMetadataIndex, :), ...
    ["OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]);

expressionBlock = standardizeAndBalance(expressionBlock, cellTrainingMask);
pathwayBlock = standardizeAndBalance(pathwayBlock, cellTrainingMask);
moduleBlock = standardizeAndBalance(moduleBlock, cellTrainingMask);
cellCategoryBlock = standardizeAndBalance( ...
    cellCategoryBlock, cellTrainingMask);
cellKernelFeatures = [expressionBlock, pathwayBlock, ...
    cellCategoryBlock, moduleBlock];
cellFeatures = [cellKernelFeatures, ones(numel(cellNames), 1)];

[knownGeneMetadata, geneMetadataIndex] = ...
    ismember(geneNames, geneMetadata.gene_symbol);
assert(all(knownGeneMetadata), ...
    "stage7:GeneMetadataCoverage", "部分标签基因缺少元数据。");
geneRows = geneMetadata(geneMetadataIndex, :);
geneNumericBlock = buildGeneNumericBlock(geneRows);
baseGeneFeatures = buildStage4BaseFeatures( ...
    geneRows, geneNumericBlock, geneTrainingMask);
geneCategoryBlock = encodeCategories(geneRows, ...
    ["curated_gene_list", "sub_mito_location", "mito_domain_score", ...
    "yeast_mito_homolog_score", "rickettsia_homolog_score", "msms_score"]);
pathwayNames = string(pathways.Properties.VariableNames(2:end));
genePathwayBlock = build_gene_pathway_membership( ...
    string(geneRows.pathways), pathwayNames);
geneEvidenceBlock = encodeDelimitedTokens(geneRows.curation_evidence, ",");

geneNumericBlock = standardizeAndBalance( ...
    geneNumericBlock, geneTrainingMask);
geneCategoryBlock = standardizeAndBalance( ...
    geneCategoryBlock, geneTrainingMask);
genePathwayBlock = standardizeAndBalance( ...
    genePathwayBlock, geneTrainingMask);
geneEvidenceBlock = standardizeAndBalance( ...
    geneEvidenceBlock, geneTrainingMask);
geneSideFeatures = [geneNumericBlock, geneCategoryBlock, ...
    genePathwayBlock, geneEvidenceBlock];
geneFeatures = [geneSideFeatures, ones(numel(geneNames), 1)];

ownExpression = expression{expressionCellIndex, expressionGeneIndex + 1};
geneExpressionProfile = ownExpression' / sqrt(size(ownExpression, 1));
geneKernelFeatures = [geneSideFeatures, geneExpressionProfile];

pairIndex = sub2ind([numel(cellNames), numel(geneNames)], ...
    rowCellIndex, rowGeneIndex);
assert(numel(unique(pairIndex)) == numel(pairIndex), ...
    "stage7:DuplicateLabels", "标签表中存在重复细胞—基因组合。");

data.labels = labels;
data.cellNames = cellNames;
data.geneNames = geneNames;
data.rowCellIndex = rowCellIndex;
data.rowGeneIndex = rowGeneIndex;
data.isOuterValidation = isOuterValidation;
data.outerGroup = outerGroup;
data.cellFeatures = cellFeatures;
data.geneFeatures = geneFeatures;
data.baseGeneFeatures = baseGeneFeatures;
data.cellKernelFeatures = cellKernelFeatures;
data.geneKernelFeatures = geneKernelFeatures;
data.ownExpression = ownExpression;
data.featureBlock = ["cell_expression"; "cell_pathway"; ...
    "cell_metadata"; "cell_modules"; "gene_numeric"; ...
    "gene_categories"; "gene_pathways"; "gene_evidence"; ...
    "own_expression"; "stage4_base_features"];
data.featureCount = [size(expressionBlock, 2); size(pathwayBlock, 2); ...
    size(cellCategoryBlock, 2); size(moduleBlock, 2); ...
    size(geneNumericBlock, 2); size(geneCategoryBlock, 2); ...
    size(genePathwayBlock, 2); size(geneEvidenceBlock, 2); 1; ...
    size(baseGeneFeatures, 2)];
end

function block = buildGeneNumericBlock(geneRows)
block = [geneRows.targetp_score, geneRows.coexpression_gnf_n50_score, ...
    geneRows.pgc_induction_score, ...
    countTokens(geneRows.curation_evidence, ","), ...
    countTokens(geneRows.pathways, "|"), ...
    countTokens(geneRows.synonyms, "|"), ...
    double(strlength(geneRows.description))];
end

function features = buildStage4BaseFeatures( ...
    geneRows, numericBlock, trainingRows)
% 严格沿用阶段4的数值标准化和五类元数据编码。
for columnNumber = 1:size(numericBlock, 2)
    observed = numericBlock(trainingRows, columnNumber);
    observed = observed(isfinite(observed));
    if isempty(observed)
        fillValue = 0;
    else
        fillValue = median(observed);
    end
    missing = ~isfinite(numericBlock(:, columnNumber));
    numericBlock(missing, columnNumber) = fillValue;
end
center = mean(numericBlock(trainingRows, :), 1);
scale = std(numericBlock(trainingRows, :), 0, 1);
scale(scale < 1e-12 | ~isfinite(scale)) = 1;
features = (numericBlock - center) ./ scale;

categoryFields = ["sub_mito_location", "mito_domain_score", ...
    "yeast_mito_homolog_score", "rickettsia_homolog_score", "msms_score"];
for fieldNumber = 1:numel(categoryFields)
    values = string(geneRows.(categoryFields(fieldNumber)));
    values(ismissing(values) | strlength(strtrim(values)) == 0) = "<missing>";
    levels = unique(values(trainingRows), "stable");
    for levelNumber = 2:numel(levels)
        features(:, end + 1) = ...
            double(values == levels(levelNumber)); %#ok<AGROW>
    end
end

% 当前数据的基因元数据完整，仍保留阶段4同名的全零缺失指示列。
features(:, end + 1) = 0;
end

function counts = countTokens(values, delimiter)
counts = zeros(numel(values), 1);
for rowNumber = 1:numel(values)
    value = values(rowNumber);
    if ismissing(value) || strlength(strtrim(value)) == 0 || value == "-"
        counts(rowNumber) = 0;
    else
        counts(rowNumber) = numel(split(value, delimiter));
    end
end
end

function block = encodeCategories(source, fieldNames)
block = zeros(height(source), 0);
for fieldNumber = 1:numel(fieldNames)
    values = string(source.(fieldNames(fieldNumber)));
    values(ismissing(values) | strlength(strtrim(values)) == 0) = "<missing>";
    levels = unique(values, "sorted");
    for levelNumber = 2:numel(levels)
        block(:, end + 1) = double(values == levels(levelNumber)); %#ok<AGROW>
    end
end
end

function block = encodeDelimitedTokens(values, delimiter)
tokenRows = cell(numel(values), 1);
vocabulary = strings(0, 1);
for rowNumber = 1:numel(values)
    if ismissing(values(rowNumber)) || strlength(strtrim(values(rowNumber))) == 0
        tokenRows{rowNumber} = strings(0, 1);
    else
        tokens = lower(strtrim(split(values(rowNumber), delimiter)));
        tokens = tokens(strlength(tokens) > 0);
        tokenRows{rowNumber} = unique(tokens);
        vocabulary = [vocabulary; tokens]; %#ok<AGROW>
    end
end
vocabulary = unique(vocabulary, "sorted");
block = zeros(numel(values), numel(vocabulary));
for rowNumber = 1:numel(values)
    [known, location] = ismember(tokenRows{rowNumber}, vocabulary);
    block(rowNumber, location(known)) = 1;
end
end

function block = standardizeAndBalance(block, trainingRows)
if isempty(block)
    return
end
for columnNumber = 1:size(block, 2)
    observed = block(trainingRows, columnNumber);
    observed = observed(isfinite(observed));
    if isempty(observed)
        fillValue = 0;
    else
        fillValue = median(observed);
    end
    missing = ~isfinite(block(:, columnNumber));
    block(missing, columnNumber) = fillValue;
end
center = mean(block(trainingRows, :), 1);
scale = std(block(trainingRows, :), 0, 1);
scale(scale < 1e-12 | ~isfinite(scale)) = 1;
block = (block - center) ./ scale;
block = block / sqrt(size(block, 2));
end

function kernel = buildRbfKernel(features)
squaredDistance = pdist2(features, features, "squaredeuclidean");
upperValues = squaredDistance(triu(true(size(squaredDistance)), 1));
positiveValues = upperValues(upperValues > 1e-12 & isfinite(upperValues));
if isempty(positiveValues)
    bandwidth = 1;
else
    bandwidth = median(positiveValues);
end
kernel = exp(-squaredDistance / (2 * bandwidth));
kernel = (kernel + kernel') / 2;
kernel(1:size(kernel, 1) + 1:end) = 1;
end

function [isTraining, isValidation] = makeInnerSplit(data, splitSeed)
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
end

function runSmokeSlice(data, resultsDir, seed)
rng(seed + 7000, "twister");
outerTraining = ~data.isOuterValidation;
availableCells = unique(data.rowCellIndex(outerTraining), "sorted");
availableGenes = unique(data.rowGeneIndex(outerTraining), "sorted");
selectedCells = availableCells(1:min(60, numel(availableCells)));
selectedGenes = availableGenes(1:min(80, numel(availableGenes)));
candidateRows = outerTraining & ismember(data.rowCellIndex, selectedCells) & ...
    ismember(data.rowGeneIndex, selectedGenes);
heldCells = selectedCells(end - 9:end);
heldGenes = selectedGenes(end - 14:end);
validationRows = candidateRows & (ismember(data.rowCellIndex, heldCells) | ...
    ismember(data.rowGeneIndex, heldGenes));
trainingRows = candidateRows & ~validationRows;

cellMap = zeros(size(data.cellNames));
cellMap(selectedCells) = 1:numel(selectedCells);
geneMap = zeros(size(data.geneNames));
geneMap(selectedGenes) = 1:numel(selectedGenes);
localTrainingCell = cellMap(data.rowCellIndex(trainingRows));
localTrainingGene = geneMap(data.rowGeneIndex(trainingRows));
localValidationCell = cellMap(data.rowCellIndex(validationRows));
localValidationGene = geneMap(data.rowGeneIndex(validationRows));
cellFeatures = data.cellFeatures(selectedCells, :);
geneFeatures = data.geneFeatures(selectedGenes, :);
ownExpression = data.ownExpression(selectedCells, selectedGenes);
validationTable = data.labels(validationRows, :);
[sliceBaseByGene, sliceBaseInfo] = stage7_fit_fold_gene_base( ...
    data.baseGeneFeatures, data.rowGeneIndex(trainingRows), ...
    data.labels.label(trainingRows), seed + 8000);
sliceBaseByRow = sliceBaseByGene(data.rowGeneIndex);
baselinePrediction = sliceBaseByRow(validationRows);
sliceTrainingResidual = data.labels.label(trainingRows) - ...
    sliceBaseByRow(trainingRows);

bilinearModel = stage7_fit_bilinear_ridge(cellFeatures, geneFeatures, ...
    ownExpression, localTrainingCell, localTrainingGene, ...
    sliceTrainingResidual, 0.01, ...
    Tolerance=1e-3, MaxIterations=20);
bilinearPrediction = baselinePrediction + ...
    stage7_predict_bilinear_ridge(bilinearModel, cellFeatures, ...
    geneFeatures, ownExpression, localValidationCell, localValidationGene);

cellKernel = buildRbfKernel(data.cellKernelFeatures(selectedCells, :));
geneKernel = buildRbfKernel(data.geneKernelFeatures(selectedGenes, :));
kernelModel = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
    localTrainingCell, localTrainingGene, ...
    sliceTrainingResidual, 1);
kernelLinearIndex = sub2ind(size(kernelModel.predictionMatrix), ...
    localValidationCell, localValidationGene);
kernelPrediction = baselinePrediction + ...
    kernelModel.predictionMatrix(kernelLinearIndex);

baselineMetrics = stage7_calculate_metrics( ...
    validationTable, baselinePrediction);
bilinearMetrics = stage7_calculate_metrics( ...
    validationTable, bilinearPrediction);
kernelMetrics = stage7_calculate_metrics( ...
    validationTable, kernelPrediction);
summary = [metricRow(0, "fold_hybrid_ridge_base", sliceBaseInfo.lambda, ...
    baselineMetrics, baselineMetrics.final_score); ...
    metricRow(0, "bilinear_ridge", 0.01, bilinearMetrics, ...
    baselineMetrics.final_score); ...
    metricRow(0, "two_step_kernel_ridge", 1, kernelMetrics, ...
    baselineMetrics.final_score)];
summary.training_rows(:) = sum(trainingRows);
summary.validation_rows(:) = sum(validationRows);
writetable(summary, fullfile(resultsDir, "stage7_smoke_summary.csv"));

assert(all(isfinite(summary.final_score)), ...
    "stage7:SmokeFinite", "冒烟测试产生了非有限指标。");
assert(bilinearModel.solverFlag <= 1, ...
    "stage7:SmokeSolver", "双线性求解器在冒烟测试中未收敛。");
fprintf("阶段7真实数据冒烟测试完成：训练 %d 行，验证 %d 行。\n", ...
    sum(trainingRows), sum(validationRows));
end

function row = metricRow(foldNumber, route, lambda, metrics, baselineScore)
row = table(foldNumber, route, lambda, metrics.final_score, ...
    metrics.rho_macro, metrics.ndcg_score, metrics.precision_score, ...
    metrics.rmse, metrics.rmse_score, baselineScore, ...
    'VariableNames', {'fold', 'route', 'lambda', 'final_score', ...
    'rho_macro', 'ndcg_score', 'precision_score', 'rmse', ...
    'rmse_score', 'baseline_score'});
end

function row = bilinearSolverRow(foldNumber, scope, lambda, model)
row = table(foldNumber, scope, "bilinear_ridge", lambda, ...
    model.solverFlag, model.relativeResidual, model.iterations, ...
    model.elapsedSeconds, norm(model.interaction, "fro"), ...
    model.ownExpressionCoefficient, NaN, ...
    'VariableNames', {'fold', 'scope', 'route', 'lambda', ...
    'solver_flag', 'relative_residual', 'iterations', ...
    'elapsed_seconds', 'interaction_norm', ...
    'own_expression_coefficient', 'observed_fraction'});
end

function row = kernelSolverRow(foldNumber, scope, lambda, model)
row = table(foldNumber, scope, "two_step_kernel_ridge", lambda, ...
    NaN, NaN, NaN, NaN, NaN, NaN, model.observedFraction, ...
    'VariableNames', {'fold', 'scope', 'route', 'lambda', ...
    'solver_flag', 'relative_residual', 'iterations', ...
    'elapsed_seconds', 'interaction_norm', ...
    'own_expression_coefficient', 'observed_fraction'});
end

function [routeSummary, selection] = selectInnerWinner(innerSummary)
candidate = innerSummary(innerSummary.route ~= "fold_hybrid_ridge_base", :);
[group, route, lambda] = findgroups(candidate.route, candidate.lambda);
meanScore = splitapply(@mean, candidate.final_score, group);
meanBaseline = splitapply(@mean, candidate.baseline_score, group);
foldsImproved = splitapply(@(score, baseline) sum(score > baseline), ...
    candidate.final_score, candidate.baseline_score, group);
routeSummary = table(route, lambda, meanScore, meanBaseline, ...
    meanScore - meanBaseline, foldsImproved, ...
    'VariableNames', {'route', 'lambda', 'mean_final_score', ...
    'mean_baseline_score', 'mean_improvement', 'folds_improved'});

eligible = routeSummary.folds_improved >= 2;
if any(eligible)
    eligibleRows = find(eligible);
    [~, bestLocal] = max(routeSummary.mean_final_score(eligible));
    bestRow = eligibleRows(bestLocal);
    selection = routeSummary(bestRow, :);
    selection.outerEligible = true;
else
    [~, bestRow] = max(routeSummary.mean_final_score);
    selection = routeSummary(bestRow, :);
    selection.outerEligible = false;
end
selection.outerPredictionWritten = false;
end

function writeFeatureSummary(data, resultsDir)
summary = table(data.featureBlock, data.featureCount, ...
    'VariableNames', {'feature_block', 'feature_count'});
writetable(summary, fullfile(resultsDir, "stage7_feature_summary.csv"));
end

function writeScoringFiles(scoringDir, validation, validationGroup, ...
    baselinePrediction, winnerPrediction, winnerName)
answer = validation(:, ["cell_line_id", "perturbation_gene", "label"]);
writetable(answer, fullfile(scoringDir, "stage7_validation_answer.csv"));
groupTable = validation(:, ["cell_line_id", "perturbation_gene"]);
groupTable.group = validationGroup;
writetable(groupTable, fullfile(scoringDir, "stage7_validation_groups.csv"));

baseline = validation(:, ["cell_line_id", "perturbation_gene"]);
baseline.label = baselinePrediction;
writetable(baseline, fullfile(scoringDir, ...
    "prediction_stage4_hybrid_base.csv"));
winner = validation(:, ["cell_line_id", "perturbation_gene"]);
winner.label = winnerPrediction;
writetable(winner, fullfile(scoringDir, ...
    "prediction_" + winnerName + ".csv"));
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
savedCells = readtable(fullfile(processedDir, ...
    "stage1_heldout_cells.csv"), TextType="string");
savedGenes = readtable(fullfile(processedDir, ...
    "stage1_heldout_genes.csv"), TextType="string");
assert(isequal(sort(heldCells), sort(savedCells.cell_line_id)), ...
    "stage7:HeldCells", "留出细胞名单与阶段1不一致。");
assert(isequal(sort(heldGenes), sort(savedGenes.perturbation_gene)), ...
    "stage7:HeldGenes", "留出基因名单与阶段1不一致。");
end
