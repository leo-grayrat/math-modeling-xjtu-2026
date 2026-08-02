function run_stage5_cell_correction(seed)
% 阶段5：利用细胞功能模块修正同一基因在不同细胞中的差异。

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
scoringDir = fullfile(resultsDir, "stage5_scoring");
if ~isfolder(scoringDir)
    mkdir(scoringDir);
end

labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");
moduleScores = readtable(fullfile(resultsDir, "stage3_cell_module_scores.csv"), ...
    TextType="string", VariableNamingRule="preserve");
geneBaseScores = readtable(fullfile(resultsDir, "stage4_gene_base_scores.csv"), ...
    TextType="string", VariableNamingRule="preserve");

[isValidation, validationGroup, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed);
checkPersistedSplit(processedDir, heldCells, heldGenes);

[knownCells, cellLocation] = ismember(labels.cell_line_id, moduleScores.cell_line_id);
[knownGenes, geneLocation] = ismember(labels.perturbation_gene, ...
    geneBaseScores.perturbation_gene);
assert(all(knownCells), "存在没有细胞模块分数的标签行。");
assert(all(knownGenes), "存在没有基因基础分的标签行。");

modules = moduleScores{cellLocation, 2:end};
geneBase = geneBaseScores.hybrid_ridge_base(geneLocation);

training = labels(~isValidation, :);
validation = labels(isValidation, :);
validationGroup = validationGroup(isValidation);
trainingRows = ~isValidation;

moduleCounts = [8, 8, 7];
interactionFlags = [false, true, true];
modelNames = ["stage4_hybrid_base"; "cell_additive_ridge"; ...
    "cell_gene_interaction_ridge"; "interaction_without_module8"];
modelFeatures = cell(3, 1);
modelFeatureNames = cell(3, 1);
for i = 1:3
    [modelFeatures{i}, modelFeatureNames{i}] = buildDesign( ...
        geneBase, modules, moduleCounts(i), interactionFlags(i));
end

% 内层验证仍然按细胞和基因留出，避免把同一对象的标签同时用于调参与评价。
innerValidation = reproduceInnerSplit(training, seed + 2000);
innerTraining = ~innerValidation;
lambdaGrid = logspace(-4, 4, 25);

predictions = nan(height(validation), numel(modelNames));
predictions(:, 1) = geneBase(isValidation);
coefficientRows = strings(0, 2);
coefficientValues = zeros(0, 2);
summaryRows = strings(0, 1);
summaryGroups = strings(0, 1);
summaryValues = zeros(0, 9);
selectedLambdas = nan(numel(modelNames), 1);
innerMses = nan(numel(modelNames), 1);
featureCounts = zeros(numel(modelNames), 1);
nonzeroCounts = zeros(numel(modelNames), 1);

for modelIndex = 2:numel(modelNames)
    featureIndex = modelIndex - 1;
    X = modelFeatures{featureIndex};
    [lambda, innerMse] = selectLambda(X(trainingRows, :), training.label, ...
        innerTraining, innerValidation, lambdaGrid);
    [XzTrain, center, scale] = standardizeByTraining(X(trainingRows, :));
    XzAll = (X - center) ./ scale;
    [beta, intercept] = fitRidge(XzTrain, training.label, lambda);
    predictions(:, modelIndex) = intercept + XzAll(isValidation, :) * beta;

    coefficientRows(end + 1, 1) = modelNames(modelIndex);
    coefficientRows(end, 2) = "截距";
    coefficientValues(end + 1, :) = [intercept, lambda];
    for coefficientIndex = 1:numel(beta)
        coefficientRows(end + 1, 1) = modelNames(modelIndex);
        coefficientRows(end, 2) = modelFeatureNames{featureIndex}(coefficientIndex);
        coefficientValues(end + 1, :) = [beta(coefficientIndex), lambda];
    end

    selectedLambdas(modelIndex) = lambda;
    innerMses(modelIndex) = innerMse;
    featureCounts(modelIndex) = numel(beta);
    nonzeroCounts(modelIndex) = sum(abs(beta) > 1e-8);
end

% 第一列是阶段4的冷启动基础分，不需要重新训练。
for modelIndex = 1:numel(modelNames)
    for groupName = ["all"; "old_cell_old_gene"; "old_cell_new_gene"; ...
            "new_cell_old_gene"; "new_cell_new_gene"]'
        if groupName == "all"
            mask = true(height(validation), 1);
        else
            mask = validationGroup == groupName;
        end
        truth = validation.label(mask);
        prediction = predictions(mask, modelIndex);
        rho = corr(truth, prediction, "Type", "Spearman", "Rows", "complete");
        summaryRows(end + 1, 1) = modelNames(modelIndex);
        summaryGroups(end + 1, 1) = groupName;
        summaryValues(end + 1, :) = [modelIndex, selectedLambdas(modelIndex), ...
            innerMses(modelIndex), featureCounts(modelIndex), ...
            nonzeroCounts(modelIndex), numel(truth), ...
            sqrt(mean((prediction - truth).^2)), mean(abs(prediction - truth)), rho];
    end
end

modelSummary = table(summaryRows, summaryGroups, summaryValues(:, 1), ...
    summaryValues(:, 2), summaryValues(:, 3), summaryValues(:, 4), ...
    summaryValues(:, 5), summaryValues(:, 6), summaryValues(:, 7), ...
    summaryValues(:, 8), summaryValues(:, 9), ...
    'VariableNames', {'model', 'group', 'model_index', 'lambda', 'inner_mse', ...
    'feature_count', 'nonzero_coefficients', 'rows', 'rmse', 'mae', 'spearman'});
writetable(modelSummary, fullfile(resultsDir, ...
    "stage5_cell_correction_summary.csv"));

coefficientTable = table(coefficientRows(:, 1), coefficientRows(:, 2), ...
    coefficientValues(:, 1), coefficientValues(:, 2), ...
    'VariableNames', {'model', 'feature', 'coefficient', 'lambda'});
writetable(coefficientTable, fullfile(resultsDir, ...
    "stage5_cell_correction_coefficients.csv"));

for modelIndex = 1:numel(modelNames)
    submission = validation(:, ["cell_line_id", "perturbation_gene"]);
    submission.label = predictions(:, modelIndex);
    writetable(submission, fullfile(scoringDir, ...
        "prediction_" + modelNames(modelIndex) + ".csv"));
end

fprintf("阶段5完成。训练行：%d；验证行：%d。\n", height(training), height(validation));
fprintf("结果已写入 %s\n", resultsDir);
end

function [isValidation, group, heldCells, heldGenes] = reproduceStage1Split(labels, seed)
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
heldOldOldRows = oldOldCandidates(randperm(numel(oldOldCandidates), nHeldOldOldRows));
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
savedCells = readtable(fullfile(processedDir, "stage1_heldout_cells.csv"), TextType="string");
savedGenes = readtable(fullfile(processedDir, "stage1_heldout_genes.csv"), TextType="string");
assert(isequal(sort(heldCells), sort(savedCells.cell_line_id)), ...
    "留出细胞名单与阶段1不一致。");
assert(isequal(sort(heldGenes), sort(savedGenes.perturbation_gene)), ...
    "留出基因名单与阶段1不一致。");
end

function [X, featureNames] = buildDesign(geneBase, modules, moduleCount, useInteraction)
selectedModules = modules(:, 1:moduleCount);
X = [geneBase, selectedModules];
featureNames = ["gene_base", "module_" + compose("%02d", 1:moduleCount)];
if useInteraction
    X = [X, geneBase .* selectedModules];
    featureNames = [featureNames, ...
        "gene_base_x_module_" + compose("%02d", 1:moduleCount)];
end
featureNames = featureNames(:);
end

function isValidation = reproduceInnerSplit(labels, seed)
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
heldOldOldRows = oldOldCandidates(randperm(numel(oldOldCandidates), nHeldOldOldRows));
isValidation = isHeldCell | isHeldGene;
isValidation(heldOldOldRows) = true;
end

function [lambda, bestMse] = selectLambda(X, y, isTrain, isValidation, lambdas)
[XzTrain, center, scale] = standardizeByTraining(X(isTrain, :));
XzValidation = (X(isValidation, :) - center) ./ scale;
bestMse = inf;
lambda = lambdas(1);
for value = lambdas
    [beta, intercept] = fitRidge(XzTrain, y(isTrain), value);
    prediction = intercept + XzValidation * beta;
    mse = mean((prediction - y(isValidation)).^2);
    if mse < bestMse
        bestMse = mse;
        lambda = value;
    end
end
end

function [Xz, center, scale] = standardizeByTraining(X)
center = mean(X, 1);
scale = std(X, 0, 1);
scale(scale < 1e-12 | ~isfinite(scale)) = 1;
Xz = (X - center) ./ scale;
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
