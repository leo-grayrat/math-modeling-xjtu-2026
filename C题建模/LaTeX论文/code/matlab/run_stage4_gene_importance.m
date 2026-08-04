function run_stage4_gene_importance(seed)
% 阶段4：根据基因元数据估计基因的普遍重要性。
% 外层划分沿用阶段1，内层再按基因留出数据选择惩罚参数，避免提前使用外层新基因标签。

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
scoringDir = fullfile(resultsDir, "stage4_scoring");
if ~isfolder(scoringDir)
    mkdir(scoringDir);
end

labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");
geneMetadata = readtable(fullfile(dataRoot, "metadata", "gene_metadata.csv"), ...
    TextType="string", VariableNamingRule="preserve");

[isValidation, validationGroup, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed);
checkPersistedSplit(processedDir, heldCells, heldGenes);

training = labels(~isValidation, :);
validation = labels(isValidation, :);
validationGroup = validationGroup(isValidation);
globalMean = mean(training.label);

[geneGroup, trainGenes] = findgroups(training.perturbation_gene);
trainGeneMean = splitapply(@mean, training.label, geneGroup);
trainGeneCount = splitapply(@numel, training.label, geneGroup);

allValidationGenes = unique(validation.perturbation_gene, "stable");
allGenes = unique([trainGenes; allValidationGenes], "stable");
[rawFeatures, metadataKnown, featureDiagnostics] = ...
    extractGeneFeatures(geneMetadata, allGenes);
trainFeatureRows = ismember(allGenes, trainGenes);
[X, featureNames, featureKinds, preprocessing] = ...
    fitAndEncodeFeatures(rawFeatures, trainFeatureRows, metadataKnown);

[~, trainLocations] = ismember(trainGenes, allGenes);
trainFeature = X(trainLocations, :);

% 内层验证按基因划分，不把同一基因的不同细胞行混到训练和验证两边。
rng(seed + 1000, "twister");
nInner = max(1, round(0.20 * numel(trainGenes)));
innerValidation = false(numel(trainGenes), 1);
innerValidation(randperm(numel(trainGenes), nInner)) = true;
innerTraining = ~innerValidation;

ridgeLambdas = logspace(-3, 3, 25);
[ridgeLambda, ridgeInnerMse] = selectRidgeLambda( ...
    trainFeature, trainGeneMean, innerTraining, innerValidation, ridgeLambdas);
[ridgeBeta, ridgeIntercept] = fitRidge(trainFeature, trainGeneMean, ridgeLambda);
ridgeGenePrediction = ridgeIntercept + X * ridgeBeta;

centerX = trainFeature(innerTraining, :) - ...
    mean(trainFeature(innerTraining, :), 1);
centerY = trainGeneMean(innerTraining) - mean(trainGeneMean(innerTraining));
lambdaMax = max(abs(centerX' * centerY)) / sum(innerTraining);
if ~isfinite(lambdaMax) || lambdaMax <= 0
    lambdaMax = 1;
end
lassoLambdas = logspace(log10(lambdaMax), log10(lambdaMax * 1e-3), 25);
[lassoLambda, lassoInnerMse] = selectLassoLambda( ...
    trainFeature, trainGeneMean, innerTraining, innerValidation, lassoLambdas);
[lassoBeta, lassoFitInfo] = lasso(trainFeature, trainGeneMean, ...
    "Lambda", lassoLambda, "Standardize", false, "Intercept", true, ...
    "MaxIter", 5000);
lassoGenePrediction = lassoFitInfo.Intercept + X * lassoBeta;

% 已见基因继续使用直接估计的基因均值，只有新基因使用元数据回归，单独观察冷启动贡献。
[knownTrainGene, trainGeneLocation] = ismember(allGenes, trainGenes);
ridgeHybrid = ridgeGenePrediction;
lassoHybrid = lassoGenePrediction;
ridgeHybrid(knownTrainGene) = trainGeneMean(trainGeneLocation(knownTrainGene));
lassoHybrid(knownTrainGene) = trainGeneMean(trainGeneLocation(knownTrainGene));

modelNames = ["gene_mean"; "metadata_ridge"; "metadata_lasso"; ...
    "hybrid_ridge"; "hybrid_lasso"];
genePredictions = [
    mapKnownGeneMean(validation.perturbation_gene, trainGenes, trainGeneMean, globalMean), ...
    mapGenePrediction(validation.perturbation_gene, allGenes, ridgeGenePrediction, globalMean), ...
    mapGenePrediction(validation.perturbation_gene, allGenes, lassoGenePrediction, globalMean), ...
    mapGenePrediction(validation.perturbation_gene, allGenes, ridgeHybrid, globalMean), ...
    mapGenePrediction(validation.perturbation_gene, allGenes, lassoHybrid, globalMean)
    ];

for modelIndex = 1:numel(modelNames)
    submission = validation(:, ["cell_line_id", "perturbation_gene"]);
    submission.label = genePredictions(:, modelIndex);
    writetable(submission, fullfile(scoringDir, ...
        "prediction_" + modelNames(modelIndex) + ".csv"));
end

modelSummary = buildModelSummary(modelNames, genePredictions, validation, ...
    validationGroup, ridgeLambda, lassoLambda, ridgeInnerMse, ...
    lassoInnerMse, metadataKnown, trainGenes, allGenes);
writetable(modelSummary, fullfile(resultsDir, "stage4_gene_model_summary.csv"));

geneBaseScores = table(allGenes, ridgeHybrid, lassoHybrid, metadataKnown, ...
    'VariableNames', {'perturbation_gene', 'hybrid_ridge_base', ...
    'hybrid_lasso_base', 'metadata_known'});
writetable(geneBaseScores, fullfile(resultsDir, ...
    "stage4_gene_base_scores.csv"));

featureSummary = table(featureNames(:), featureKinds(:), ...
    'VariableNames', {'feature', 'kind'});
writetable(featureSummary, fullfile(resultsDir, "stage4_gene_feature_summary.csv"));

coverageSummary = table( ...
    ["metadata_rows"; "metadata_known_all_genes"; "metadata_known_train_genes"; ...
     "metadata_known_validation_genes"; "train_genes"; "validation_genes"], ...
    [height(geneMetadata); sum(metadataKnown); sum(metadataKnown(trainFeatureRows)); ...
     sum(metadataKnown(ismember(allGenes, allValidationGenes))); numel(trainGenes); ...
     numel(allValidationGenes)], ...
    'VariableNames', {'metric', 'value'});
writetable([coverageSummary; featureDiagnostics], ...
    fullfile(resultsDir, "stage4_gene_metadata_summary.csv"));

fprintf("Stage 4 complete. Training genes: %d; validation genes: %d.\n", ...
    numel(trainGenes), numel(allValidationGenes));
fprintf("Ridge lambda %.5g; LASSO lambda %.5g.\n", ridgeLambda, lassoLambda);
fprintf("Results written to %s\n", resultsDir);
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
    "Held-out cells differ from Stage 1.");
assert(isequal(sort(heldGenes), sort(savedGenes.perturbation_gene)), ...
    "Held-out genes differ from Stage 1.");
end

function [raw, metadataKnown, diagnostics] = extractGeneFeatures(metadata, geneNames)
[metadataKnown, location] = ismember(geneNames, metadata.gene_symbol);
knownLocations = location(metadataKnown);
assert(numel(unique(knownLocations)) == numel(knownLocations), ...
    "Gene metadata contains duplicate symbols.");

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
for j = 1:numel(categoryFields)
    values = strings(numel(geneNames), 1);
    if any(metadataKnown)
        values(metadataKnown) = metadata.(categoryFields(j))(knownLocations);
    end
    values(ismissing(values) | strlength(strtrim(values)) == 0) = "<missing>";
    categories(:, j) = values;
end

raw.numeric = numeric;
raw.categories = categories;
raw.numericNames = ["targetp_score", "coexpression_gnf_n50_score", ...
    "pgc_induction_score", "evidence_count", "pathway_count", ...
    "synonym_count", "description_length"];
raw.categoryNames = categoryFields;

msmsMissing = ismissing(metadata.msms_score) | ...
    strlength(strtrim(metadata.msms_score)) == 0;
knownMsmsMissing = false(numel(geneNames), 1);
knownMsmsMissing(metadataKnown) = msmsMissing(knownLocations);
diagnostics = table( ...
    ["missing_targetp"; "missing_coexpression"; "missing_pgc_induction"; ...
     "missing_msms"; "missing_metadata"], ...
    [sum(isnan(numeric(:, 1))); sum(isnan(numeric(:, 2))); ...
     sum(isnan(numeric(:, 3))); sum(knownMsmsMissing); sum(~metadataKnown)], ...
    'VariableNames', {'metric', 'value'});
end

function counts = tokenCount(values, delimiter)
counts = zeros(numel(values), 1);
for i = 1:numel(values)
    value = values(i);
    if ismissing(value) || strlength(strtrim(value)) == 0 || value == "-"
        counts(i) = 0;
    else
        counts(i) = numel(strsplit(value, delimiter));
    end
end
end

function [X, featureNames, featureKinds, state] = ...
    fitAndEncodeFeatures(raw, trainRows, metadataKnown)
Xnumeric = raw.numeric;
medians = zeros(1, size(Xnumeric, 2));
for j = 1:size(Xnumeric, 2)
    observed = Xnumeric(trainRows, j);
    medians(j) = median(observed(~isnan(observed)));
    if isempty(medians(j)) || ~isfinite(medians(j))
        medians(j) = 0;
    end
    missing = isnan(Xnumeric(:, j));
    Xnumeric(missing, j) = medians(j);
end
means = mean(Xnumeric(trainRows, :), 1);
scales = std(Xnumeric(trainRows, :), 0, 1);
scales(scales < 1e-12 | ~isfinite(scales)) = 1;
Xnumeric = (Xnumeric - means) ./ scales;

featureNames = raw.numericNames;
featureKinds = repmat("numeric", 1, numel(raw.numericNames));
X = Xnumeric;
for j = 1:size(raw.categories, 2)
    values = raw.categories(:, j);
    levels = unique(values(trainRows), "stable");
    if isempty(levels)
        levels = "<missing>";
    end
        % 去掉第一个类别，把它的平均作用交给截距表示。
    for k = 2:numel(levels)
        X(:, end + 1) = double(values == levels(k));
        featureNames(end + 1) = raw.categoryNames(j) + "=" + levels(k);
        featureKinds(end + 1) = "category";
    end
end
X(:, end + 1) = double(~metadataKnown);
featureNames(end + 1) = "metadata_missing";
featureKinds(end + 1) = "missing_indicator";
state.medians = medians;
state.means = means;
state.scales = scales;
end

function [lambda, bestMse] = selectRidgeLambda(X, y, isTrain, isValidation, lambdas)
bestMse = inf;
lambda = lambdas(1);
for value = lambdas
    [beta, intercept] = fitRidge(X(isTrain, :), y(isTrain), value);
    prediction = intercept + X(isValidation, :) * beta;
    mse = mean((prediction - y(isValidation)).^2);
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

function [lambda, bestMse] = selectLassoLambda(X, y, isTrain, isValidation, lambdas)
bestMse = inf;
lambda = lambdas(1);
for value = lambdas
    [beta, fitInfo] = lasso(X(isTrain, :), y(isTrain), ...
        "Lambda", value, "Standardize", false, "Intercept", true, ...
        "MaxIter", 5000);
    prediction = fitInfo.Intercept + X(isValidation, :) * beta;
    mse = mean((prediction - y(isValidation)).^2);
    if mse < bestMse
        bestMse = mse;
        lambda = value;
    end
end
end

function prediction = mapKnownGeneMean(geneNames, knownGenes, knownMeans, fallback)
prediction = repmat(fallback, numel(geneNames), 1);
[known, location] = ismember(geneNames, knownGenes);
prediction(known) = knownMeans(location(known));
end

function prediction = mapGenePrediction(geneNames, allGenes, geneValues, fallback)
prediction = repmat(fallback, numel(geneNames), 1);
[known, location] = ismember(geneNames, allGenes);
prediction(known) = geneValues(location(known));
end

function summary = buildModelSummary(modelNames, predictions, validation, groups, ...
    ridgeLambda, lassoLambda, ridgeInnerMse, lassoInnerMse, metadataKnown, ...
    trainGenes, allGenes)
rows = strings(0, 1);
values = zeros(0, 6);
for i = 1:numel(modelNames)
    groupNames = ["all"; "old_cell_old_gene"; "old_cell_new_gene"; ...
        "new_cell_old_gene"; "new_cell_new_gene"];
    for groupName = groupNames'
        if groupName == "all"
            mask = true(height(validation), 1);
        else
            mask = groups == groupName;
        end
        truth = validation.label(mask);
        prediction = predictions(mask, i);
        if isempty(truth)
            continue
        end
        rho = corr(truth, prediction, "Type", "Spearman", "Rows", "complete");
        rows(end + 1) = modelNames(i) + "|" + groupName;
        values(end + 1, :) = [i, double(groupName == "all"), numel(truth), ...
            sqrt(mean((prediction - truth).^2)), mean(abs(prediction - truth)), rho];
    end
end
summary = array2table(values, 'VariableNames', ...
    {'model_index', 'is_overall', 'rows', 'rmse', 'mae', 'spearman'});
summary.model = rows(:);
summary = movevars(summary, "model", "Before", 1);
summary.ridge_lambda = repmat(ridgeLambda, height(summary), 1);
summary.lasso_lambda = repmat(lassoLambda, height(summary), 1);
summary.ridge_inner_mse = repmat(ridgeInnerMse, height(summary), 1);
summary.lasso_inner_mse = repmat(lassoInnerMse, height(summary), 1);
summary.train_genes = repmat(numel(trainGenes), height(summary), 1);
summary.metadata_known_all_genes = repmat(sum(metadataKnown), height(summary), 1);
summary.all_genes = repmat(numel(allGenes), height(summary), 1);
end
