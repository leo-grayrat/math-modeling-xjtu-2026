function run_stage2_baselines(seed)
% 阶段2：在固定冷启动划分上建立透明的基线模型。

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
scoringDir = fullfile(resultsDir, "stage2_scoring");

if ~isfolder(scoringDir)
    mkdir(scoringDir);
end

labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");

[isValidation, validationGroup, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed);
checkPersistedSplit(processedDir, heldCells, heldGenes);

training = labels(~isValidation, :);
validation = labels(isValidation, :);
validationGroup = validationGroup(isValidation);

globalMean = mean(training.label);

[cellGroup, knownCells] = findgroups(training.cell_line_id);
cellMean = splitapply(@mean, training.label, cellGroup);

[geneGroup, knownGenes] = findgroups(training.perturbation_gene);
geneMean = splitapply(@mean, training.label, geneGroup);

predictionGlobal = repmat(globalMean, height(validation), 1);
predictionCell = mapMeanOrFallback(validation.cell_line_id, ...
    knownCells, cellMean, globalMean);
predictionGene = mapMeanOrFallback(validation.perturbation_gene, ...
    knownGenes, geneMean, globalMean);

[cellEffect, geneEffect, additiveIterations, additiveDelta] = ...
    fitAdditiveEffects(training.label, cellGroup, geneGroup, ...
    numel(knownCells), numel(knownGenes), globalMean);
predictionAdditive = mapAdditive(validation, knownCells, cellEffect, ...
    knownGenes, geneEffect, globalMean);

predictions = {
    "global_mean", predictionGlobal
    "cell_mean", predictionCell
    "gene_mean", predictionGene
    "cell_gene_additive", predictionAdditive
};

answer = validation(:, ["cell_line_id", "perturbation_gene", "label"]);
writetable(answer, fullfile(scoringDir, "stage2_validation_answer.csv"));

groupTable = validation(:, ["cell_line_id", "perturbation_gene"]);
groupTable.group = validationGroup;
writetable(groupTable, fullfile(scoringDir, "stage2_validation_groups.csv"));

for i = 1:size(predictions, 1)
    modelName = predictions{i, 1};
    prediction = predictions{i, 2};
    assert(all(isfinite(prediction)), "Prediction contains non-finite values: %s", modelName);

    submission = validation(:, ["cell_line_id", "perturbation_gene"]);
    submission.label = prediction;
    writetable(submission, fullfile(scoringDir, "prediction_" + modelName + ".csv"));
end

summary = table( ...
    ["training_rows"; "validation_rows"; "global_mean"; ...
     "additive_iterations"; "additive_final_delta"], ...
    [height(training); height(validation); globalMean; ...
     additiveIterations; additiveDelta], ...
    'VariableNames', {'metric', 'value'});
writetable(summary, fullfile(resultsDir, "stage2_baseline_model_summary.csv"));

expectedCounts = readtable(fullfile(resultsDir, "stage1_validation_groups.csv"), ...
    TextType="string");
actualCounts = groupsummary(table(validationGroup, 'VariableNames', {'group'}), "group");
[matched, locations] = ismember(expectedCounts.group, actualCounts.group);
assert(all(matched), "Stage 2 is missing a validation group from Stage 1.");
assert(all(expectedCounts.row_count == actualCounts.GroupCount(locations)), ...
    "Stage 2 validation counts do not match Stage 1.");

fprintf("Stage 2 predictions ready. Training rows: %d; validation rows: %d.\n", ...
    height(training), height(validation));
fprintf("Additive model converged in %d iterations; final delta %.3g.\n", ...
    additiveIterations, additiveDelta);
fprintf("Scoring files written to %s\n", scoringDir);
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
    "Held-out cell list differs from Stage 1.");
assert(isequal(sort(heldGenes), sort(savedGenes.perturbation_gene)), ...
    "Held-out gene list differs from Stage 1.");
end

function prediction = mapMeanOrFallback(items, knownItems, knownMeans, fallback)
prediction = repmat(fallback, numel(items), 1);
[isKnown, location] = ismember(items, knownItems);
prediction(isKnown) = knownMeans(location(isKnown));
end

function [cellEffect, geneEffect, iteration, finalDelta] = fitAdditiveEffects( ...
    label, cellGroup, geneGroup, nCells, nGenes, globalMean)
cellEffect = zeros(nCells, 1);
geneEffect = zeros(nGenes, 1);
maxIterations = 100;
tolerance = 1e-10;
finalDelta = inf;

for iteration = 1:maxIterations
    previousCell = cellEffect;
    previousGene = geneEffect;

    cellEffect = splitapply(@mean, label - globalMean - geneEffect(geneGroup), cellGroup);
    geneEffect = splitapply(@mean, label - globalMean - cellEffect(cellGroup), geneGroup);

    finalDelta = max([
        max(abs(cellEffect - previousCell))
        max(abs(geneEffect - previousGene))
    ]);
    if finalDelta < tolerance
        break
    end
end
end

function prediction = mapAdditive(validation, knownCells, cellEffect, ...
    knownGenes, geneEffect, globalMean)
prediction = repmat(globalMean, height(validation), 1);

[knownCell, cellLocation] = ismember(validation.cell_line_id, knownCells);
prediction(knownCell) = prediction(knownCell) + cellEffect(cellLocation(knownCell));

[knownGene, geneLocation] = ismember(validation.perturbation_gene, knownGenes);
prediction(knownGene) = prediction(knownGene) + geneEffect(geneLocation(knownGene));
end
