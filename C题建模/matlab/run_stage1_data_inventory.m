function run_stage1_data_inventory(seed)
% 阶段1：读取 C 题数据并建立冷启动分组。

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

if ~isfolder(processedDir)
    mkdir(processedDir);
end
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

fprintf("Reading C-problem data...\n");
labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");
testPairs = readtable(fullfile(dataRoot, "submission", "sample_submission_gene.csv"), ...
    TextType="string", VariableNamingRule="preserve");
expression = readtable(fullfile(dataRoot, "features", "cell_expression_zscore.csv"), ...
    TextType="string", VariableNamingRule="preserve");
pathways = readtable(fullfile(dataRoot, "features", "cell_pathway_scores_zscore.csv"), ...
    TextType="string", VariableNamingRule="preserve");
cellMetadata = readtable(fullfile(dataRoot, "metadata", "cell_line_metadata.csv"), ...
    TextType="string", VariableNamingRule="preserve");
geneMetadata = readtable(fullfile(dataRoot, "metadata", "gene_metadata.csv"), ...
    TextType="string", VariableNamingRule="preserve");
pathwayMetadata = readtable(fullfile(dataRoot, "metadata", "pathway_metadata.csv"), ...
    TextType="string", VariableNamingRule="preserve");

trainCells = sort(unique(labels.cell_line_id));
trainGenes = sort(unique(labels.perturbation_gene));
testCells = sort(unique(testPairs.cell_line_id));
testGenes = sort(unique(testPairs.perturbation_gene));

isNewTestCell = ~ismember(testPairs.cell_line_id, trainCells);
isNewTestGene = ~ismember(testPairs.perturbation_gene, trainGenes);

groupNames = [
    "old_cell_old_gene"
    "old_cell_new_gene"
    "new_cell_old_gene"
    "new_cell_new_gene"
];
officialCounts = [
    sum(~isNewTestCell & ~isNewTestGene)
    sum(~isNewTestCell & isNewTestGene)
    sum(isNewTestCell & ~isNewTestGene)
    sum(isNewTestCell & isNewTestGene)
];
officialGroups = table(groupNames, officialCounts, ...
    'VariableNames', {'group', 'row_count'});
writetable(officialGroups, fullfile(resultsDir, "stage1_official_test_groups.csv"));

holdoutCellFraction = 0.11;
holdoutGeneFraction = 0.15;
oldOldRowFraction = 0.08;

nHeldCells = max(1, round(holdoutCellFraction * numel(trainCells)));
nHeldGenes = max(1, round(holdoutGeneFraction * numel(trainGenes)));
heldCells = trainCells(randperm(numel(trainCells), nHeldCells));
heldGenes = trainGenes(randperm(numel(trainGenes), nHeldGenes));

isHeldCellRow = ismember(labels.cell_line_id, heldCells);
isHeldGeneRow = ismember(labels.perturbation_gene, heldGenes);
oldOldCandidates = find(~isHeldCellRow & ~isHeldGeneRow);
nHeldOldOldRows = max(1, round(oldOldRowFraction * numel(oldOldCandidates)));
heldOldOldRows = oldOldCandidates(randperm(numel(oldOldCandidates), nHeldOldOldRows));

isOldOldValidation = false(height(labels), 1);
isOldOldValidation(heldOldOldRows) = true;
isValidation = isHeldCellRow | isHeldGeneRow | isOldOldValidation;

validationCounts = [
    sum(isValidation & ~isHeldCellRow & ~isHeldGeneRow)
    sum(isValidation & ~isHeldCellRow & isHeldGeneRow)
    sum(isValidation & isHeldCellRow & ~isHeldGeneRow)
    sum(isValidation & isHeldCellRow & isHeldGeneRow)
];
validationGroups = table(groupNames, validationCounts, ...
    'VariableNames', {'group', 'row_count'});
writetable(validationGroups, fullfile(resultsDir, "stage1_validation_groups.csv"));

writetable(table(sort(heldCells), 'VariableNames', {'cell_line_id'}), ...
    fullfile(processedDir, "stage1_heldout_cells.csv"));
writetable(table(sort(heldGenes), 'VariableNames', {'perturbation_gene'}), ...
    fullfile(processedDir, "stage1_heldout_genes.csv"));

featureCells = string(expression.cell_line_id);
pathwayCells = string(pathways.cell_line_id);
metadataCells = string(cellMetadata.cell_line_id);
metadataGenes = string(geneMetadata.gene_symbol);

allNeededCells = unique([trainCells; testCells]);
allNeededGenes = unique([trainGenes; testGenes]);

metric = [
    "train_rows"
    "test_rows"
    "train_unique_cells"
    "train_unique_genes"
    "test_unique_cells"
    "test_unique_genes"
    "new_test_cells"
    "new_test_genes"
    "expression_cells"
    "expression_gene_features"
    "pathway_cells"
    "pathway_features"
    "cell_metadata_rows"
    "gene_metadata_rows"
    "pathway_metadata_rows"
    "missing_needed_cells_in_expression"
    "missing_needed_cells_in_pathways"
    "missing_needed_cells_in_metadata"
    "missing_needed_genes_in_metadata"
    "label_missing"
    "label_min"
    "label_max"
    "label_mean"
    "label_std"
    "internal_training_rows"
    "internal_validation_rows"
    "heldout_cells"
    "heldout_genes"
];

value = [
    height(labels)
    height(testPairs)
    numel(trainCells)
    numel(trainGenes)
    numel(testCells)
    numel(testGenes)
    sum(~ismember(testCells, trainCells))
    sum(~ismember(testGenes, trainGenes))
    height(expression)
    width(expression) - 1
    height(pathways)
    width(pathways) - 1
    height(cellMetadata)
    height(geneMetadata)
    height(pathwayMetadata)
    sum(~ismember(allNeededCells, featureCells))
    sum(~ismember(allNeededCells, pathwayCells))
    sum(~ismember(allNeededCells, metadataCells))
    sum(~ismember(allNeededGenes, metadataGenes))
    sum(ismissing(labels.label))
    min(labels.label, [], "omitmissing")
    max(labels.label, [], "omitmissing")
    mean(labels.label, "omitmissing")
    std(labels.label, "omitmissing")
    sum(~isValidation)
    sum(isValidation)
    nHeldCells
    nHeldGenes
];

datasetSummary = table(metric, value);
writetable(datasetSummary, fullfile(resultsDir, "stage1_dataset_summary.csv"));

configName = ["seed"; "holdout_cell_fraction"; "holdout_gene_fraction"; "old_old_row_fraction"];
configValue = [seed; holdoutCellFraction; holdoutGeneFraction; oldOldRowFraction];
writetable(table(configName, configValue), fullfile(resultsDir, "stage1_split_config.csv"));

assert(sum(officialCounts) == height(testPairs), "Official test grouping lost rows.");
assert(sum(validationCounts) == sum(isValidation), "Validation grouping lost rows.");
assert(all(validationCounts > 0), "At least one validation group is empty.");

fprintf("Stage 1 complete. Train rows: %d; test rows: %d.\n", ...
    height(labels), height(testPairs));
fprintf("Official test groups [OO, ON, NO, NN]: %s\n", ...
    mat2str(officialCounts'));
fprintf("Internal validation groups [OO, ON, NO, NN]: %s\n", ...
    mat2str(validationCounts'));
fprintf("Results written to %s\n", resultsDir);
end
