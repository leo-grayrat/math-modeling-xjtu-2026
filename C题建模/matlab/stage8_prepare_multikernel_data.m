function data = stage8_prepare_multikernel_data(repoRoot, projectDir, seed)
%STAGE8_PREPARE_MULTIKERNEL_DATA 读取原始资料并保留九个未压缩特征块。

arguments
    repoRoot (1, 1) string
    projectDir (1, 1) string
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
end

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
    "stage8:LabelMapping", "标签行无法映射到细胞或基因名单。");

[isOuterValidation, outerGroup, heldCells, heldGenes] = ...
    reproduceStage1Split(labels, seed);
checkPersistedSplit(processedDir, heldCells, heldGenes);

[knownExpressionCells, expressionCellIndex] = ...
    ismember(cellNames, expression.cell_line_id);
[knownPathwayCells, pathwayCellIndex] = ...
    ismember(cellNames, pathways.cell_line_id);
[knownCellMetadata, cellMetadataIndex] = ...
    ismember(cellNames, cellMetadata.cell_line_id);
[knownModules, moduleIndex] = ismember(cellNames, moduleScores.cell_line_id);
assert(all(knownExpressionCells & knownPathwayCells & ...
    knownCellMetadata & knownModules), ...
    "stage8:CellCoverage", "部分标签细胞缺少完整特征。");

expressionGeneNames = string(expression.Properties.VariableNames(2:end));
[knownExpressionGenes, expressionGeneIndex] = ...
    ismember(geneNames, expressionGeneNames);
assert(all(knownExpressionGenes), "stage8:ExpressionGeneCoverage", ...
    "部分待敲基因没有对应的细胞表达列。");

[knownGeneMetadata, geneMetadataIndex] = ...
    ismember(geneNames, geneMetadata.gene_symbol);
assert(all(knownGeneMetadata), "stage8:GeneMetadataCoverage", ...
    "部分标签基因缺少元数据。");

cellExpression = expression{expressionCellIndex, 2:end};
cellPathway = pathways{pathwayCellIndex, 2:end};
cellModules = moduleScores{moduleIndex, 2:end};
cellRows = cellMetadata(cellMetadataIndex, :);
geneRows = geneMetadata(geneMetadataIndex, :);
geneNumeric = buildGeneNumericBlock(geneRows);
pathwayNames = string(pathways.Properties.VariableNames(2:end));
genePathway = build_gene_pathway_membership( ...
    string(geneRows.pathways), pathwayNames);
ownExpression = expression{expressionCellIndex, expressionGeneIndex + 1};

pairIndex = sub2ind([numel(cellNames), numel(geneNames)], ...
    rowCellIndex, rowGeneIndex);
assert(numel(unique(pairIndex)) == numel(pairIndex), ...
    "stage8:DuplicateLabels", "标签表中存在重复细胞—基因组合。");

data.labels = labels;
data.cellNames = cellNames;
data.geneNames = geneNames;
data.rowCellIndex = rowCellIndex;
data.rowGeneIndex = rowGeneIndex;
data.isOuterValidation = isOuterValidation;
data.outerGroup = outerGroup;
data.cellExpression = cellExpression;
data.cellPathway = cellPathway;
data.cellMetadata = cellRows;
data.cellModules = cellModules;
data.geneNumeric = geneNumeric;
data.geneMetadata = geneRows;
data.genePathway = genePathway;
data.geneExpressionProfile = ownExpression';
data.cellBlockNames = ["cell_expression"; "cell_pathway"; ...
    "cell_metadata"; "cell_modules"];
data.geneBlockNames = ["gene_numeric"; "gene_categories"; ...
    "gene_pathways"; "gene_evidence"; "gene_expression_profile"];
end

function block = buildGeneNumericBlock(geneRows)
block = [geneRows.targetp_score, geneRows.coexpression_gnf_n50_score, ...
    geneRows.pgc_induction_score, ...
    countTokens(geneRows.curation_evidence, ","), ...
    countTokens(geneRows.pathways, "|"), ...
    countTokens(geneRows.synonyms, "|"), ...
    double(strlength(geneRows.description))];
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
    "stage8:HeldCells", "留出细胞名单与阶段1不一致。");
assert(isequal(sort(heldGenes), sort(savedGenes.perturbation_gene)), ...
    "stage8:HeldGenes", "留出基因名单与阶段1不一致。");
end
