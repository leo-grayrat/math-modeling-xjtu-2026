function data = stage10_prepare_full_data(repoRoot, projectDir)
%STAGE10_PREPARE_FULL_DATA 为训练标签和正式提交对象建立统一特征。

arguments
    repoRoot (1, 1) string
    projectDir (1, 1) string
end

dataRoot = fullfile(repoRoot, "2026年校赛题目", "C", "数据文件");
resultsDir = fullfile(projectDir, "results");
labels = readtable(fullfile(dataRoot, "labels", "gene_dependency.csv"), ...
    TextType="string", VariableNamingRule="preserve");
submission = readtable(fullfile(dataRoot, "submission", ...
    "sample_submission_gene.csv"), TextType="string", ...
    VariableNamingRule="preserve");
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

cellNames = sort(unique([labels.cell_line_id; submission.cell_line_id]));
geneNames = sort(unique([labels.perturbation_gene; ...
    submission.perturbation_gene]));
[labelCellKnown, rowCellIndex] = ismember(labels.cell_line_id, cellNames);
[labelGeneKnown, rowGeneIndex] = ismember(labels.perturbation_gene, geneNames);
[submissionCellKnown, submissionCellIndex] = ismember( ...
    submission.cell_line_id, cellNames);
[submissionGeneKnown, submissionGeneIndex] = ismember( ...
    submission.perturbation_gene, geneNames);
assert(all(labelCellKnown & labelGeneKnown) && ...
    all(submissionCellKnown & submissionGeneKnown), ...
    "stage10:ObjectMapping", "训练或提交对象映射失败。");

[expressionKnown, expressionIndex] = ismember(cellNames, ...
    string(expression.cell_line_id));
[pathwayKnown, pathwayIndex] = ismember(cellNames, ...
    string(pathways.cell_line_id));
[cellMetadataKnown, cellMetadataIndex] = ismember(cellNames, ...
    string(cellMetadata.cell_line_id));
[moduleKnown, moduleIndex] = ismember(cellNames, ...
    string(moduleScores.cell_line_id));
assert(all(expressionKnown & pathwayKnown & cellMetadataKnown & moduleKnown), ...
    "stage10:CellCoverage", "正式预测细胞缺少完整特征。");

expressionGeneNames = string(expression.Properties.VariableNames(2:end));
[expressionGeneKnown, expressionGeneIndex] = ismember( ...
    geneNames, expressionGeneNames);
[geneMetadataKnown, geneMetadataIndex] = ismember( ...
    geneNames, string(geneMetadata.gene_symbol));
assert(all(expressionGeneKnown & geneMetadataKnown), ...
    "stage10:GeneCoverage", "正式预测基因缺少表达列或元数据。");

cellTrainingMask = ismember(cellNames, unique(labels.cell_line_id));
geneTrainingMask = ismember(geneNames, unique(labels.perturbation_gene));
cellExpression = expression{expressionIndex, 2:end};
cellPathway = pathways{pathwayIndex, 2:end};
cellModules = moduleScores{moduleIndex, 2:end};
cellRows = cellMetadata(cellMetadataIndex, :);
geneRows = geneMetadata(geneMetadataIndex, :);
geneNumeric = buildGeneNumericBlock(geneRows);
cellCategory = encodeCategories(cellRows, ...
    ["OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]);
geneCategory = encodeCategories(geneRows, ...
    ["curated_gene_list", "sub_mito_location", "mito_domain_score", ...
    "yeast_mito_homolog_score", "rickettsia_homolog_score", ...
    "msms_score"]);
pathwayNames = string(pathways.Properties.VariableNames(2:end));
genePathway = build_gene_pathway_membership( ...
    string(geneRows.pathways), pathwayNames);
geneEvidence = encodeDelimitedTokens(geneRows.curation_evidence, ",");
ownExpression = expression{expressionIndex, expressionGeneIndex + 1};
geneExpressionProfile = ownExpression' / sqrt(size(ownExpression, 1));
baseGeneFeatures = buildBaseGeneFeatures( ...
    geneRows, geneNumeric, geneTrainingMask);

cellBlocks = {cellExpression, cellPathway, cellCategory, cellModules};
geneBlocks = {geneNumeric, geneCategory, genePathway, ...
    geneEvidence, geneExpressionProfile};
cellKernelFeatures = processBlocks(cellBlocks, cellTrainingMask);
geneKernelFeatures = processBlocks(geneBlocks, geneTrainingMask);
cellKernel = buildRbfKernel(cellKernelFeatures);
geneKernel = buildRbfKernel(geneKernelFeatures);

labelPairIndex = sub2ind([numel(cellNames), numel(geneNames)], ...
    rowCellIndex, rowGeneIndex);
submissionPairIndex = sub2ind([numel(cellNames), numel(geneNames)], ...
    submissionCellIndex, submissionGeneIndex);
assert(numel(unique(labelPairIndex)) == numel(labelPairIndex), ...
    "stage10:DuplicateLabels", "训练标签存在重复细胞—基因组合。");
assert(numel(unique(submissionPairIndex)) == numel(submissionPairIndex), ...
    "stage10:DuplicateSubmission", "提交模板存在重复细胞—基因组合。");

data.labels = labels;
data.submission = submission;
data.cellNames = cellNames;
data.geneNames = geneNames;
data.rowCellIndex = rowCellIndex;
data.rowGeneIndex = rowGeneIndex;
data.submissionCellIndex = submissionCellIndex;
data.submissionGeneIndex = submissionGeneIndex;
data.cellTrainingMask = cellTrainingMask;
data.geneTrainingMask = geneTrainingMask;
data.baseGeneFeatures = baseGeneFeatures;
data.cellKernelFeatures = cellKernelFeatures;
data.geneKernelFeatures = geneKernelFeatures;
data.cellKernel = cellKernel;
data.geneKernel = geneKernel;
end

function processed = processBlocks(blocks, trainingMask)
processedBlocks = cell(size(blocks));
for blockNumber = 1:numel(blocks)
    block = blocks{blockNumber};
    for columnNumber = 1:size(block, 2)
        observed = block(trainingMask, columnNumber);
        observed = observed(isfinite(observed));
        fillValue = 0;
        if ~isempty(observed)
            fillValue = median(observed);
        end
        block(~isfinite(block(:, columnNumber)), columnNumber) = fillValue;
    end
    center = mean(block(trainingMask, :), 1);
    scale = std(block(trainingMask, :), 0, 1);
    scale(scale < 1e-12 | ~isfinite(scale)) = 1;
    block = (block - center) ./ scale;
    block = block / sqrt(size(block, 2));
    processedBlocks{blockNumber} = block;
end
processed = cat(2, processedBlocks{:});
end

function kernel = buildRbfKernel(features)
squaredDistance = pdist2(features, features, "squaredeuclidean");
upperValues = squaredDistance(triu(true(size(squaredDistance)), 1));
positiveValues = upperValues(upperValues > 1e-12 & isfinite(upperValues));
bandwidth = 1;
if ~isempty(positiveValues)
    bandwidth = median(positiveValues);
end
kernel = exp(-squaredDistance / (2 * bandwidth));
kernel = (kernel + kernel') / 2;
kernel(1:size(kernel, 1) + 1:end) = 1;
end

function features = buildBaseGeneFeatures(geneRows, numericBlock, trainingRows)
for columnNumber = 1:size(numericBlock, 2)
    observed = numericBlock(trainingRows, columnNumber);
    observed = observed(isfinite(observed));
    fillValue = 0;
    if ~isempty(observed)
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
for fieldNumber = 1:numel(categoryFields)
    values = string(geneRows.(categoryFields(fieldNumber)));
    values(ismissing(values) | strlength(strtrim(values)) == 0) = "<missing>";
    levels = unique(values(trainingRows), "stable");
    for levelNumber = 2:numel(levels)
        features(:, end + 1) = double(values == levels(levelNumber)); %#ok<AGROW>
    end
end
features(:, end + 1) = 0;
end

function block = buildGeneNumericBlock(geneRows)
block = [geneRows.targetp_score, geneRows.coexpression_gnf_n50_score, ...
    geneRows.pgc_induction_score, countTokens(geneRows.curation_evidence, ","), ...
    countTokens(geneRows.pathways, "|"), countTokens(geneRows.synonyms, "|"), ...
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
        tokens = unique(tokens(strlength(tokens) > 0));
        tokenRows{rowNumber} = tokens;
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
