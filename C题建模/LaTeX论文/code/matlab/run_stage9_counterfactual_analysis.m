function run_stage9_counterfactual_analysis(seed, runMode)
% 阶段9：用冻结阶段7模型完成虚拟敲除输入敏感性检查。

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
    runMode (1, 1) string {mustBeMember(runMode, ...
        ["smoke", "full"])} = "full"
end

rng(seed, "twister");
matlabDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(matlabDir);
repoRoot = fileparts(projectDir);
dataRoot = fullfile(repoRoot, "2026年校赛题目", "C", "数据文件");
resultsDir = fullfile(projectDir, "results");

bundle = run_stage7_full_information_regression(seed, "stage9_bundle");
data = bundle.data;
rawExpression = readtable(fullfile(dataRoot, "features", ...
    "cell_expression_raw.csv"), TextType="string", ...
    VariableNamingRule="preserve");
zExpression = readtable(fullfile(dataRoot, "features", ...
    "cell_expression_zscore.csv"), TextType="string", ...
    VariableNamingRule="preserve");
zPathways = readtable(fullfile(dataRoot, "features", ...
    "cell_pathway_scores_zscore.csv"), TextType="string", ...
    VariableNamingRule="preserve");
geneMetadata = readtable(fullfile(dataRoot, "metadata", ...
    "gene_metadata.csv"), TextType="string", ...
    VariableNamingRule="preserve");
moduleScores = readtable(fullfile(resultsDir, ...
    "stage3_cell_module_scores.csv"), TextType="string", ...
    VariableNamingRule="preserve");
pathwayStructure = readtable(fullfile(resultsDir, ...
    "stage3_pathway_structure.csv"), TextType="string", ...
    VariableNamingRule="preserve");

cellNames = data.cellNames;
geneNames = data.geneNames;
expressionGeneNames = string(zExpression.Properties.VariableNames(2:end))';
pathwayNames = string(zPathways.Properties.VariableNames(2:end))';
[expressionCellKnown, expressionCellIndex] = ismember( ...
    cellNames, string(zExpression.cell_line_id));
[pathwayCellKnown, pathwayCellIndex] = ismember( ...
    cellNames, string(zPathways.cell_line_id));
[moduleCellKnown, moduleCellIndex] = ismember( ...
    cellNames, string(moduleScores.cell_line_id));
assert(all(expressionCellKnown & pathwayCellKnown & moduleCellKnown), ...
    "stage9:CellFeatureCoverage", "阶段9细胞特征覆盖不完整。");

expressionZ = zExpression{expressionCellIndex, 2:end};
pathwayZ = zPathways{pathwayCellIndex, 2:end};
moduleValue = moduleScores{moduleCellIndex, 2:end};
[knownGenes, expressionGeneIndex] = ismember(geneNames, expressionGeneNames);
assert(all(knownGenes), "stage9:GeneExpressionCoverage", ...
    "阶段9目标基因缺少表达列。");

[structureKnown, structureLocation] = ismember( ...
    pathwayNames, pathwayStructure.pathway_name);
assert(all(structureKnown), "stage9:PathwayStructure", ...
    "阶段9通路结构与输入列不一致。");
structure = pathwayStructure(structureLocation, :);
leafMask = logical(structure.is_leaf);
originalPathwayOrder = zPathways{:, 2:end};
encoder = rebuildModuleEncoder(originalPathwayOrder(:, leafMask), seed);
encoder.leafColumns = find(leafMask);
reconstructedModules = stage9_apply_bartlett_scores( ...
    pathwayZ(:, leafMask), encoder);
assert(max(abs(reconstructedModules(:) - moduleValue(:))) < 1e-8, ...
    "stage9:ModuleReconstruction", "重建模块与阶段3保存结果不一致。");

[membership, pathwayGeneCount] = stage9_build_pathway_membership( ...
    expressionGeneNames, geneMetadata, pathwayNames);
geneMembership = stage9_align_gene_membership( ...
    membership, expressionGeneIndex);
trainingCellMask = false(numel(cellNames), 1);
trainingCellMask(unique(data.rowCellIndex(bundle.outerTraining))) = true;
blockWidths = data.featureCount(1:4);
expressionTransform = fitBlockTransform(expressionZ, trainingCellMask);
pathwayTransform = fitBlockTransform(pathwayZ, trainingCellMask);
moduleTransform = fitBlockTransform(moduleValue, trainingCellMask);
expressionBlock = data.cellKernelFeatures(:, 1:blockWidths(1));
pathwayBlock = data.cellKernelFeatures(:, ...
    blockWidths(1) + (1:blockWidths(2)));
moduleStart = sum(blockWidths(1:3)) + 1;
moduleBlock = data.cellKernelFeatures(:, ...
    moduleStart:sum(blockWidths(1:4)));
expectedExpressionBlock = applyBlockTransform( ...
    expressionZ, expressionTransform);
expectedPathwayBlock = applyBlockTransform(pathwayZ, pathwayTransform);
expectedModuleBlock = applyBlockTransform(moduleValue, moduleTransform);
assert(max(abs(expressionBlock(:) - expectedExpressionBlock(:))) < 1e-8, ...
    "stage9:ExpressionTransform", "表达块变换与阶段7不一致。");
assert(max(abs(pathwayBlock(:) - expectedPathwayBlock(:))) < 1e-8, ...
    "stage9:PathwayTransform", "通路块变换与阶段7不一致。");
assert(max(abs(moduleBlock(:) - expectedModuleBlock(:))) < 1e-8, ...
    "stage9:ModuleTransform", "模块块变换与阶段7不一致。");
cellBandwidth = rbfBandwidth(data.cellKernelFeatures);

validationRows = find(data.isOuterValidation);
geneHasPathway = sum(geneMembership, 2) > 0;
linearIndex = sub2ind(size(bundle.model.predictionMatrix), ...
    data.rowCellIndex, data.rowGeneIndex);
predictedByRow = bundle.baseByRow + bundle.model.predictionMatrix(linearIndex);
candidateRows = validationRows(geneHasPathway(data.rowGeneIndex(validationRows)));
[~, order] = sort(predictedByRow(candidateRows), "descend");
candidateRows = candidateRows(order);
caseCount = 3;
intensities = [0.5, 0.75, 1.0];
filePrefix = "stage9";
if runMode == "smoke"
    caseCount = 1;
    intensities = 0.5;
    filePrefix = "stage9_smoke";
end
selectedRows = selectDistinctCases(candidateRows, data.rowCellIndex, ...
    data.rowGeneIndex, caseCount);
assert(numel(selectedRows) == caseCount, "stage9:CaseSelection", ...
    "阶段9未能选出足够数量的不同案例。");
caseRows = table;
predictionRows = table;
pathwayRows = table;
moduleRows = table;
for caseNumber = 1:numel(selectedRows)
    rowNumber = selectedRows(caseNumber);
    cellIndex = data.rowCellIndex(rowNumber);
    geneIndex = data.rowGeneIndex(rowNumber);
    [~, rawGeneIndex] = ismember(geneNames(geneIndex), expressionGeneNames);
    rawColumnAll = rawExpression{:, rawGeneIndex + 1};
    rawMean = mean(rawColumnAll, "omitnan");
    rawScale = std(rawColumnAll, 0, "omitnan");
    knockoutZ = (0 - rawMean) / (rawScale + 1e-8);
    originalPrediction = predictedByRow(rowNumber);
    zeroState = stage9_build_counterfactual_state( ...
            expressionZ(cellIndex, :), pathwayZ(cellIndex, :), ...
            expressionGeneIndex(geneIndex), knockoutZ, ...
            geneMembership(geneIndex, :), pathwayGeneCount, 0, encoder);
    zeroFeatures = data.cellKernelFeatures(cellIndex, :);
    zeroFeatures(1:blockWidths(1)) = applyBlockTransform( ...
        zeroState.expression, expressionTransform);
    zeroFeatures(blockWidths(1) + (1:blockWidths(2))) = ...
        applyBlockTransform(zeroState.pathway, pathwayTransform);
    zeroFeatures(moduleStart:sum(blockWidths(1:4))) = ...
        applyBlockTransform(zeroState.modules, moduleTransform);
    zeroKernelRow = rbfKernelRow(zeroFeatures, ...
        data.cellKernelFeatures, cellBandwidth);
    assert(max(abs(zeroKernelRow - bundle.cellKernel(cellIndex, :))) < 1e-10, ...
        "stage9:ZeroKernel", "零扰动没有还原阶段7细胞核行。");
    zeroResidual = stage9_predict_counterfactual_cell( ...
        bundle.model, zeroKernelRow);
    zeroPrediction = bundle.baseByGene(geneIndex) + zeroResidual(geneIndex);
    assert(abs(zeroPrediction - originalPrediction) < 1e-10, ...
        "stage9:ZeroPrediction", "零扰动没有还原阶段7预测。");
    for intensity = intensities
        state = stage9_build_counterfactual_state( ...
            expressionZ(cellIndex, :), pathwayZ(cellIndex, :), ...
            expressionGeneIndex(geneIndex), knockoutZ, ...
            geneMembership(geneIndex, :), pathwayGeneCount, intensity, encoder);
        counterCellFeatures = data.cellKernelFeatures(cellIndex, :);
        counterCellFeatures(1:blockWidths(1)) = applyBlockTransform( ...
            state.expression, expressionTransform);
        counterPathway = applyBlockTransform( ...
            state.pathway, pathwayTransform);
        counterModule = applyBlockTransform( ...
            state.modules, moduleTransform);
        counterCellFeatures(sum(blockWidths(1)) + (1:blockWidths(2))) = ...
            counterPathway;
        counterCellFeatures(moduleStart:sum(blockWidths(1:4))) = counterModule;
        counterKernelRow = rbfKernelRow(counterCellFeatures, ...
            data.cellKernelFeatures, cellBandwidth);
        residual = stage9_predict_counterfactual_cell( ...
            bundle.model, counterKernelRow);
        prediction = bundle.baseByGene(geneIndex) + residual(geneIndex);
        predictionRows = [predictionRows; table(caseNumber, ...
            cellNames(cellIndex), geneNames(geneIndex), intensity, ...
            prediction, prediction - originalPrediction, ...
            state.expressionChange, norm(state.pathwayChange), ...
            norm(state.moduleChange), 'VariableNames', ...
            {'case_id','cell_line_id','gene','intensity','prediction', ...
            'prediction_change','expression_change','pathway_change_norm', ...
            'module_change_norm'})]; %#ok<AGROW>
        pathwayRows = [pathwayRows; makePathwayRows( ...
            caseNumber, cellNames(cellIndex), geneNames(geneIndex), ...
            intensity, pathwayNames, state.pathwayChange)]; %#ok<AGROW>
        moduleRows = [moduleRows; makeModuleRows( ...
            caseNumber, cellNames(cellIndex), geneNames(geneIndex), ...
            intensity, state.moduleChange)]; %#ok<AGROW>
    end
    strongest = max(abs(moduleRows.module_change( ...
        moduleRows.case_id == caseNumber)));
    caseRows = [caseRows; table(caseNumber, cellNames(cellIndex), ...
        geneNames(geneIndex), originalPrediction, knockoutZ, ...
        sum(geneMembership(geneIndex, :)), strongest, ...
        'VariableNames', {'case_id','cell_line_id','gene', ...
        'baseline_prediction','knockout_z','pathway_count', ...
        'max_module_change'})]; %#ok<AGROW>
end

writetable(caseRows, fullfile(resultsDir, filePrefix + ...
    "_case_summary.csv"));
writetable(predictionRows, fullfile(resultsDir, filePrefix + ...
    "_counterfactual_predictions.csv"));
writetable(pathwayRows, fullfile(resultsDir, filePrefix + ...
    "_case_pathway_changes.csv"));
writetable(moduleRows, fullfile(resultsDir, filePrefix + ...
    "_case_module_changes.csv"));
fprintf("阶段9完成：生成 %d 个高预测值案例、%d 条强度结果。\n", ...
    height(caseRows), height(predictionRows));
end

function encoder = rebuildModuleEncoder(leafPathways, seed)
% 按阶段3同一随机种子和因子顺序重建冻结模块编码器。
rng(seed, "twister");
pathwayCenter = mean(leafPathways, 1);
pathwayScale = std(leafPathways, 0, 1);
pathwayScale(pathwayScale < 1e-12) = 1;
factorInput = (leafPathways - pathwayCenter) ./ pathwayScale;
factorCenter = mean(factorInput, 1);
factorScale = std(factorInput, 0, 1);
factorScale(factorScale < 1e-12) = 1;
% 阶段3已经记录一条唯一方差触及下界；重建时只关闭同一条已知提示。
warningId = "stats:factoran:ZeroVarianceSignificance";
warningState = warning("query", warningId);
restoreWarning = onCleanup(@() warning(warningState.state, warningId));
warning("off", warningId);
[loadings, specificVariance, ~, ~, ~] = factoran(factorInput, 8, ...
    Rotate="varimax", Scores="Bartlett");
clear restoreWarning
for moduleNumber = 1:size(loadings, 2)
    [~, strongest] = max(abs(loadings(:, moduleNumber)));
    if loadings(strongest, moduleNumber) < 0
        loadings(:, moduleNumber) = -loadings(:, moduleNumber);
    end
end
encoder = struct(pathwayCenter=pathwayCenter, ...
    pathwayScale=pathwayScale, factorCenter=factorCenter, ...
    factorScale=factorScale, specificVariance=specificVariance, ...
    loadings=loadings);
end

function transformer = fitBlockTransform(block, trainingMask)
% 只用冻结训练细胞拟合阶段7同款填补和标准化参数。
fillValue = zeros(1, size(block, 2));
filled = block;
for columnNumber = 1:size(block, 2)
    observed = block(trainingMask, columnNumber);
    observed = observed(isfinite(observed));
    if isempty(observed)
        fillValue(columnNumber) = 0;
    else
        fillValue(columnNumber) = median(observed);
    end
    missing = ~isfinite(filled(:, columnNumber));
    filled(missing, columnNumber) = fillValue(columnNumber);
end
center = mean(filled(trainingMask, :), 1);
scale = std(filled(trainingMask, :), 0, 1);
scale(scale < 1e-12 | ~isfinite(scale)) = 1;
transformer = struct(fillValue=fillValue, center=center, scale=scale, ...
    divisor=sqrt(size(block, 2)));
end

function transformed = applyBlockTransform(block, transformer)
% 应用冻结变换，不让反事实细胞改变训练期标准化参数。
transformed = block;
for columnNumber = 1:size(transformed, 2)
    missing = ~isfinite(transformed(:, columnNumber));
    transformed(missing, columnNumber) = transformer.fillValue(columnNumber);
end
transformed = (transformed - transformer.center) ./ transformer.scale;
transformed = transformed / transformer.divisor;
end

function bandwidth = rbfBandwidth(reference)
% 只计算一次阶段7细胞核使用的中位距离。
upperValues = pdist2(reference, reference, "squaredeuclidean");
upperValues = upperValues(triu(true(size(upperValues)), 1));
positiveValues = upperValues(upperValues > 1e-12 & isfinite(upperValues));
bandwidth = median(positiveValues);
end

function row = rbfKernelRow(features, reference, bandwidth)
% 用冻结带宽生成一个反事实细胞核行。
squaredDistance = sum((reference - features) .^ 2, 2)';
row = exp(-squaredDistance / (2 * bandwidth));
end

function selected = selectDistinctCases(rows, cellIndex, geneIndex, count)
% 按模型预测从高到低选取不同细胞和基因的案例，不查看验证真值。
selected = zeros(0, 1);
usedCells = zeros(0, 1);
usedGenes = zeros(0, 1);
for rowNumber = rows'
    if ~ismember(cellIndex(rowNumber), usedCells) && ...
            ~ismember(geneIndex(rowNumber), usedGenes)
        selected(end + 1, 1) = rowNumber; %#ok<AGROW>
        usedCells(end + 1, 1) = cellIndex(rowNumber); %#ok<AGROW>
        usedGenes(end + 1, 1) = geneIndex(rowNumber); %#ok<AGROW>
    end
    if numel(selected) == count
        break
    end
end
end

function rows = makePathwayRows(caseId, cellName, geneName, ...
    intensity, pathwayNames, changes)
rows = table(repmat(caseId, numel(pathwayNames), 1), ...
    repmat(cellName, numel(pathwayNames), 1), ...
    repmat(geneName, numel(pathwayNames), 1), ...
    repmat(intensity, numel(pathwayNames), 1), pathwayNames, changes(:), ...
    'VariableNames', {'case_id','cell_line_id','gene','intensity', ...
    'pathway_name','pathway_change'});
end

function rows = makeModuleRows(caseId, cellName, geneName, intensity, changes)
names = "module_" + compose("%02d", (1:numel(changes))');
rows = table(repmat(caseId, numel(changes), 1), ...
    repmat(cellName, numel(changes), 1), ...
    repmat(geneName, numel(changes), 1), ...
    repmat(intensity, numel(changes), 1), names, changes(:), ...
    'VariableNames', {'case_id','cell_line_id','gene','intensity', ...
    'module_name','module_change'});
end
