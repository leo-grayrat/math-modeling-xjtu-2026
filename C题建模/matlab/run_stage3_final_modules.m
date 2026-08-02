function run_stage3_final_modules(seed)
% 阶段3：导出选定的八维功能模块表示。

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
end

rng(seed, "twister");

matlabDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(matlabDir);
repoRoot = fileparts(projectDir);
dataRoot = fullfile(repoRoot, "2026年校赛题目", "C", "数据文件");
resultsDir = fullfile(projectDir, "results");

pathways = readtable(fullfile(dataRoot, "features", ...
    "cell_pathway_scores_zscore.csv"), TextType="string", ...
    VariableNamingRule="preserve");
structure = readtable(fullfile(resultsDir, "stage3_pathway_structure.csv"), ...
    TextType="string", VariableNamingRule="preserve");

pathwayNames = string(pathways.Properties.VariableNames(2:end))';
[matched, structureLocation] = ismember(pathwayNames, structure.pathway_name);
assert(all(matched), "Pathway structure does not match feature columns.");
structure = structure(structureLocation, :);

leafMask = logical(structure.is_leaf);
leafNames = pathwayNames(leafMask);
leafDescription = structure.description(leafMask);
X = zscore(pathways{:, find(leafMask)' + 1});

nModules = 8;
[loadings, specificVariance, ~, stats, scores] = factoran(X, nModules, ...
    Rotate="varimax", Scores="Bartlett");

for moduleIndex = 1:nModules
    [~, strongest] = max(abs(loadings(:, moduleIndex)));
    if loadings(strongest, moduleIndex) < 0
        loadings(:, moduleIndex) = -loadings(:, moduleIndex);
        scores(:, moduleIndex) = -scores(:, moduleIndex);
    end
end

expectedAnchors = [
    "mtDNA_replication"
    "Fatty_acid_oxidation"
    "Xenobiotic_metabolism"
    "OXPHOS_subunits"
    "Vitamin_B12_metabolism"
    "Iron_homeostasis"
    "Creatine_metabolism"
    "TCA_cycle"
];
for moduleIndex = 1:nModules
    [~, strongest] = max(abs(loadings(:, moduleIndex)));
    assert(leafNames(strongest) == expectedAnchors(moduleIndex), ...
        "Factor order changed; module names require review.");
end

moduleId = "module_" + compose("%02d", 1:nModules)';
moduleNameZh = [
    "线粒体遗传信息维护与表达"
    "脂肪酸和支链氨基酸利用"
    "解毒与氧化还原代谢"
    "氧化磷酸化"
    "复合体II与辅因子代谢"
    "质量控制与铁稳态"
    "维生素-血红素与能量缓冲"
    "TCA-复合体II与动力学平衡"
];
moduleNameEn = [
    "Mitochondrial genetic maintenance and expression"
    "Fatty-acid and branched-chain amino-acid utilization"
    "Detoxification and redox metabolism"
    "Oxidative phosphorylation"
    "Complex II and cofactor metabolism"
    "Quality control and iron homeostasis"
    "Vitamin-heme metabolism and energy buffering"
    "TCA-Complex II-dynamics balance"
];
interpretationConfidence = [
    "high"; "high"; "high"; "high"; ...
    "medium"; "medium"; "medium"; "exploratory"
];
interpretation = [
    "由mtDNA复制、核样体、mtRNA加工和线粒体核糖体装配共同表征。"
    "由脂肪酸氧化、肉碱穿梭和支链氨基酸代谢共同表征。"
    "由外源物代谢、硫代谢、谷胱甘肽和相关氧化还原反应共同表征。"
    "由呼吸链复合体I、III、IV、V及OXPHOS亚基共同表征。"
    "主要关联维生素B12、复合体II、呼吸体装配和相关底物代谢。"
    "主要关联铁稳态、自噬、线粒体自噬和细胞器接触。"
    "主要关联肌酸、维生素A/D、血红素和细胞色素代谢。"
    "由TCA、复合体II、血红素和线粒体融合等较弱载荷共同构成，暂作探索性指标。"
];

topLevel = leafDescription;
hasSeparator = contains(topLevel, ".");
topLevel(hasSeparator) = extractBefore(topLevel(hasSeparator), ".");

topPathways = strings(nModules, 1);
topLoadings = strings(nModules, 1);
nStrongLoadings = zeros(nModules, 1);
top8Purity = zeros(nModules, 1);
for moduleIndex = 1:nModules
    nStrongLoadings(moduleIndex) = sum(abs(loadings(:, moduleIndex)) >= 0.35);
    [~, order] = sort(abs(loadings(:, moduleIndex)), "descend");
    topIndex = order(1:8);
    topPathways(moduleIndex) = join(leafNames(topIndex), "|");
    topLoadings(moduleIndex) = join(compose("%.6f", ...
        loadings(topIndex, moduleIndex)), "|");
    categoryCounts = groupcounts(categorical(topLevel(topIndex)));
    top8Purity(moduleIndex) = max(categoryCounts) / 8;
end

definitionTable = table(moduleId, moduleNameZh, moduleNameEn, ...
    interpretationConfidence, interpretation, expectedAnchors, ...
    nStrongLoadings, top8Purity, topPathways, topLoadings, ...
    'VariableNames', {'module_id', 'module_name_zh', 'module_name_en', ...
    'interpretation_confidence', 'interpretation', 'anchor_pathway', ...
    'n_loadings_ge_035', 'top8_category_purity', ...
    'top_pathways', 'top_loadings'});

loadingTable = array2table(loadings, VariableNames=moduleId');
loadingTable = addvars(loadingTable, leafDescription, ...
    Before=1, NewVariableNames="description");
loadingTable = addvars(loadingTable, leafNames, ...
    Before=1, NewVariableNames="pathway_name");

scoreTable = array2table(scores, VariableNames=moduleId');
scoreTable = addvars(scoreTable, pathways.cell_line_id, ...
    Before=1, NewVariableNames="cell_line_id");

modelSummary = table(nModules, height(pathways), numel(leafNames), ...
    mean(sum(loadings .^ 2, 2)), median(sum(loadings .^ 2, 2)), ...
    sum(sum(loadings .^ 2, 2) >= 0.50), min(specificVariance), ...
    sum(specificVariance <= 0.0051), getStatOrNaN(stats, "loglike"), ...
    getStatOrNaN(stats, "dfe"), ...
    'VariableNames', {'n_modules', 'n_cells', 'n_leaf_pathways', ...
    'mean_communality', 'median_communality', ...
    'pathways_communality_ge_050', 'min_specific_variance', ...
    'specific_variance_at_lower_bound', 'model_loglike', 'model_df'});

writetable(definitionTable, ...
    fullfile(resultsDir, "stage3_module_definitions.csv"));
writetable(loadingTable, ...
    fullfile(resultsDir, "stage3_module_loadings.csv"));
writetable(scoreTable, ...
    fullfile(resultsDir, "stage3_cell_module_scores.csv"));
writetable(modelSummary, ...
    fullfile(resultsDir, "stage3_module_model_summary.csv"));

disp(definitionTable(:, ["module_id", "module_name_zh", ...
    "interpretation_confidence", "n_loadings_ge_035"]));
fprintf("Final module representation written to %s\n", resultsDir);
end

function value = getStatOrNaN(stats, fieldName)
if isfield(stats, fieldName)
    value = stats.(fieldName);
else
    value = NaN;
end
end
