function run_stage3_pca_control()
%RUN_STAGE3_PCA_CONTROL Build the planned PCA comparison on leaf pathways.

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

[coefficients, scores, eigenvalues, ~, explained] = pca(X);
correlationLoadings = coefficients .* sqrt(eigenvalues');

maxComponents = 18;
for component = 1:maxComponents
    [~, strongest] = max(abs(correlationLoadings(:, component)));
    if correlationLoadings(strongest, component) < 0
        coefficients(:, component) = -coefficients(:, component);
        scores(:, component) = -scores(:, component);
        correlationLoadings(:, component) = -correlationLoadings(:, component);
    end
end

topLevel = leafDescription;
hasSeparator = contains(topLevel, ".");
topLevel(hasSeparator) = extractBefore(topLevel(hasSeparator), ".");

[componentSummary, topPathways] = summarizeComponents( ...
    correlationLoadings(:, 1:maxComponents), eigenvalues, explained, ...
    leafNames, leafDescription, topLevel);

candidateCounts = [12, 15, 18];
candidateRows = cell(numel(candidateCounts), 1);
for candidateIndex = 1:numel(candidateCounts)
    nComponents = candidateCounts(candidateIndex);
    loadingSubset = correlationLoadings(:, 1:nComponents);
    communality = sum(loadingSubset .^ 2, 2);
    sortedAbs = sort(abs(loadingSubset), 2, "descend");
    loadingGap = sortedAbs(:, 1) - sortedAbs(:, 2);

    candidateRows{candidateIndex} = table( ...
        nComponents, sum(explained(1:nComponents)), ...
        mean(communality), median(communality), ...
        sum(communality >= 0.50), mean(loadingGap), ...
        sum(componentSummary.n_loadings_ge_035(1:nComponents) >= 3), ...
        mean(componentSummary.top8_category_purity(1:nComponents)), ...
        'VariableNames', {'n_components', 'cumulative_explained_pct', ...
        'mean_communality', 'median_communality', ...
        'pathways_communality_ge_050', 'mean_top_loading_gap', ...
        'components_with_3_loadings_ge_035', ...
        'mean_top8_category_purity'});
end
candidateSummary = vertcat(candidateRows{:});

componentNames = "PC_" + compose("%02d", 1:maxComponents);
loadingTable = array2table(correlationLoadings(:, 1:maxComponents), ...
    VariableNames=componentNames);
loadingTable = addvars(loadingTable, leafDescription, ...
    Before=1, NewVariableNames="description");
loadingTable = addvars(loadingTable, leafNames, ...
    Before=1, NewVariableNames="pathway_name");

scoreTable = array2table(scores(:, 1:maxComponents), ...
    VariableNames=componentNames);
scoreTable = addvars(scoreTable, pathways.cell_line_id, ...
    Before=1, NewVariableNames="cell_line_id");

writetable(candidateSummary, ...
    fullfile(resultsDir, "stage3_pca_candidate_summary.csv"));
writetable(componentSummary, ...
    fullfile(resultsDir, "stage3_pca_component_summary.csv"));
writetable(topPathways, ...
    fullfile(resultsDir, "stage3_pca_top_pathways.csv"));
writetable(loadingTable, ...
    fullfile(resultsDir, "stage3_pca_correlation_loadings_k18.csv"));
writetable(scoreTable, ...
    fullfile(resultsDir, "stage3_pca_scores_k18.csv"));

disp(candidateSummary);
fprintf("PCA control results written to %s\n", resultsDir);
end

function [componentSummary, topTable] = summarizeComponents(loadings, ...
    eigenvalues, explained, pathwayNames, descriptions, topLevel)
nComponents = size(loadings, 2);
topRows = cell(nComponents, 1);
purity = zeros(nComponents, 1);
nStrong = zeros(nComponents, 1);

for component = 1:nComponents
    nStrong(component) = sum(abs(loadings(:, component)) >= 0.35);
    [~, order] = sort(abs(loadings(:, component)), "descend");
    topIndex = order(1:min(8, numel(order)));
    categories = topLevel(topIndex);
    categoryCounts = groupcounts(categorical(categories));
    purity(component) = max(categoryCounts) / numel(topIndex);

    topRows{component} = table( ...
        repmat(component, numel(topIndex), 1), ...
        (1:numel(topIndex))', pathwayNames(topIndex), ...
        descriptions(topIndex), topLevel(topIndex), ...
        loadings(topIndex, component), abs(loadings(topIndex, component)), ...
        'VariableNames', {'component', 'rank', 'pathway_name', ...
        'description', 'top_level', 'loading', 'abs_loading'});
end
topTable = vertcat(topRows{:});

componentSummary = table((1:nComponents)', eigenvalues(1:nComponents), ...
    explained(1:nComponents), cumsum(explained(1:nComponents)), ...
    nStrong, purity, ...
    'VariableNames', {'component', 'eigenvalue', 'explained_pct', ...
    'cumulative_explained_pct', 'n_loadings_ge_035', ...
    'top8_category_purity'});
end
