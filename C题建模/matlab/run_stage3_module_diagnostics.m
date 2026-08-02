function run_stage3_module_diagnostics()
%RUN_STAGE3_MODULE_DIAGNOSTICS Check whether pathway data suit factor analysis.

matlabDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(matlabDir);
repoRoot = fileparts(projectDir);
dataRoot = fullfile(repoRoot, "2026年校赛题目", "C", "数据文件");
resultsDir = fullfile(projectDir, "results");

pathways = readtable(fullfile(dataRoot, "features", ...
    "cell_pathway_scores_zscore.csv"), TextType="string", ...
    VariableNamingRule="preserve");
metadata = readtable(fullfile(dataRoot, "metadata", "pathway_metadata.csv"), ...
    TextType="string", VariableNamingRule="preserve");

pathwayNames = string(pathways.Properties.VariableNames(2:end))';
X = pathways{:, 2:end};
assert(all(isfinite(X), "all"), "Pathway matrix contains non-finite values.");

[matched, metadataLocation] = ismember(pathwayNames, metadata.pathway_name);
assert(all(matched), "At least one pathway feature lacks metadata.");
alignedMetadata = metadata(metadataLocation, :);

isLeaf = identifyLeafPathways(alignedMetadata.description);
assert(sum(isLeaf) >= 2, "Leaf pathway set is too small.");

[summaryAll, eigenAll, pairsAll] = analyzePathwaySet(X, pathwayNames, "all_pathways");
[summaryLeaf, eigenLeaf, pairsLeaf] = analyzePathwaySet( ...
    X(:, isLeaf), pathwayNames(isLeaf), "leaf_pathways");

pathwayStructure = alignedMetadata;
pathwayStructure.is_leaf = isLeaf;
writetable(pathwayStructure, fullfile(resultsDir, "stage3_pathway_structure.csv"));
writetable([summaryAll; summaryLeaf], ...
    fullfile(resultsDir, "stage3_factor_suitability.csv"));
writetable(eigenAll, fullfile(resultsDir, "stage3_eigenvalues_all_pathways.csv"));
writetable(eigenLeaf, fullfile(resultsDir, "stage3_eigenvalues_leaf_pathways.csv"));
writetable([pairsAll; pairsLeaf], ...
    fullfile(resultsDir, "stage3_high_correlation_pairs.csv"));

disp([summaryAll; summaryLeaf]);
fprintf("Leaf pathways: %d of %d.\n", sum(isLeaf), numel(isLeaf));
fprintf("Diagnostics written to %s\n", resultsDir);
end

function isLeaf = identifyLeafPathways(description)
description = string(description);
isLeaf = true(numel(description), 1);
for i = 1:numel(description)
    descendantPrefix = description(i) + ".";
    isLeaf(i) = ~any(startsWith(description, descendantPrefix));
end
end

function [summary, eigenTable, pairTable] = analyzePathwaySet(X, names, setName)
[n, p] = size(X);
columnStd = std(X, 0, 1);
assert(all(columnStd > eps), "Constant pathway found in %s.", setName);

R = corrcoef(X);
eigenvalues = sort(real(eig(R)), "descend");
contribution = eigenvalues / sum(eigenvalues);
cumulative = cumsum(contribution);

inverseR = pinv(R);
scale = sqrt(diag(inverseR) * diag(inverseR)');
partial = -inverseR ./ scale;
partial(1:p+1:end) = 0;
simple = R;
simple(1:p+1:end) = 0;
kmo = sum(simple .^ 2, "all") / ...
    (sum(simple .^ 2, "all") + sum(partial .^ 2, "all"));

logDeterminant = sum(log(max(eigenvalues, realmin)));
bartlettDf = p * (p - 1) / 2;
bartlettChi2 = -(n - 1 - (2 * p + 5) / 6) * logDeterminant;
bartlettP = 1 - chi2cdf(bartlettChi2, bartlettDf);

kaiserFactors = sum(eigenvalues > 1);
factors80 = find(cumulative >= 0.80, 1, "first");
factors90 = find(cumulative >= 0.90, 1, "first");
numericalRank = rank(R, 1e-10);

[firstIndex, secondIndex] = find(triu(abs(R) >= 0.95, 1));
pairCorrelation = R(sub2ind(size(R), firstIndex, secondIndex));

summary = table(string(setName), n, p, sum(firstIndex > 0), kmo, ...
    bartlettChi2, bartlettDf, bartlettP, numericalRank, ...
    kaiserFactors, factors80, factors90, min(eigenvalues), ...
    'VariableNames', {'pathway_set', 'n_cells', 'n_pathways', ...
    'pairs_abs_corr_ge_095', 'kmo', 'bartlett_chi2', 'bartlett_df', ...
    'bartlett_p', 'correlation_rank', 'kaiser_factors', ...
    'factors_for_80pct', 'factors_for_90pct', 'min_eigenvalue'});

eigenTable = table(repmat(string(setName), p, 1), (1:p)', eigenvalues, ...
    contribution, cumulative, ...
    'VariableNames', {'pathway_set', 'component', 'eigenvalue', ...
    'contribution', 'cumulative_contribution'});

pairTable = table(repmat(string(setName), numel(firstIndex), 1), ...
    names(firstIndex), names(secondIndex), pairCorrelation, ...
    'VariableNames', {'pathway_set', 'pathway_1', 'pathway_2', 'correlation'});
end
