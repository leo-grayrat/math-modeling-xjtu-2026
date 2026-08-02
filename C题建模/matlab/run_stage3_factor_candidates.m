function run_stage3_factor_candidates(seed, candidateCounts)
%RUN_STAGE3_FACTOR_CANDIDATES Compare interpretable factor counts.

arguments
    seed (1, 1) double {mustBeInteger, mustBeNonnegative} = 42
    candidateCounts (1, :) double {mustBeInteger, mustBePositive} = [8, 12, 15, 18]
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
assert(all(isfinite(X), "all"), "Standardized leaf pathway matrix is invalid.");

summaryTables = cell(numel(candidateCounts), 1);
topTables = cell(numel(candidateCounts), 1);

for candidateIndex = 1:numel(candidateCounts)
    nFactors = candidateCounts(candidateIndex);
    fprintf("Fitting %d-factor model...\n", nFactors);
    [loadings, specificVariance, ~, stats] = factoran(X, nFactors, ...
        Rotate="varimax", Scores="Bartlett");

    [summaryTables{candidateIndex}, topTables{candidateIndex}] = ...
        summarizeCandidate(loadings, specificVariance, stats, ...
        leafNames, leafDescription, nFactors);

    factorNames = "factor_" + compose("%02d", 1:nFactors);
    loadingTable = array2table(loadings, VariableNames=factorNames);
    loadingTable = addvars(loadingTable, leafDescription, ...
        Before=1, NewVariableNames="description");
    loadingTable = addvars(loadingTable, leafNames, ...
        Before=1, NewVariableNames="pathway_name");
    writetable(loadingTable, fullfile(resultsDir, ...
        "stage3_factor_loadings_k" + nFactors + ".csv"));
end

candidateSummary = vertcat(summaryTables{:});
topPathways = vertcat(topTables{:});
summaryPath = fullfile(resultsDir, "stage3_factor_candidate_summary.csv");
topPath = fullfile(resultsDir, "stage3_factor_candidate_top_pathways.csv");

if isfile(summaryPath)
    existingSummary = readtable(summaryPath, TextType="string");
    existingSummary(ismember(existingSummary.n_factors, candidateCounts), :) = [];
    candidateSummary = [existingSummary; candidateSummary];
end
if isfile(topPath)
    existingTop = readtable(topPath, TextType="string");
    existingTop(ismember(existingTop.n_factors, candidateCounts), :) = [];
    topPathways = [existingTop; topPathways];
end

candidateSummary = sortrows(candidateSummary, "n_factors");
topPathways = sortrows(topPathways, ["n_factors", "factor", "rank"]);
writetable(candidateSummary, summaryPath);
writetable(topPathways, topPath);

disp(candidateSummary);
fprintf("Candidate factor results written to %s\n", resultsDir);
end

function [summary, topTable] = summarizeCandidate(loadings, specificVariance, ...
    stats, pathwayNames, descriptions, nFactors)
nPathways = size(loadings, 1);
communality = sum(loadings .^ 2, 2);
sortedAbs = sort(abs(loadings), 2, "descend");
loadingGap = sortedAbs(:, 1) - sortedAbs(:, 2);
supportedFactors = sum(sum(abs(loadings) >= 0.35, 1) >= 3);

topLevel = descriptions;
hasSeparator = contains(topLevel, ".");
topLevel(hasSeparator) = extractBefore(topLevel(hasSeparator), ".");

topRows = cell(nFactors, 1);
purity = zeros(nFactors, 1);
for factorIndex = 1:nFactors
    [~, order] = sort(abs(loadings(:, factorIndex)), "descend");
    topIndex = order(1:min(8, nPathways));
    categories = topLevel(topIndex);
    categoryCounts = groupcounts(categorical(categories));
    purity(factorIndex) = max(categoryCounts) / numel(topIndex);

    topRows{factorIndex} = table( ...
        repmat(nFactors, numel(topIndex), 1), ...
        repmat(factorIndex, numel(topIndex), 1), ...
        (1:numel(topIndex))', pathwayNames(topIndex), ...
        descriptions(topIndex), topLevel(topIndex), ...
        loadings(topIndex, factorIndex), abs(loadings(topIndex, factorIndex)), ...
        'VariableNames', {'n_factors', 'factor', 'rank', 'pathway_name', ...
        'description', 'top_level', 'loading', 'abs_loading'});
end
topTable = vertcat(topRows{:});

summary = table(nFactors, mean(communality), median(communality), ...
    sum(communality >= 0.50), mean(loadingGap), supportedFactors, ...
    mean(purity), min(specificVariance), sum(specificVariance <= 0.0051), ...
    getStatOrNaN(stats, "loglike"), getStatOrNaN(stats, "chisq"), ...
    getStatOrNaN(stats, "dfe"), getStatOrNaN(stats, "p"), ...
    'VariableNames', {'n_factors', 'mean_communality', ...
    'median_communality', 'pathways_communality_ge_050', ...
    'mean_top_loading_gap', 'factors_with_3_loadings_ge_035', ...
    'mean_top8_category_purity', 'min_specific_variance', ...
    'specific_variance_at_lower_bound', 'model_loglike', ...
    'model_chi2', 'model_df', 'model_p'});
end

function value = getStatOrNaN(stats, fieldName)
if isfield(stats, fieldName)
    value = stats.(fieldName);
else
    value = NaN;
end
end
