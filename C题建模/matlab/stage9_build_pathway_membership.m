function [membership, pathwayGeneCount] = ...
    stage9_build_pathway_membership(expressionGeneNames, ...
    geneMetadata, pathwayNames)
%STAGE9_BUILD_PATHWAY_MEMBERSHIP 按题目注释建立基因到通路的连接表。

arguments
    expressionGeneNames (:, 1) string
    geneMetadata table
    pathwayNames (:, 1) string
end

assert(all(ismember(["gene_symbol", "pathways"], ...
    string(geneMetadata.Properties.VariableNames))), ...
    "stage9:GeneMetadataColumns", "基因元数据缺少名称或通路字段。");
[knownGenes, metadataIndex] = ismember(expressionGeneNames, ...
    string(geneMetadata.gene_symbol));
assert(all(knownGenes), "stage9:ExpressionGeneMetadata", ...
    "部分表达基因无法连接到基因元数据。");

membership = false(numel(expressionGeneNames), numel(pathwayNames));
normalizedPathwayNames = normalizePathwayName(pathwayNames);
annotations = string(geneMetadata.pathways(metadataIndex));
for geneNumber = 1:numel(expressionGeneNames)
    if ismissing(annotations(geneNumber)) || ...
            strlength(strtrim(annotations(geneNumber))) == 0
        continue
    end
    branches = split(annotations(geneNumber), "|");
    tokens = strings(0, 1);
    for branchNumber = 1:numel(branches)
        tokens = [tokens; split(branches(branchNumber), ">")] ; %#ok<AGROW>
    end
    tokens = normalizePathwayName(tokens);
    [knownPathways, pathwayIndex] = ismember( ...
        unique(tokens), normalizedPathwayNames);
    membership(geneNumber, pathwayIndex(knownPathways)) = true;
end

pathwayGeneCount = sum(membership, 1);
end

function names = normalizePathwayName(names)
names = strtrim(names);
names = regexprep(names, "[^A-Za-z0-9]+", "_");
names = regexprep(names, "^_+|_+$", "");
end
