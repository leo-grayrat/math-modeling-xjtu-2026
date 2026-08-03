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

annotations = string(geneMetadata.pathways(metadataIndex));
[membership, pathwayGeneCount] = build_gene_pathway_membership( ...
    annotations, pathwayNames);
membership = logical(membership);
end
