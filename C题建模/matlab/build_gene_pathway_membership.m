function [membership, pathwayGeneCount] = ...
    build_gene_pathway_membership(pathwayAnnotations, pathwayNames)
%build_gene_pathway_membership 将基因通路注释映射到题目的通路列。
arguments
    pathwayAnnotations (:, 1) string
    pathwayNames string
end

pathwayAnnotations = string(pathwayAnnotations);
pathwayNames = string(pathwayNames(:));
normalizedPathwayNames = normalizePathwayName(pathwayNames);
assert(numel(unique(normalizedPathwayNames)) == numel(pathwayNames), ...
    "通路名称标准化后出现重复，无法唯一确定通路列。");

membership = zeros(numel(pathwayAnnotations), numel(pathwayNames));
for rowNumber = 1:numel(pathwayAnnotations)
    annotation = pathwayAnnotations(rowNumber);
    if ismissing(annotation) || strlength(strtrim(annotation)) == 0
        continue
    end

    branches = split(annotation, "|");
    tokens = strings(0, 1);
    for branchNumber = 1:numel(branches)
        tokens = [tokens; split(branches(branchNumber), ">")]; %#ok<AGROW>
    end
    tokens = normalizePathwayName(tokens);
    [known, location] = ismember(unique(tokens), normalizedPathwayNames);
    membership(rowNumber, location(known)) = 1;
end

pathwayGeneCount = sum(membership, 1);
end

function names = normalizePathwayName(names)
% MATLAB 会把部分表头中的连字符改成下划线，两侧必须使用同一规则。
names = strtrim(names);
names = regexprep(names, "[^A-Za-z0-9]+", "_");
names = regexprep(names, "^_+|_+$", "");
end
