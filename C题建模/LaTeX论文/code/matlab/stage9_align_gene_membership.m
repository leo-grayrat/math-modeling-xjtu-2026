function aligned = stage9_align_gene_membership( ...
    expressionMembership, expressionGeneIndex)
%STAGE9_ALIGN_GENE_MEMBERSHIP 把表达列顺序转换为目标基因顺序。

arguments
    expressionMembership logical
    expressionGeneIndex (:, 1) double {mustBeInteger, mustBePositive}
end

assert(all(expressionGeneIndex <= size(expressionMembership, 1)), ...
    "stage9:MembershipIndex", "目标基因的表达列位置超出成员矩阵范围。");
aligned = expressionMembership(expressionGeneIndex, :);
end
