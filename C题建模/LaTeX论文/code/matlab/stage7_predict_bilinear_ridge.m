function prediction = stage7_predict_bilinear_ridge( ...
    model, cellFeatures, geneFeatures, ownExpression, cellIndex, geneIndex)
%STAGE7_PREDICT_BILINEAR_RIDGE 为指定的细胞—基因组合计算残差修正。

arguments
    model struct
    cellFeatures double
    geneFeatures double
    ownExpression double
    cellIndex (:, 1) double {mustBeInteger, mustBePositive}
    geneIndex (:, 1) double {mustBeInteger, mustBePositive}
end

assert(numel(cellIndex) == numel(geneIndex), ...
    "stage7:PredictionRowCount", "预测时的细胞和基因索引行数不一致。");
assert(size(model.interaction, 1) == size(cellFeatures, 2) && ...
    size(model.interaction, 2) == size(geneFeatures, 2), ...
    "stage7:PredictionFeatureSize", "预测特征列数与模型不一致。");

cellProjection = cellFeatures(cellIndex, :) * model.interaction;
geneRows = geneFeatures(geneIndex, :);
linearIndex = sub2ind(size(ownExpression), cellIndex, geneIndex);
prediction = sum(cellProjection .* geneRows, 2) + ...
    model.ownExpressionCoefficient * ownExpression(linearIndex);
end
