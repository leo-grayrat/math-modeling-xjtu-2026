function scores = stage9_apply_bartlett_scores(pathwayRows, encoder)
%STAGE9_APPLY_BARTLETT_SCORES 使用阶段3冻结参数计算模块得分。

arguments
    pathwayRows double
    encoder (1, 1) struct
end

requiredFields = ["pathwayCenter", "pathwayScale", "factorCenter", ...
    "factorScale", "specificVariance", "loadings"];
assert(all(isfield(encoder, requiredFields)), ...
    "stage9:ModuleEncoderFields", "模块编码器字段不完整。");
assert(size(pathwayRows, 2) == numel(encoder.pathwayCenter), ...
    "stage9:ModulePathwayCount", "通路列数与冻结编码器不一致。");

factorInput = (pathwayRows - encoder.pathwayCenter) ./ ...
    encoder.pathwayScale;
standardized = (factorInput - encoder.factorCenter) ./ ...
    encoder.factorScale;
inverseSqrtVariance = diag(1 ./ sqrt(encoder.specificVariance(:)));
scoreCoefficient = inverseSqrtVariance / ...
    (encoder.loadings' * inverseSqrtVariance);
scores = standardized * scoreCoefficient;
end
