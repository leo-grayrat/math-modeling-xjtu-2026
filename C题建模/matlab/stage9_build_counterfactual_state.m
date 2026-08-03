function state = stage9_build_counterfactual_state( ...
    expressionRow, pathwayRow, targetExpressionColumn, knockoutZ, ...
    targetMembership, pathwayGeneCount, intensity, moduleEncoder)
%STAGE9_BUILD_COUNTERFACTUAL_STATE 按通路均值定义生成虚拟敲除状态。

arguments
    expressionRow (1, :) double
    pathwayRow (1, :) double
    targetExpressionColumn (1, 1) double {mustBeInteger, mustBePositive}
    knockoutZ (1, 1) double {mustBeFinite}
    targetMembership (1, :) logical
    pathwayGeneCount (1, :) double {mustBePositive}
    intensity (1, 1) double {mustBeInRange(intensity, 0, 1)}
    moduleEncoder (1, 1) struct
end

assert(targetExpressionColumn <= numel(expressionRow), ...
    "stage9:ExpressionColumn", "待敲基因的表达列超出范围。");
assert(numel(pathwayRow) == numel(targetMembership) && ...
    numel(pathwayRow) == numel(pathwayGeneCount), ...
    "stage9:PathwayWidth", "通路、成员关系和通路基因数的列数不一致。");
assert(isfield(moduleEncoder, "leafColumns"), ...
    "stage9:LeafColumns", "模块编码器缺少末级通路位置。");

state.expression = expressionRow;
originalExpression = expressionRow(targetExpressionColumn);
counterfactualExpression = originalExpression + ...
    intensity * (knockoutZ - originalExpression);
state.expression(targetExpressionColumn) = counterfactualExpression;
state.expressionChange = counterfactualExpression - originalExpression;

state.pathwayChange = zeros(size(pathwayRow));
state.pathwayChange(targetMembership) = state.expressionChange ./ ...
    pathwayGeneCount(targetMembership);
state.pathway = pathwayRow + state.pathwayChange;

leafColumns = moduleEncoder.leafColumns;
state.originalModules = stage9_apply_bartlett_scores( ...
    pathwayRow(leafColumns), moduleEncoder);
state.modules = stage9_apply_bartlett_scores( ...
    state.pathway(leafColumns), moduleEncoder);
state.moduleChange = state.modules - state.originalModules;
state.intensity = intensity;
end
