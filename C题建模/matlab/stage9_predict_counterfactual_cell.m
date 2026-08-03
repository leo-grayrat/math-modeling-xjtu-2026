function residualPrediction = stage9_predict_counterfactual_cell( ...
    model, cellKernelRow)
%STAGE9_PREDICT_COUNTERFACTUAL_CELL 使用冻结模型预测一个反事实细胞。

arguments
    model (1, 1) struct
    cellKernelRow (1, :) double
end

requiredFields = ["trainingCells", "cellTrainingKernel", ...
    "trainingResidualMatrix", "geneMap", "lambda"];
assert(all(isfield(model, requiredFields)), ...
    "stage9:FrozenModelFields", "冻结模型缺少反事实预测所需的中间量。");
assert(numel(cellKernelRow) >= max(model.trainingCells), ...
    "stage9:CellKernelWidth", "反事实细胞核向量没有覆盖全部训练细胞。");

trainingKernelRow = cellKernelRow(model.trainingCells);
cellMap = trainingKernelRow / (model.cellTrainingKernel + ...
    model.lambda * eye(numel(model.trainingCells)));
residualPrediction = (cellMap * model.trainingResidualMatrix) * ...
    model.geneMap';
end
