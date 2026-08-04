function result = stage7_calculate_metrics(answer, prediction)
%STAGE7_CALCULATE_METRICS 按题目官方口径计算内层筛选指标。

arguments
    answer table
    prediction (:, 1) double
end

assert(height(answer) == numel(prediction), ...
    "stage7:MetricRowCount", "评分真值与预测行数不一致。");
[cellGroup, ~] = findgroups(answer.cell_line_id);
nCells = max(cellGroup);
spearmanValues = zeros(nCells, 1);
precisionValues = zeros(nCells, 3);
ndcgValues = zeros(nCells, 3);
kValues = [5, 10, 15];

for cellNumber = 1:nCells
    mask = cellGroup == cellNumber;
    truth = answer.label(mask);
    predicted = prediction(mask);
    genes = answer.perturbation_gene(mask);
    nRows = numel(truth);
    originalOrder = (1:nRows)';
    predictedRank = tiedrank(-predicted);
    trueRank = tiedrank(-truth);
    spearmanValues(cellNumber) = pearsonOrZero(predictedRank, trueRank);
    [~, predictedOrder] = sortrows([-predicted, originalOrder], [1, 2]);
    [~, trueOrder] = sortrows([-truth, originalOrder], [1, 2]);

    for kNumber = 1:numel(kValues)
        k = kValues(kNumber);
        take = min(k, nRows);
        predictedTop = predictedOrder(1:take);
        trueTopGenes = genes(trueOrder(1:take));
        precisionValues(cellNumber, kNumber) = ...
            sum(ismember(genes(predictedTop), trueTopGenes)) / k;

        relevance = max(k - trueRank(predictedTop) + 1, 0);
        relevance(trueRank(predictedTop) > k) = 0;
        discounts = log2((2:(take + 1))');
        idealRelevance = (k:-1:1)';
        idealDiscounts = log2((2:(k + 1))');
        ndcgValues(cellNumber, kNumber) = ...
            sum((2 .^ relevance - 1) ./ discounts) / ...
            sum((2 .^ idealRelevance - 1) ./ idealDiscounts);
    end
end

result.rho_macro = mean(spearmanValues);
result.spearman_score = (result.rho_macro + 1) / 2;
result.precision_score = mean(precisionValues, "all");
result.ndcg_score = mean(ndcgValues, "all");
result.rmse = sqrt(mean((prediction - answer.label) .^ 2));
labelScale = std(answer.label, 1);
result.rmse_score = 1 / (1 + result.rmse / (labelScale + 1e-12));
result.final_score = 100 * (0.30 * result.spearman_score + ...
    0.30 * result.ndcg_score + 0.25 * result.precision_score + ...
    0.15 * result.rmse_score);
end

function value = pearsonOrZero(x, y)
x = x - mean(x);
y = y - mean(y);
denominator = sqrt(sum(x .^ 2) * sum(y .^ 2));
if denominator <= 1e-12
    value = 0;
else
    value = sum(x .* y) / denominator;
end
end
