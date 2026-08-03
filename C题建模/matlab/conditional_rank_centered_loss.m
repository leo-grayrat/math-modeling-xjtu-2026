function loss = conditional_rank_centered_loss(errorMatrix)
%CONDITIONAL_RANK_CENTERED_LOSS 计算同一细胞内全部基因差值误差的等价损失。
% 每行代表一个细胞。该写法等于逐行全部两两差分平方和除以该行基因数。

arguments
    errorMatrix double
end

centeredError = errorMatrix - mean(errorMatrix, 2);
loss = sum(centeredError .^ 2, "all");
end

