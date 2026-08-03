classdef conditionalRankRlsTest < matlab.unittest.TestCase
    %conditionalRankRlsTest 检查条件排序求解器的退回和差分边界。

    methods (TestClassSetup)
        function addModelFolder(testCase)
            testFolder = fileparts(mfilename("fullpath"));
            modelFolder = fullfile(fileparts(testFolder), "matlab");
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                modelFolder));
        end
    end

    methods (Test)
        function testZeroEtaReturnsStage7Model(testCase)
            cellCoordinates = (-2:2)';
            geneCoordinates = (-1.5:1:1.5)';
            cellKernel = exp(-squareform(pdist(cellCoordinates)) .^ 2);
            geneKernel = exp(-squareform(pdist(geneCoordinates)) .^ 2);
            [trainCell, trainGene] = ndgrid(1:4, 1:3);
            target = sin(trainCell(:)) + cos(trainGene(:));

            stage7Model = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
                trainCell(:), trainGene(:), target, 0.1);
            rankModel = stage7_fit_conditional_rank_kernel( ...
                cellKernel, geneKernel, trainCell(:), trainGene(:), ...
                target, 0.1, 0);

            testCase.verifyEqual(rankModel.predictionMatrix, ...
                stage7Model.predictionMatrix, AbsTol=1e-11);
            testCase.verifyEqual(rankModel.geneMap, ...
                stage7Model.geneMap, AbsTol=1e-11);
        end

        function testCenteredLossMatchesExplicitPairs(testCase)
            errors = [0.4, -0.1, 0.8, 0.2; -0.5, 0.3, 0.1, 0.9];
            explicitLoss = sum(pdist(errors(1, :)') .^ 2) / 4 + ...
                sum(pdist(errors(2, :)') .^ 2) / 4;

            fastLoss = conditional_rank_centered_loss(errors);

            testCase.verifyEqual(fastLoss, explicitLoss, AbsTol=1e-12);
        end

        function testLossDoesNotCompareDifferentCells(testCase)
            errors = [0.4, -0.1, 0.8; -0.5, 0.3, 0.1];
            shifted = errors + [7; -4];

            originalLoss = conditional_rank_centered_loss(errors);
            shiftedLoss = conditional_rank_centered_loss(shifted);

            testCase.verifyEqual(shiftedLoss, originalLoss, AbsTol=1e-12);
        end
    end
end
