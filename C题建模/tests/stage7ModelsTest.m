classdef stage7ModelsTest < matlab.unittest.TestCase
    %stage7ModelsTest 检查阶段7两个公开求解接口的基本行为。

    methods (TestClassSetup)
        function addModelFolder(testCase)
            testFolder = fileparts(mfilename("fullpath"));
            modelFolder = fullfile(fileparts(testFolder), "matlab");
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                modelFolder));
        end
    end

    methods (TestMethodSetup)
        function fixRandomSeed(testCase)
            oldState = rng;
            testCase.addTeardown(@() rng(oldState));
            rng(42, "twister");
        end
    end

    methods (Test)
        function testBilinearModelRecoversSignal(testCase)
            cellFeatures = [ones(7, 1), randn(7, 2)];
            geneFeatures = [ones(6, 1), randn(6, 2)];
            trueInteraction = [0.2, -0.1, 0.3; 0.4, 0.2, -0.2; -0.3, 0.1, 0.5];
            ownExpression = randn(7, 6);
            truthMatrix = cellFeatures * trueInteraction * geneFeatures' + ...
                0.35 * ownExpression;
            [cellGrid, geneGrid] = ndgrid(1:7, 1:6);
            model = stage7_fit_bilinear_ridge(cellFeatures, geneFeatures, ...
                ownExpression, cellGrid(:), geneGrid(:), truthMatrix(:), ...
                1e-8, Tolerance=1e-9, MaxIterations=200);
            prediction = stage7_predict_bilinear_ridge(model, ...
                cellFeatures, geneFeatures, ownExpression, ...
                cellGrid(:), geneGrid(:));
            testCase.verifyLessThan(sqrt(mean((prediction - truthMatrix(:)) .^ 2)), ...
                1e-5);
        end

        function testBilinearModelPredictsColdCombination(testCase)
            cellFeatures = [ones(5, 1), (-2:2)'];
            geneFeatures = [ones(4, 1), (-1.5:1:1.5)'];
            ownExpression = reshape(linspace(-1, 1, 20), 5, 4);
            truthMatrix = cellFeatures * [0.1, 0.3; -0.2, 0.4] * ...
                geneFeatures' + 0.2 * ownExpression;
            [trainCell, trainGene] = ndgrid(1:4, 1:3);
            trainingTruth = truthMatrix(1:4, 1:3);
            model = stage7_fit_bilinear_ridge(cellFeatures, geneFeatures, ...
                ownExpression, trainCell(:), trainGene(:), ...
                trainingTruth(:), 1e-6, ...
                Tolerance=1e-9, MaxIterations=200);
            prediction = stage7_predict_bilinear_ridge(model, ...
                cellFeatures, geneFeatures, ownExpression, 5, 4);
            testCase.verifyTrue(isfinite(prediction));
            testCase.verifySize(prediction, [1, 1]);
        end

        function testTwoStepKernelReturnsFiniteMatrix(testCase)
            cellCoordinates = (-2:2)';
            geneCoordinates = (-1.5:1:1.5)';
            cellKernel = exp(-squareform(pdist(cellCoordinates)) .^ 2);
            geneKernel = exp(-squareform(pdist(geneCoordinates)) .^ 2);
            [trainCell, trainGene] = ndgrid(1:4, 1:3);
            target = sin(trainCell(:)) + cos(trainGene(:));
            model = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
                trainCell(:), trainGene(:), target, 0.1);
            testCase.verifySize(model.predictionMatrix, [5, 4]);
            testCase.verifyTrue(all(isfinite(model.predictionMatrix), "all"));
        end

        function testFoldBaseUsesTrainingLabelsOnly(testCase)
            geneFeatures = [(-2:3)', ((-2:3)'.^2)];
            trainingGeneIndex = [1; 1; 2; 2; 3; 3];
            trainingTarget = [0.8; 1.2; 1.8; 2.2; 2.8; 3.2];
            [baseByGene, info] = stage7_fit_fold_gene_base( ...
                geneFeatures, trainingGeneIndex, trainingTarget, 42);

            testCase.verifyEqual(baseByGene(1:3), [1; 2; 3], ...
                AbsTol=1e-12);
            testCase.verifySize(baseByGene, [6, 1]);
            testCase.verifyTrue(all(isfinite(baseByGene(4:6))));
            testCase.verifyEqual(info.trainingRows, 6);
            testCase.verifyEqual(info.trainingGenes, 3);
        end

        function testPathwayNamesUseSameNormalization(testCase)
            annotations = [ ...
                "Metabolism > Fe-S proteins|Translation > mt-tRNA"; ...
                "Respiration > Complex I" ...
            ];
            pathwayNames = ["Fe-S_proteins"; "mt-tRNA"; "Complex_I"];

            [membership, counts] = build_gene_pathway_membership( ...
                annotations, pathwayNames);

            testCase.verifyEqual(membership, [1, 1, 0; 0, 0, 1]);
            testCase.verifyEqual(counts, [1, 1, 1]);
        end

        function testMissingPathwayAnnotationsStayUnconnected(testCase)
            annotations = ["0"; missing; ""];
            pathwayNames = ["Fe-S_proteins"; "mt-tRNA"];

            [membership, counts] = build_gene_pathway_membership( ...
                annotations, pathwayNames);

            testCase.verifyEqual(membership, zeros(3, 2));
            testCase.verifyEqual(counts, zeros(1, 2));
        end
    end
end
