classdef stage9ModelsTest < matlab.unittest.TestCase
    %stage9ModelsTest 检查虚拟敲除状态和冻结预测接口。

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
        function testBartlettScoresMatchFactoran(testCase)
            latent = randn(120, 2);
            loadingPattern = [0.55, 0.10; 0.45, 0.15; 0.35, 0.20; ...
                0.10, 0.50; 0.15, 0.40; 0.20, 0.35];
            data = latent * loadingPattern' + 0.85 * randn(120, 6);
            pathwayCenter = mean(data, 1);
            pathwayScale = std(data, 0, 1);
            factorInput = (data - pathwayCenter) ./ pathwayScale;
            [loadings, specificVariance, ~, ~, expected] = factoran( ...
                factorInput, 2, Rotate="varimax", Scores="Bartlett");
            encoder = struct( ...
                pathwayCenter=pathwayCenter, pathwayScale=pathwayScale, ...
                factorCenter=mean(factorInput, 1), ...
                factorScale=std(factorInput, 0, 1), ...
                specificVariance=specificVariance, loadings=loadings);

            actual = stage9_apply_bartlett_scores(data, encoder);

            testCase.verifyEqual(actual, expected, AbsTol=1e-10);
        end

        function testCounterfactualUsesPathwayMeanDefinition(testCase)
            encoder = makeIdentityEncoder(3);
            expression = [1.5, -0.2, 0.7];
            pathways = [0.4, -0.3, 0.8];
            membership = logical([1, 0, 1]);
            counts = [3, 5, 4];

            state = stage9_build_counterfactual_state(expression, ...
                pathways, 1, -1.5, membership, counts, 0.5, encoder);

            testCase.verifyEqual(state.expression(1), 0, AbsTol=1e-12);
            testCase.verifyEqual(state.pathwayChange, ...
                [-0.5, 0, -0.375], AbsTol=1e-12);
            testCase.verifyEqual(state.pathway(2), pathways(2), ...
                AbsTol=1e-12);
        end

        function testZeroIntensityKeepsStateUnchanged(testCase)
            encoder = makeIdentityEncoder(3);
            expression = [0.2, 0.4];
            pathways = [-0.1, 0.3, 0.8];
            state = stage9_build_counterfactual_state(expression, ...
                pathways, 2, -2, logical([1, 1, 0]), [2, 4, 3], ...
                0, encoder);

            testCase.verifyEqual(state.expression, expression, ...
                AbsTol=1e-12);
            testCase.verifyEqual(state.pathway, pathways, AbsTol=1e-12);
            testCase.verifyEqual(state.moduleChange, zeros(1, 2), ...
                AbsTol=1e-12);
        end

        function testPathwayNamesUseSameNormalization(testCase)
            metadata = table("GENE1", ...
                "Metabolism > Fe-S proteins|Translation > mt-tRNA", ...
                'VariableNames', {'gene_symbol', 'pathways'});

            [membership, counts] = stage9_build_pathway_membership( ...
                "GENE1", metadata, ["Fe-S_proteins"; "mt-tRNA"]);

            testCase.verifyEqual(membership, logical([1, 1]));
            testCase.verifyEqual(counts, [1, 1]);
        end

        function testFrozenPredictionReproducesKernelRow(testCase)
            coordinate = (-2:2)';
            cellKernel = exp(-pdist2(coordinate, coordinate) .^ 2 / 2);
            geneCoordinate = (-1.5:1:1.5)';
            geneKernel = exp(-pdist2(geneCoordinate, geneCoordinate) .^ 2);
            [cellIndex, geneIndex] = ndgrid(1:4, 1:3);
            target = sin(cellIndex(:)) + cos(geneIndex(:));
            model = stage7_fit_two_step_kernel(cellKernel, geneKernel, ...
                cellIndex(:), geneIndex(:), target, 0.1);

            actual = stage9_predict_counterfactual_cell(model, ...
                cellKernel(5, :));

            testCase.verifyEqual(actual, model.predictionMatrix(5, :), ...
                AbsTol=1e-12);
        end
    end
end

function encoder = makeIdentityEncoder(pathwayCount)
encoder.pathwayCenter = zeros(1, pathwayCount);
encoder.pathwayScale = ones(1, pathwayCount);
encoder.factorCenter = zeros(1, pathwayCount);
encoder.factorScale = ones(1, pathwayCount);
encoder.specificVariance = ones(pathwayCount, 1);
encoder.loadings = eye(pathwayCount, 2);
encoder.leafColumns = 1:pathwayCount;
end
