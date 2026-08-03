classdef stage8ModelsTest < matlab.unittest.TestCase
    %stage8ModelsTest 检查阶段8建核与权重学习的关键约束。

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
        function testKernelParametersUseTrainingObjectsOnly(testCase)
            features = [0, 1; 1, 0; 2, 1; 3, 2; 4, 3];
            trainingRows = [true; true; true; false; false];
            [kernelA, infoA, processedA] = stage8_build_rbf_kernel( ...
                features, trainingRows);

            changed = features;
            changed(~trainingRows, :) = changed(~trainingRows, :) + 1000;
            [kernelB, infoB, processedB] = stage8_build_rbf_kernel( ...
                changed, trainingRows);

            testCase.verifyEqual(infoA.bandwidth, infoB.bandwidth, ...
                AbsTol=1e-12);
            testCase.verifyEqual(processedA(trainingRows, :), ...
                processedB(trainingRows, :), AbsTol=1e-12);
            testCase.verifyEqual(kernelA(trainingRows, trainingRows), ...
                kernelB(trainingRows, trainingRows), AbsTol=1e-12);
        end

        function testWeightsPreferAlignedKernel(testCase)
            coordinate = linspace(-2, 2, 10)';
            goodKernel = exp(-pdist2(coordinate, coordinate) .^ 2);
            shuffled = coordinate([1, 6, 2, 7, 3, 8, 4, 9, 5, 10]);
            weakKernel = exp(-pdist2(shuffled, shuffled) .^ 2);
            result = stage8_learn_kernel_weights( ...
                {goodKernel, weakKernel}, (1:10)', goodKernel, 0.01);

            testCase.verifyGreaterThan(result.weights(1), result.weights(2));
            testCase.verifyGreaterThanOrEqual(result.weights, zeros(2, 1));
            testCase.verifyEqual(sum(result.weights), 1, AbsTol=1e-12);
            testCase.verifyGreaterThan(result.exitflag, 0);
        end

        function testEqualKernelsReceiveEqualWeights(testCase)
            coordinate = (-3:3)';
            kernel = exp(-pdist2(coordinate, coordinate) .^ 2 / 2);
            result = stage8_learn_kernel_weights( ...
                {kernel, kernel}, (1:7)', kernel, 0.05);

            testCase.verifyEqual(result.weights, [0.5; 0.5], ...
                AbsTol=1e-8);
        end
    end
end
