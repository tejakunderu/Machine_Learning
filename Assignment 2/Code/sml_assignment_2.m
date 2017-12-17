clc;
clear all;
load('data.mat');

% Initializing data variables
numParams = size(data, 2) - 1;
numExamples = size(data, 1);

numTrain = floor((2 * numExamples) / 3);
numTest = ceil(numExamples - numTrain);

fractions = [.01 .02 .03 .125 .625 1];
numIterations = 5;
learningRate = 0.01;

% Initializing matrices to store accuracy and data size
accuracyNB = [];
accuracyLR = [];
dataSize = [];

% Running alrorithms for different fractions of the data set
for f = 1 : size(fractions, 2)
    currentFraction = fractions(1, f);
    fractionalDataSize = floor(numTrain * currentFraction);
    
    sumAccuracyNB = 0;
    sumAccuracyLR = 0;
    
    % Running the algorithms multiple times to calculate average accuracy
    for itr = 1 : numIterations
        newData = [];
        
        % Initializing new random data during each iteration
        startIndex = randi(numTrain + 1);
        if (startIndex + fractionalDataSize) > numTrain
            endIndex = startIndex + fractionalDataSize - numTrain - 1;
            for index = startIndex : numTrain
                newData = [newData; data(index, 1:size(data, 2))];
            end
            for index = 1 : endIndex
                newData = [newData; data(index, 1:size(data, 2))];
            end
        else
            endIndex = startIndex + fractionalDataSize - 1;
            for index = startIndex : endIndex
                newData = [newData; data(index, 1:size(data, 2))];
            end
        end
        
        % ***************** Naive Bayes ***************** %
        
        % Calculating y probablities
        numY0 = 0;
        numY1 = 0;
        for i = 1 : size(newData, 1)
            if newData(i, 10) == -1
                numY0 = numY0 + 1;
            else 
                numY1 = numY1 + 1;
            end
        end

        pY0 = numY0 / fractionalDataSize;
        pY1 = numY1 / fractionalDataSize;

        % Calculating x|y probabilities
        pX0 = [];
        pX1 = [];

        for feature = 1 : numParams
            pXtemp = [];

            for i = 1 : size(newData, 1)
                value = newData(i, feature);

                count = 0;
                for j = 1 : size(newData, 1)
                    if (newData(j, feature) == value) && (newData(j, 10) == -1)
                        count = count + 1;
                    end
                end

                pXtemp = [pXtemp; value ((count + 1) / (numY0 + 10))];
            end

            pXtemp = unique(pXtemp, 'rows');

            for val = 1 : 10
                if ~ismember(val, pXtemp(1:size(pXtemp, 1), 1))
                    pXtemp = [pXtemp; val (1 / (numY0 + 10))];
                end
            end

            pXtemp = sortrows(pXtemp);

            pX0 = [pX0 pXtemp(1:10, 2)];
        end

        for feature = 1 : numParams
            pXtemp = [];

            for i = 1 : size(newData, 1)
                value = newData(i, feature);

                count = 0;
                for j = 1 : size(newData, 1)
                    if (newData(j, feature) == value) && (newData(j, 10) == 1)
                        count = count + 1;
                    end
                end

                pXtemp = [pXtemp; value ((count + 1) / (numY1 + 10))];
            end

            pXtemp = unique(pXtemp, 'rows');

            for val = 1 : 10
                if ~ismember(val, pXtemp(1:size(pXtemp, 1), 1))
                    pXtemp = [pXtemp; val (1 / (numY1 + 10))];
                end
            end

            pXtemp = sortrows(pXtemp);

            pX1 = [pX1 pXtemp(1:10, 2)];
        end

        % Testing the accuracy of the algorithm
        correctPredictionsNB = 0;
        for i = numTrain + 1 : numTrain + numTest
            y = data(i, 10);
            
            term0 = pY0;
            term1 = pY1;
            denom = 0;
            
            for feature = 1 : numParams
                val = data(i, feature);
                term0 = term0 * pX0(val, feature);
                term1 = term1 * pX1(val, feature);
            end
            
            denom = term0 + term1;
            p = term1 / denom;
            
            if (p < 0.5 && y == -1)
                correctPredictionsNB = correctPredictionsNB + 1;
            elseif (p >= 0.5 && y == 1)
                correctPredictionsNB = correctPredictionsNB + 1;
            end
        end
        
         % ***************** Logistic Regression ***************** %
        
        % Initializing weight matrices
        params = zeros(1, numParams);
        newparams = zeros(1, numParams);

        % Updating the weights 
        for i = 1 : size(newData, 1)
            for j = 1 : numParams
                y = (newData(i, 10) + 1) / 2;
                h = 1 / (1 + exp(-(params * transpose(newData(i, 1:numParams)))));
                x = newData(i, j);
                newparams(1, j) = params(1, j) + (learningRate * (y - h) * x);
            end

            params = newparams;
        end

        % Calculating the accuracy of the algorithm
        correctPredictionsLR = 0;
        for i = numTrain + 1 : numTrain + numTest
            y = data(i, 10);
            h = 1 / (1 + exp(-(params * transpose(data(i, 1:numParams)))));
            if (h < 0.5 && y == -1)
                correctPredictionsLR = correctPredictionsLR + 1;
            elseif (h >= 0.5 && y == 1)
                correctPredictionsLR = correctPredictionsLR + 1;
            end
        end
        
        currentAccuracyNB = correctPredictionsNB / numTest;
        sumAccuracyNB = sumAccuracyNB + currentAccuracyNB; 
        
        currentAccuracyLR = correctPredictionsLR / numTest;
        sumAccuracyLR = sumAccuracyLR + currentAccuracyLR;       
    end
    
    % Calculaing average accuracies
    accuracyNB = [accuracyNB (sumAccuracyNB / numIterations)];
    accuracyLR = [accuracyLR (sumAccuracyLR / numIterations)];
    dataSize = [dataSize fractionalDataSize];
end

% Plotting accuracy vs data size
hold on;
title('Learning Curves');
xlabel('Size of training data');
ylabel('Classification Accuracy');
plot(dataSize, accuracyNB, '--');
plot(dataSize, accuracyLR);
hold off;

legend('Naive Bayes', 'Logistic Regression', 'Location', 'east');
legend('boxoff');
