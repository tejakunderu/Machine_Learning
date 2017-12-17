clc;
clear all;
load('data.mat');

% Initializing values
numFeatures = size(data, 2) - 1;
numExamples = size(data, 1);
threshold = 0.0001;

% Initializing matrix to store objective function values
% Iterating over different number of clusters
objectiveFunction = [];
for numClusters = 2 : 10
    
    % Initializing old centers to zero
    centers = [];
    for c = 1 : numClusters
        centers = [centers; zeros(1, numFeatures)];
    end

    % New centers are randomly allotted from the data set
    newcenters = [];
    randomNumbers = [];
    for c = 1 : numClusters
        newrandom = randi(numExamples);
        while any(randomNumbers) == newrandom
            newrandom = randi(numExamples);
        end

        randomNumbers = [randomNumbers; newrandom];
        newcenters = [newcenters; data(newrandom, 2:end)];
    end

    % Iterate until new centers converge with old
    numIterations = 0;
    while any(abs(centers - newcenters)) > threshold
        centers = newcenters;
        % Initializing matrix to keep track of individuala membership of
        % the examples in the data set
        membership = [];

        % Classifying examples to find out membership clusters
        for i = 1 : numExamples
            % Initializing a matrix to calculate the distance of the
            % current example to every cluster
            currentDistances = [];
            currentExample = data(i, 2:end);

            % Calculating the distance of the example from each cluster
            for c = 1 : numClusters
                currentDistances = [currentDistances; getDistance(currentExample, centers(c, :))];
            end

            % Finding out the minimum distance and the index of the closest
            % cluster to allot membership
            [minValue, clusterIndex] = min(currentDistances);
            membership = [membership; clusterIndex];
        end

        % Updating center for each cluster
        newcenters = [];
        for c = 1 : numClusters
            clusterMembers = [];

            % Setting member points of the cluster in consideration 
            % into a new matrix
            for i = 1 : numExamples
                if membership(i, 1) == c
                    clusterMembers = [clusterMembers; data(i, 2:end)];
                end
            end

            % Summing up the values of each data point
            clusterCenter = [zeros(1, numFeatures)];
            for m = 1 : size(clusterMembers, 1)
                clusterCenter = clusterCenter + clusterMembers(m, :);
            end

            % Averaging the values to find out the new center
            clusterCenter = clusterCenter / size(clusterMembers, 1);
            newcenters = [newcenters; clusterCenter];
        end

        numIterations = numIterations + 1;
    end

    centers = newcenters;

    % computing objective function by adding distances from every point to
    % its cluster center
    total = 0;
    for i = 1 : numExamples
        total = total + getDistance(data(i, 2:end), centers(membership(i, 1), :));
    end
    
    % Keeping track of the objective function values to plot
    objectiveFunction = [objectiveFunction total];
    
end

% Plotting the values of the objective function against the number of
% clusters using in kmeans
hold on;
title('Objective Function vs Number of Clusters');
xlabel('Num Clusters');
ylabel('Objective Function');
plot(2:10, objectiveFunction);
hold off;
