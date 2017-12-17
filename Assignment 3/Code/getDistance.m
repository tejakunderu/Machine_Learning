function distance = getDistance(point1, point2)
    % calculates the distance between two points in a multi-dimensional
    % space
    distance = 0;
    
    for i = 1 : size(point1, 2)
        distance = distance + power((point1(1, i) - point2(1, i)), 2);
    end

end

