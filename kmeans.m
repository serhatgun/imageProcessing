function [label_vector,cluster_centers] = my_kmeans(Data,K)

max_iterations = 10000;
centroids = initCentroids(Data, K);

for i = 1:max_iterations
    indices = getClosestCentroids(Data, centroids);
    centroids = computeCentroids(Data, indices, K);
end

label_vector = indices;
cluster_centers = centroids;
end

function centroids = initCentroids(X, K)
centroids = zeros(K,size(X,2));
r = min(size(X,1),size(X,2));
randindex = randperm(r);
centroids = X(randindex(1:K), :);
end

function indices = getClosestCentroids(X, centroids)
K = size(centroids, 1);
indices = zeros(size(X,1), 1);
m = size(X,1);

for i=1:m
    k = 1;
    min_dist = sum((X(i,:) - centroids(1,:)) .^ 2);
    for j=2:K
        dist = sum((X(i,:) - centroids(j,:)) .^ 2);
        if(dist < min_dist)
            min_dist = dist;
            k = j;
        end
    end
    indices(i) = k;
end
end


function centroids = computeCentroids(X, idx, K)

[m n] = size(X);
centroids = zeros(K, n);

for i=1:K
    xi = X(idx==i,:);
    ck = size(xi,1);
    centroids(i, :) = (1/ck) * [sum(xi)];
end
end

