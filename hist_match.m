function [new_im, new_hist] = hist_match(im, designed_hist)

[rows, cols] = size(im);

index = 1:256;
index = index / 255;

pixelNumber = rows*cols;
finalResult = uint8(zeros(rows,cols));
frequency = zeros(256,1);
pdf = zeros(256,1);
cdf = zeros(256,1);
cummlative = zeros(256,1);
out = zeros(256,1);

for im_rows = 1:rows
    for im_cols = 1:cols
        gray_level = im(im_rows, im_cols);
        frequency(gray_level+1) = frequency(gray_level+1) + 1;
        pdf(gray_level+1) = frequency(gray_level+1)/pixelNumber;
    end
end

% finding cdf
sum = 0;
L = 255;

for i = 1:size(pdf)
    sum = sum + frequency(i);
    cummlative(i) = sum;
    cdf(i) = cummlative(i)/pixelNumber;
    out(i) = round(cdf(i)*L);
end

figure('NumberTitle', 'off', 'Name', 'Original Histogram')
bar(index, pdf)

figure('NumberTitle', 'off', 'Name', 'Designed Histogram')
bar(index, designed_hist)

%Create map & cdf of desired histogram
M = zeros(256,1,'uint8');
cdf_designed = cumsum(designed_hist);

%Obtain G(z)
for i = 1:size(cdf)
   [~,s] = min(abs(cdf(i) - cdf_designed));
   M(i) = s - 1;
end

%Apply G(z)
finalResult = M(double(im)+1);
frequency = zeros(256,1);
pdf = zeros(256,1);

%Find new histogram
for im_rows = 1:rows
    for im_cols = 1:cols
        gray_level = finalResult(im_rows, im_cols);
        frequency(gray_level+1) = frequency(gray_level+1) + 1;
        pdf(gray_level+1) = frequency(gray_level+1)/pixelNumber;
    end
end

new_hist = pdf;
new_im = finalResult;

figure('NumberTitle', 'off', 'Name', 'New Histogram')
bar(index, pdf)

end
