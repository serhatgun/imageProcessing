function [cleaned_img] = filters(im, filter)

filterType = filter.filter_type
filterSize = filter.filter_size
filterForm = filter.params.form
filterDegree = filter.params.degree


if strcmp(filterType, 'mean')
    mask = meanMask(filterSize);
    cleaned_img = meanFilter(im, mask, filterForm, filterDegree);
    
elseif strcmp(filterType, 'max')
    mask = medianMask(filterSize);
    cleaned_img = maxFilter(im,mask);
    
elseif strcmp(filterType, 'min')
    mask = medianMask(filterSize);
    cleaned_img = minFilter(im,mask);
    
elseif strcmp(filterType, 'median')
    mask = medianMask(filterSize);
    cleaned_img = medianFilter(im,mask,filterForm);
    
elseif strcmp(filterType, 'mid')
    mask = medianMask(filterSize);
    cleaned_img = midPointFilter(im,mask);
    
elseif strcmp(filterType, 'laplacian')
    mask = laplacianMask(filterSize)
    cleaned_img = laplacianFilter(im, mask);
    
elseif strcmp(filterType, 'sobel')
    mask = sobelMask(filterSize);
    cleaned_img = sobelFilter(im, mask);
    
elseif strcmp(filterType, 'first')
    cleaned_img = firstOrderFilter(im);
    
elseif strcmp(filterType, 'second')
    cleaned_img = secondOrderFilter(im);
%     
% elseif strcmp(filterType, 'lowpass')
%     cleaned_img = lowpassFilter(im, filterSize, filterForm, filterDegree);
%     
% elseif strcmp(filterType, 'highpass')
%     cleaned_img = highpassFilter(im, filterSize, filterForm, filterDegree);
%     
% elseif strcmp(filterType, 'bandreject')
%     cleaned_img = bandrejectFilter(im, filterSize, filterForm, filterDegree);
end
end

function [new_im] = meanFilter(im, mask, form, degree)
[u,v] = size(mask);
[x,y] = size(im);
new_im = uint8(zeros(x,y));

padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
[x,y] = size(padded_im);

if strcmp(form, 'geometric')
    mask = ones(size(mask,1));
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            val = 1;
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    val = val * double(padded_im(row+s,col+t)*mask(s+round(u/2),t+round(v/2)));
                end
            end
            new_im(row - (u-1)/2,col - (v-1)/2) = uint8(val^(1/(u*v)));
        end
    end
    
elseif strcmp(form, 'harmonic')
    mask = ones(size(mask,1));
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            val = 0;
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    val = val + (1/double(padded_im(row+s,col+t)*mask(s+round(u/2),t+round(v/2))));
                end
            end
            new_im(row - (u-1)/2,col - (v-1)/2) = uint8((u*v)/val);
        end
    end
elseif strcmp(form, 'contra')
    mask = ones(size(mask,1));
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            val = 0;
            val2 = 0;
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    mul = double(padded_im(row+s,col+t)*mask(s+round(u/2),t+round(v/2)));
                    val = val + (mul^(-2+1));
                    val2 = val2 + (mul^(-2));
                end
            end
            new_im(row - (u-1)/2,col - (v-1)/2) = uint8(val/val2);
        end
    end
elseif strcmp(form, 'alpha')
    mask = ones(size(mask,1));
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            val = 0;
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    val = val + double(padded_im(row+s,col+t)*mask(s+round(u/2),t+round(v/2)));
                end
            end
            new_im(row - (u-1)/2,col - (v-1)/2) = uint8(((1/(u*v))-degree)*val);
        end
    end
else
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            val = 0;
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    val = val + double(padded_im(row+s,col+t)*mask(s+round(u/2),t+round(v/2)));
                end
            end
            new_im(row - (u-1)/2,col - (v-1)/2) = uint8(val);
        end
    end
end
end

function [new_im] = maxFilter(im, mask)
[u,v] = size(mask);
[x,y] = size(im);
new_im = uint8(zeros(x,y));

padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
[x,y] = size(padded_im);

for row = (u+1)/2:x-((u-1)/2)
    for col = (v+1)/2:y-((v-1)/2)
        for s = -(u-round(u/2)):u-round(u/2)
            for t = -(v-round(v/2)):v-round(v/2)
                mask(s+round(u/2),t+round(v/2)) = padded_im(row+s, col+t);
            end
        end
        new_im(row - (u-1)/2,col - (v-1)/2) = uint8(max(mask(:)));
        mask = zeros(u,v);
    end
end
end

function [new_im] = minFilter(im, mask)

[u,v] = size(mask);
[x,y] = size(im);
new_im = uint8(zeros(x,y));

padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
[x,y] = size(padded_im);

for row = (u+1)/2:x-((u-1)/2)
    for col = (v+1)/2:y-((v-1)/2)
        for s = -(u-round(u/2)):u-round(u/2)
            for t = -(v-round(v/2)):v-round(v/2)
                mask(s+round(u/2),t+round(v/2)) = padded_im(row+s, col+t);
            end
        end
        new_im(row - (u-1)/2,col - (v-1)/2) = uint8(min(mask(:)));
        mask = zeros(u,v);
    end
end
end

function [new_im] = medianFilter(im, mask, form)
if strcmp(form, 'adaptive')
    [u,v] = size(mask);
    [x,y] = size(im);
    new_im = uint8(zeros(x,y));
    
    padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
    [x,y] = size(padded_im);
    
    smax = 7;

    
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            while size(mask,1) <= smax
                for s = -(u-round(u/2)):u-round(u/2)
                    for t = -(v-round(v/2)):v-round(v/2)
                        mask(s+round(u/2),t+round(v/2)) = padded_im((row+abs((3-u)/2)+s), (col+abs((3-u)/2)+t));
                    end
                end

                zmin = double(min(mask(:)));
                zmax = double(max(mask(:)));
                zmed = double(median(mask(:)));
                zxy = double(padded_im(row,col));

                A1 = zmed - zmin;
                A2 = zmed - zmax;

                if A1>0 && A2<0
                    B1 = zxy - zmin;
                    B2 = zxy - zmax;

                    if B1>0 && B2<0
                        new_im((row+abs((3-u)/2)) - (u-1)/2,((col+abs((3-u))/2)) - (v-1)/2) = zxy;
                        mask = zeros(3);
                        [u v] = size(mask);
                        padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
                        [x,y] = size(padded_im);
                        break;
                    else
                        new_im((row+abs((3-u)/2)) - (u-1)/2,((col+abs((3-u))/2)) - (v-1)/2)  = zmed;
                        mask = zeros(3);
                        [u v] = size(mask);
                        padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
                        [x,y] = size(padded_im);
                        break;
                    end
                else
                    mask = ones(u+2,u+2);
                    [u v] = size(mask);
                    padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
                    [x,y] = size(padded_im);
                    if(size(mask,1) >= smax)
                        new_im((row+abs((3-u)/2)) - (u-1)/2,((col+abs((3-u))/2)) - (v-1)/2)  = zmed;
                        mask = zeros(3);
                        [u v] = size(mask);
                        padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
                        [x,y] = size(padded_im);
                        break;
                    end
                end
            end
        end
    end
else
    [u,v] = size(mask);
    [x,y] = size(im);
    new_im = uint8(zeros(x,y));
    
    padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
    [x,y] = size(padded_im);
    
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    mask(s+round(u/2),t+round(v/2)) = padded_im(row+s, col+t);
                end
            end
            new_im(row - (u-1)/2,col - (v-1)/2) = uint8(median(mask(:)));
            mask = zeros(u,v);
        end
    end
end
end

function [new_im] = midPointFilter(im, mask)

[u,v] = size(mask);
[x,y] = size(im);
new_im = uint8(zeros(x,y));

padded_im = padarray(im, [(u-1)/2 (v-1)/2], 'replicate', 'both');
[x,y] = size(padded_im);

for row = (u+1)/2:x-((u-1)/2)
    for col = (v+1)/2:y-((v-1)/2)
        for s = -(u-round(u/2)):u-round(u/2)
            for t = -(v-round(v/2)):v-round(v/2)
                mask(s+round(u/2),t+round(v/2)) = padded_im(row+s, col+t);
            end
        end
        new_im(row - (u-1)/2,col - (v-1)/2) = uint8((1/2)*(min(mask(:)) + max(mask(:))));
        mask = zeros(u,v);
    end
end
end

function [new_im] = laplacianFilter(im, mask)
[u,v] = size(mask);
[x,y] = size(im);

new_im = uint8(zeros(x,y));
filtered_im = uint8(zeros(x,y));
padded_im = padarray(double(im), [(u-1)/2 (v-1)/2], 'replicate', 'both');
padded_im = im2double(padded_im);

[x,y] = size(padded_im);

for row = (u+1)/2:x-((u-1)/2)
    for col = (v+1)/2:y-((v-1)/2)
        sum = double(0);
        for s = -(u-round(u/2)):u-round(u/2)
            for t = -(v-round(v/2)):v-round(v/2)
                sum = sum + double(padded_im(row+s,col+t)*mask(s+round(u/2),t+round(v/2)));
            end
        end
         filtered_im(row - (u-1)/2,col - (v-1)/2) = uint8(sum);
    end
end

imshow(filtered_im)
new_im = im - uint8(filtered_im);

end

function [new_im] = sobelFilter(im, mask)
[u,v] = size(mask);
[x,y] = size(im);
new_im = uint8(zeros(x,y));
filtered_im = uint8(zeros(x,y));

masks(:,:,1) = mask;
masks(:,:,2) = transpose(mask);

padded_im = padarray(double(im), [(u-1)/2 (v-1)/2], 'replicate', 'both');
[x,y] = size(padded_im);

for i = 1:2
    for row = (u+1)/2:x-((u-1)/2)
        for col = (v+1)/2:y-((v-1)/2)
            val = 0;
            for s = -(u-round(u/2)):u-round(u/2)
                for t = -(v-round(v/2)):v-round(v/2)
                    val = val + padded_im(row+s,col+t)*masks(s+round(u/2),t+round(v/2),i);
                end
            end
            filtered_im(row - (u-1)/2,col - (v-1)/2) = uint8(val);
        end
    end
    new_im = new_im + filtered_im;
end
end

function [new_im] = firstOrderFilter(im)
[x,y] = size(im);
new_im = uint8(zeros(x,y));
filtered_im = uint8(zeros(x,y));

for row = 1:x-1
       for col = 1:y-1
           filtered_im(row,col) = im(row+1,col+1) - im(row,col);
       end
end

imshow(filtered_im)
new_im = im - filtered_im;
end

function [new_im] = secondOrderFilter(im)
[x,y] = size(im);
new_im = uint8(zeros(x,y));
filtered_im = uint8(zeros(x,y));

for row = 2:x-1
       for col = 2:y-1
           filtered_im(row,col) = im(row+1,col+1) + im(row-1,col-1) - 2*im(row,col);
       end
end

imshow(filtered_im)
new_im = im - filtered_im;   
end

function [new_im] = lowpassFilter(im, radius, form, degree)
[M N] = size(im);
freq_im = fft2(double(im));

u = 0:(M-1);
v = 0:(N-1);

if strcmp(form, 'butterworth')
    [V,U] = meshgrid(v,u);
    D = U+V;
    H = double(1./(1+(D./radius).^(2*degree)));
    imshow(H);
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
    
elseif strcmp(form, 'gaussian')
    [V,U] = meshgrid(v,u);
    D = U+V;
    H = exp((-D.^2)./(2*radius.^2));
    imshow(H);
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
    
    
else
    indx = find(u>M/2);
    u(indx) = u(indx) - M;
    
    indy = find(v>N/2);
    v(indy) = v(indy) - N;
    
    [V,U] = meshgrid(v,u);
    D = sqrt(U.^2 + V.^2);
    H = double(D <= radius);
    imshow(H, [])
    
    IM = H.*freq_im;
    
    new_im = uint8(real(ifft2(double(IM))));
end
end

function [new_im] = highpassFilter(im, radius, form, degree)
[M N] = size(im);
freq_im = fft2(double(im));

u = 0:(M-1);
v = 0:(N-1);

if strcmp(form, 'butterworth')
    [V,U] = meshgrid(v,u);
    D = U+V;
    H = double(1./(1+(radius./D).^(2*degree)));
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
    
elseif strcmp(form, 'gaussian')
    [V,U] = meshgrid(v,u);
    D = U+V;
    H = 1-exp((-D.^2)./(2*radius.^2));
    imshow(H);
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
    
    
else
    indx = find(u>M/2);
    u(indx) = u(indx) - M;
    
    indy = find(v>N/2);
    v(indy) = v(indy) - N;
    
    [V,U] = meshgrid(v,u);
    D = sqrt(U.^2 + V.^2);
    H = double(D >= radius);
    imshow(abs(H), [])
    IM = H.*freq_im;
    
    new_im = uint8(real(ifft2(double(IM))));
end
end

function [new_im] = bandrejectFilter(im, radius, form, degree)
[M N] = size(im);
freq_im = fft2(double(im));

u = 0:(M-1);
v = 0:(N-1);


if strcmp(form, 'butterworth')
    indx = find(u>M/2);
    u(indx) = u(indx) - M;
    
    indy = find(v>N/2);
    v(indy) = v(indy) - N;
    
    [V,U] = meshgrid(v,u);
    D = sqrt(U.^2 + V.^2);
    
    H = double(zeros(M,N));    
    for i=1:size(D,1)
        for j=1:size(D,2)
            if (radius-20 <= D(i,j)) && (D(i,j) <= radius+20)
                H(i,j) = double(1./(1+(radius./D(i,j)).^(2*degree)));
            else
                H(i,j) = double(0);
            end
        end
    end
    imshow(abs(H), [])
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
    
    
elseif strcmp(form, 'gaussian')
    [V,U] = meshgrid(v,u);
    D = U+V;
    H = 1-exp((-D.^2)./(2*radius.^2));
    imshow(H);
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
    
    
else
    indx = find(u>M/2);
    u(indx) = u(indx) - M;
    
    indy = find(v>N/2);
    v(indy) = v(indy) - N;
    
    [V,U] = meshgrid(v,u);
    D = sqrt(U.^2 + V.^2);
    
    H = double(zeros(M,N));
    
    for i=1:size(D,1)
        for j=1:size(D,2)
            if (abs(radius-10 <= D(i,j))) && (D(i,j) <= radius+10)
                H(i,j) = double(1);
            else
                H(i,j) = double(0);
            end
        end
    end
    imshow(abs(H), [])
    
    IM = H.*freq_im;
    new_im = uint8(real(ifft2(double(IM))));
end
end

function mask = meanMask(n)
mask = ones(n);
mask = mask/n^2;
end

function mask = medianMask(n)
mask = zeros(n);
end

function mask = laplacianMask(n)
mask = ones(n);
mask(ceil((n^2)/2)) = (1 - n^2);
end

function mask = sobelMask(n)
if n == 3
    mask = [-1 -2 -1; 0 0 0; 1 2 1;];
elseif n == 5
    mask = [2 2 4 2 2; 1 1 2 1 1; 0 0 0 0 0; -1 -1 -2 -1 -1; -2 -2 -4 -2 -2];  
end
end

