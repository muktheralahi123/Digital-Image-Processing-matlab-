% Load image package for Octave
pkg load image

% 1. imread: Read image
img = imread('f4.png');

% 2. imshow: Display original image
figure; imshow(img); title('Original Image');

% 3. imresize: Resize image to 50%
img_resized = imresize(img, 0.5);
figure; imshow(img_resized); title('Resized Image (50%)');

% 4. imwrite: Save resized image
imwrite(img_resized, 'resized_f4.png');
disp('Resized image saved as resized_f4.png');

% 5. imcrop: Crop a 100x100 area at (50,50)
img_cropped = imcrop(img_resized, [50 50 100 100]);
figure; imshow(img_cropped); title('Cropped Image');

% 6. imwrite: Save cropped image
imwrite(img_cropped, 'cropped_f4.png');
disp('Cropped image saved as cropped_f4.png');

% 7. imfinfo: Get info about saved image
if exist('cropped_f4.png', 'file')
    info = imfinfo('cropped_f4.png');
    disp('Image info of cropped_f4.png:');
    fprintf('Filename: %s\n', info.Filename);
    fprintf('FileSize: %d bytes\n', info.FileSize);
    fprintf('Width: %d pixels\n', info.Width);
    fprintf('Height: %d pixels\n', info.Height);
    fprintf('BitDepth: %d\n', info.BitDepth);
    fprintf('ColorType: %s\n', info.ColorType);
else
    disp('cropped_f4.png file does not exist!');
end

% 8. Convert color image to grayscale ---
% im2gray is a more general MATLAB function; Octave's image package typically uses rgb2gray.
gray_img = []; % Initialize
if (ndims(img) == 3) % Check if it's a color image (3 dimensions: HxWxChannels)
    gray_img = rgb2gray(img);
    figure;
    subplot(1,2,2); imshow(gray_img); title("Grayscale");
    title("Color to Grayscale Conversion (rgb2gray)");
    disp("Image converted to grayscale and displayed (rgb2gray).");
else
    gray_img = img; % Already grayscale or binary
    disp("7. rgb2gray - Image is already grayscale or binary. Using as is.");
    figure; imshow(gray_img); title("Original Grayscale/Binary");
end


% 9. imhist - Calculate pixel intensity distribution
% (Using gray_img from step 7)
figure;
imhist(gray_img);
title("imhist - Histogram of Grayscale Image");
disp("imhist - Grayscale image histogram displayed.");

% 10. im2frame: Convert image to frame (useful for videos)
frame = im2frame(img_cropped);
figure; imshow(frame.cdata);
title('Image as Frame');

% 11. edge - Detect boundaries/contours in images
% (Using gray_img from step 7)
edges_sobel = edge(gray_img, "sobel");
edges_canny = edge(gray_img, "canny");

figure;
subplot(1,3,1); imshow(gray_img); title("Grayscale");
subplot(1,3,2); imshow(edges_sobel); title("Sobel Edges");
subplot(1,3,3); imshow(edges_canny); title("Canny Edges");
title("edge - Edge Detection");
disp("edge - Edges detected and displayed.");

% 12. hough - Identifies geometric shapes (lines)
% (Using edges_canny from step 10, which should be a binary image)
[H, T, R] = hough(edges_canny); % H: Hough transform, T: theta, R: rho

figure;
imshow(H, [], "XData", T, "YData", R, "InitialMagnification", "fit");
xlabel('\theta (degrees)'); ylabel('\rho');
axis on; axis normal; hold on;
colormap(gca, hot);
title("hough - Hough Transform Accumulator");

% Find peaks in Hough transform
P = houghpeaks(H, 5, "threshold", ceil(0.3*max(H(:))));
plot(T(P(:,2)), R(P(:,1)), "s", "color", "blue");
hold off;
disp("hough - Hough transform calculated and peaks identified.");
% To draw lines: use houghlines(edges_canny, T, R, P, ...)

% 13. impixel: Select pixels interactively and display their values
try
    figure; imshow(img_cropped);
    title('Select pixels (click on image, then close window)');

    [x, y, colors] = impixel(img_cropped);  % Try selecting pixels

    if ~isempty(x)
        fprintf('Selected pixels:\n');
        disp(table(x, y, colors(:,1), colors(:,2), colors(:,3), ...
            'VariableNames', {'X','Y','Red','Green','Blue'}));
    else
        disp('No pixels selected.');
    end
catch
    disp('impixel is not fully supported in Octave. Skipping pixel selection.');
end

% 14. DCT using custom dct2_oct if needed
if exist('dct2', 'file') == 2
    dct_img = dct2(im2double(gray_manual));
elseif exist('dct', 'file') == 2
    dct_img = dct2_oct(im2double(gray_manual));  % Use custom
else
    disp('No DCT function found.');
end

if exist('dct_img', 'var')
    dct_magnitude = log(abs(dct_img) + 1);
    figure; imshow(dct_magnitude, []);
    title('2D DCT (log magnitude)');
end

% 15. IDCT using custom idct2_oct if needed
if exist('dct_img', 'var')
    if exist('idct2', 'file') == 2
        idct_img = idct2(dct_img);
    elseif exist('idct', 'file') == 2
        idct_img = idct2_oct(dct_img);  % Use custom
    else
        disp('No IDCT function found.');
    end
end

if exist('idct_img', 'var')
    figure; imshow(idct_img, []);
    title('Reconstructed Image from IDCT');
end

% 16. imfilter: Filter grayscale image using Sobel filter for edges
h_avg = fspecial("average", [5 5]); % Averaging filter kernel
blurred_img_filter = imfilter(gray_img, h_avg, "replicate"); % "replicate" handles border

h_laplacian = fspecial("laplacian", 0.2);
sharpened_img_filter = gray_img - imfilter(gray_img, h_laplacian, "replicate");

figure;
subplot(1,3,1); imshow(gray_img); title("Original Grayscale");
subplot(1,3,2); imshow(blurred_img_filter); title("Blurred (imfilter)");
subplot(1,3,3); imshow(sharpened_img_filter); title("Sharpened (imfilter)");
title("imfilter - Linear Spatial Filtering");
disp("imfilter - Averaging and sharpening filters applied.");

% 17. imgaussfilt - Gaussian smoothing filter ---
sigma_gauss = 2; % Standard deviation of Gaussian
gaussian_blurred_img = imgaussfilt(gray_img, sigma_gauss);

figure;
subplot(1,2,1); imshow(gray_img); title("Original Grayscale");
subplot(1,2,2); imshow(gaussian_blurred_img); title(sprintf("Gaussian Blurred (sigma=%.1f)", sigma_gauss));
title("imgaussfilt - Gaussian Smoothing");
disp("imgaussfilt - Gaussian filter applied.");

% 18. gabor - Texture analysis filter (kernel creation)
lambda  = 10; theta_rad = pi/4; psi = 0; gamma_aspect = 0.5; sigma_env = 5; sz = 21;
gb_kernel_real = zeros(sz, sz);
for x = -floor(sz/2):floor(sz/2)
    for y = -floor(sz/2):floor(sz/2)
        x_theta = x * cos(theta_rad) + y * sin(theta_rad);
        y_theta = -x * sin(theta_rad) + y * cos(theta_rad);
        idx_x = x + floor(sz/2) + 1; idx_y = y + floor(sz/2) + 1;
        gb_kernel_real(idx_y, idx_x) = exp(-(x_theta^2 + gamma_aspect^2 * y_theta^2) / (2 * sigma_env^2)) * cos(2 * pi * x_theta / lambda + psi);
    end
end
figure; imshow(gb_kernel_real, []); title("Gabor Filter Kernel (Real Part - Simplified)");
disp("gabor - Simplified Gabor kernel created and displayed.");
disp("   For applying Gabor filters, see 'imgaborfilt' or 'gaborfilter'.");

% 19. Applying Gabor Filter (using manual kernel and imfilter if gaborfilter is missing)
disp("--- 20. Applying Gabor Filter ---");
if (exist('gaborfilter', 'file'))
    disp("Using built-in gaborfilter.");
    g_sigma = 4;
    g_freq = 1/8;
    g_theta_deg = 45;
    [gabor_real_comp_builtin, gabor_imag_comp_builtin] = gaborfilter(double(gray_img), g_sigma, g_freq, g_theta_deg);
    gabor_mag_builtin = sqrt(gabor_real_comp_builtin.^2 + gabor_imag_comp_builtin.^2);

    figure;
    subplot(1,3,1); imshow(gray_img); title("Original Grayscale");
    subplot(1,3,2); imshow(gabor_real_comp_builtin, []); title(sprintf("Gabor Real (builtin, theta=%d)", g_theta_deg));
    subplot(1,3,3); imshow(gabor_mag_builtin, []); title(sprintf("Gabor Magnitude (builtin, theta=%d)", g_theta_deg));
    suptitle("20. Gabor Filter (via Octave's gaborfilter)");
    disp("Built-in gaborfilter applied.");

elseif (exist('gb_kernel_real', 'var')) % Check if our manual kernel exists
    disp("Built-in gaborfilter not found. Applying manually created Gabor kernel using imfilter.");
    % Ensure gray_img is double for filtering
    gray_img_double = double(gray_img);

    % Apply the real part of the Gabor kernel
    % 'replicate' handles border issues, 'conv' specifies convolution
    gabor_filtered_real_manual = imfilter(gray_img_double, gb_kernel_real, 'conv', 'replicate');

    figure;
    subplot(1,2,1); imshow(gray_img); title("Original Grayscale");
    subplot(1,2,2); imshow(gabor_filtered_real_manual, []); title("Gabor Filtered (Manual Kernel - Real Part)");
    subtitle("20. Gabor Filter (Manual Kernel Application)");
    disp("Manually created Gabor kernel (real part) applied using imfilter.");
else
    disp("gaborfilter not found, and 'gb_kernel_real' (manual kernel) not found.");
    disp("   Skipping Gabor filter application for step 20.");
endif
disp(" ");

% 20. montage - Combines multiple images in a grid
if ndims(gray_img) == 2, gray_img_rgb = cat(3, gray_img, gray_img, gray_img); else gray_img_rgb = gray_img; end
if ndims(edges_canny) == 2, edges_canny_rgb = cat(3, uint8(edges_canny)*255, uint8(edges_canny)*255, uint8(edges_canny)*255); else edges_canny_rgb = edges_canny; end

% Resize for consistent montage if sizes vary greatly
img_s = imresize(img, [128 128]);
gray_s = imresize(gray_img_rgb, [128 128]);
edges_s = imresize(edges_canny_rgb, [128 128]);

montage_array = cat(4, img_s, gray_s, edges_s); % Create 4D array HxWxDxN
figure;
montage(montage_array);
title("21. montage - Montage of Images");
disp("21. montage - Multiple images displayed as a montage.");


% 21. imsave
disp("imsave - 'imsave' is not native to Octave/MATLAB.");
disp("   The equivalent function is 'imwrite'. See example 5.");


% 22. imrotate - Rotates image by specified angle
angle_rot = 30; % degrees
rotated_img = imrotate(img, angle_rot, "bicubic", "crop"); % "crop" to keep original size
% Use "loose" to make output image larger to contain whole rotated original

figure;
subplot(1,2,1); imshow(img); title("Original for Rotation");
subplot(1,2,2); imshow(rotated_img); title(sprintf("Rotated %d deg (cropped)", angle_rot));
title("imrotate - Image Rotation");
disp("imrotate - Image rotated and displayed.");

% 23. imerase - Removes/masks selected image regions
disp("24. imgerase - 'imgerase' is not a standard Octave/MATLAB function.");
disp("   Region erasure is done by direct pixel manipulation.");
img_to_erase = img; % Make a copy

% Define region to erase (e.g., a rectangle [col_start row_start width height])
erase_rect_roi = [size(img,2)/4, size(img,1)/4, size(img,2)/2, size(img,1)/2]; % Approx center
r_start = round(erase_rect_roi(2));
r_end = round(erase_rect_roi(2) + erase_rect_roi(4) - 1);
c_start = round(erase_rect_roi(1));
c_end = round(erase_rect_roi(1) + erase_rect_roi(3) - 1);

% Ensure coordinates are within image bounds
[rows, cols, channels] = size(img_to_erase);
r_start = max(1, r_start); r_end = min(rows, r_end);
c_start = max(1, c_start); c_end = min(cols, c_end);

% Erase by setting to black (0)
if channels == 3 % Color image
    img_to_erase(r_start:r_end, c_start:c_end, :) = 0;
else % Grayscale image
    img_to_erase(r_start:r_end, c_start:c_end) = 0;
end

figure;
subplot(1,2,1); imshow(img); title("Original for Erasing");
subplot(1,2,2); imshow(img_to_erase); title("With Erased Region");
title("'imerase' (manual) - Region Erased");
disp("Region erased by setting pixels to 0 and displayed.");

disp("--- All examples completed ---");
% To close all figure windows: close all;


