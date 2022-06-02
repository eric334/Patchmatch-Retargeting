clear;
clc;
close all;

global patch_size;
patch_size = 11;

%% PART 1 - SINGLE IMAGE HOLE FILLING

% NOTE - If you are using the mex files on GS during the development of your
% code, make sure your refer to the README

%% STEP 1 - Read in the image with the hole region 
% Read in the image, convert to RGBA with holes denoted by 0 alpha.
% Identify the region and size of the hole.

[source, ~, alpha] = imread("source1_hole.png");

source = rgb2lab(source); % Convert the source and target Images
source = double(source)/255;

%source = im2double(source);
alpha = abs(im2double(alpha));
source_size = size(source,1:2);

alpha_points_i = [source_size(1), 0];
alpha_points_j = [source_size(2), 0];
for i=1:source_size(1)
    for j=1:source_size(2)
        if alpha(i,j) == 0  
            if i < alpha_points_i(1)
                alpha_points_i(1) = i;
            end
            if i > alpha_points_i(2)
                alpha_points_i(1) = i;
            end
            if j < alpha_points_j(1)
                alpha_points_j(1) = j;
            end
            if j > alpha_points_j(2)
                alpha_points_j(2) = j;
            end
        end
    end
end

hole_size_x = alpha_points_i(2) - alpha_points_i(1);
hole_size_y = alpha_points_j(2) - alpha_points_j(1); 

target_scale = patch_size / hole_size_y;
if (hole_size_x > hole_size_y)
    target_scale = patch_size / hole_size_x;
end

%% STEP 2 - Downsample image
% Iteratively downsample image till the dimension of the path is around the dimension
% of the hole. Store images at these multiple scales

% Parameters
numScales = 20;                 % number of scales

source_images = cell(numScales);
target_images = cell(numScales);
source_alphas = cell(numScales);

figure

for i=1:numScales
    scale = 1 - ((i-1) * (1 - target_scale) / (numScales-1));

    source_images{i} = imresize(source,scale, "bicubic");
    source_alphas{i} = abs(imresize(alpha, scale, "bicubic"));

    imshow(lab2rgb(source_images{i} * 255))
end


%% STEP 3 - Perform Hole filling
% Perform hole filling at each scale, starting at the coarsest. Use
% repeated search and vote steps (refer to HW8 and the final project 
% descriptions) at each scale till values within the hole have converged.
% Pixels in the hole region are the targets, patches outside the hole are
% the source.
% Upsample the resulting image, and blend it with the original downsampled
% image at the same scale, to refine the values outside the hole.

for i=1:numScales
    source_image = source_images{numScales - i + 1};

    source_size = size(source_image);

    source_alpha = source_alphas{numScales - i + 1};
    target_image = [];

    if (i == 1)
        target_image = interpolate_hole (source_image, source_alpha);
        for j = 1:2
            target_image = voteNNF(patchMatchNNF_hole(target_image, source_image, source_alpha), source_image, source_alpha);
            imshow(lab2rgb(target_image * 255))
        end
    else
        target_image = target_images{numScales - i + 2};
        target_image = imresize(target_image, [source_size(1), source_size(2)], "bicubic");
        target_image = blend_images(target_image, source_image, source_alpha);
    end

    padded_bounding_box = get_padded_bounding_box(source_alpha, (patch_size - 1) / 2);
    % crop target image for speed
    cropped_target = target_image(padded_bounding_box(1):padded_bounding_box(3), padded_bounding_box(2):padded_bounding_box(4),:);

    cropped_target = voteNNF(patchMatchNNF_hole(cropped_target, source_image, source_alpha), source_image, source_alpha);

    % put target back in
    target_image(padded_bounding_box(1):padded_bounding_box(3), padded_bounding_box(2):padded_bounding_box(4),:) = cropped_target;

%     target_image = voteNNF(patchMatchNNF_hole(target_image, source_image, source_alpha), source_image, source_alpha);

    imshow(lab2rgb(target_image * 255))

    target_images{numScales - i + 1} = target_image;

    "image scale "
    i
end

"Done"


function bounding_box = get_padded_bounding_box(source_image_alpha, offset)
    bounding_box = get_hole_bounding_box(source_image_alpha);
    image_size = size(source_image_alpha);
    if bounding_box(1) - offset < 1
        bounding_box(1) = 1;
    else
        bounding_box(1) = bounding_box(1) - offset;
    end
    if bounding_box(2) - offset < 1
        bounding_box(2) = 1;
    else
        bounding_box(2) = bounding_box(2) - offset;
    end
    if bounding_box(3) + offset > image_size(1)
        bounding_box(3) = image_size(1);
    else
        bounding_box(3) = bounding_box(3) + offset;
    end
    if bounding_box(4) + offset > image_size(2)
        bounding_box(4) = image_size(2);
    else
        bounding_box(4) = bounding_box(4) + offset;
    end

end

function bounding_box = get_hole_bounding_box(source_image_alpha)
    image_size = size(source_image_alpha);
    
    bounding_box = [image_size(1) + 1,image_size(2) + 1, 0, 0];
    for i=1:image_size(1)
        for j=1:image_size(2)
            if source_image_alpha(i,j) < .99  
                if i < bounding_box(1)
                    bounding_box(1) = i;
                end
                if j < bounding_box(2)
                    bounding_box(2) = j;
                end
                if i > bounding_box(3)
                    bounding_box(3) = i;
                end
                if j > bounding_box(4)
                    bounding_box(4) = j;
                end
            end
        end
    end
end


function image = blend_images(target_image, source_image, source_alpha)

    %adjusted_alpha = imbinarize(source_alpha);

    for i=1:3
        
        target_image(:,:,i) = target_image(:,:,i) .* (1-source_alpha);
        source_image(:,:,i) = source_image(:,:,i) .* source_alpha;
    end

    image = target_image + source_image;
end

function image = interpolate_hole(image, alpha)
    image_size = size(image)
    for i = 1:image_size(1)
        start_index = -1;
        encountered_hole = 0;

        for j = 1:image_size(2)
            if (alpha(i,j) > .999)
                if (encountered_hole == 1)
                    image = gradient_horizontal(image, i, start_index, j);
                    encountered_hole = 0;
                end
                start_index = j;
            else
                encountered_hole = 1;
            end
        end
        if start_index ~= -1
            image = gradient_horizontal(image, i, start_index, -1);
        end
    end
end

function image = gradient_horizontal(image, row, index1, index2)

    % cases where unfilled pixels intersect sides
    if (index1 == -1) 
        for i=1:index2-1
            image(row, i, :) = image(row, index2, :);
        end
        return
    elseif (index2 == -1)
        for i=index1+1:size(image,2)
            image(row, i, :) = image(row, index1, :);
        end
        return
    end

    color1 = image(row, index1, :);
    color2 = image(row, index2, :);

    for j=index1+1:index2-1 
        image(row, j, :) = color1 + (j - index1) * (color2 - color1) / (index2 - index1);
    end
end


% computes the NNF between patches in the target image and those in the source image
function NNF = patchMatchNNF_hole(target_image, source_image, source_alpha)
    global patch_size;
    
    fprintf("Computing NNF using PatchMatch...\n");
    
    target_size = size(target_image)

    offset = (patch_size - 1) / 2;

    source_padded = padarray(source_image,[offset offset],'both','symmetric');
    source_size = size(source_image)
    source_padded_size = size(source_padded)

    % initialize the NNF
    NNF = zeros(target_size(1), target_size(2), 3);
    
    tic

    valid_source_points = [];

    for i=1:source_size(1)
        for j=1:source_size(2)
            if source_alpha(i,j,1) > .999 
                valid_source_points(end+1,:) = [i,j];
            end
        end
    end
    
    iterations = 15;

    valid_source_size = size(valid_source_points);

    for i=1:target_size(1)
        for j=1: target_size(2)
            NNF(i,j,1:2) = valid_source_points(randperm(valid_source_size(1),1));
            source_pixel = [(NNF(i,j,1)+offset) (NNF(i,j,2)+offset)];
            NNF(i,j,3) = patch_distance(i,j, source_padded, target_image, target_size, offset, source_pixel);
        end
    end

    for iteration = 1: iterations

        iteration

        for operation_index = 1: 4

            NNF_offset = [0 0];
            i_start = 1;
            i_end = target_size(1);
            j_start = 1;
            j_end = target_size(2);
            invert_indices = 0;

            if operation_index == 1
                %"propogation towards +j"
                NNF_offset = [0, -1];
                j_start = 2;
            elseif operation_index == 2
                %"propogation towards +i"
                NNF_offset = [-1, 0];
                i_start = 2;
                invert_indices = 1;
            elseif operation_index == 3
                %"propogation towards -j"
                NNF_offset = [0, 1];
                j_end = target_size(2) - 1;
            elseif operation_index == 4
                %"propogation towards -i"
                NNF_offset = [1, 0];
                i_end = target_size(1) - 1;
                invert_indices = 1;
            end

            if (invert_indices == 0)
                for i=i_start:i_end
                    for j=j_start:j_end
                        NNF_val = propogate(i, j, NNF, NNF_offset, source_padded, source_padded_size, target_image, target_size, offset);
                        if (NNF_val(1) ~= 0)
                            NNF(i,j,:) = NNF_val;
                        end
                    end
                end
            else
                for j=j_start:j_end
                    for i=i_start:i_end
                        NNF_val = propogate(i, j, NNF, NNF_offset, source_padded, source_padded_size, target_image, target_size, offset);
                        if (NNF_val(1) ~= 0)
                            NNF(i,j,:) = NNF_val;
                        end
                    end
                end
            end

            
        end

        %"randomization"

        window_size = [target_size(1), target_size(2)];
        while window_size(1) > offset && window_size(2) > offset
            for i=1:target_size(1)
                for j=1:target_size(2)
                    NNF_val = randomize(i, j, NNF, window_size, source_padded, target_image, target_size, offset);
                    if (NNF_val(1) ~= 0)
                        NNF(i,j,:) = NNF_val;
                    end
                end
            end
            window_size = window_size / 2;
        end
        for local_iteration=1:3
            for i=1:target_size(1)
                for j=1:target_size(2)
                    NNF_val = randomize(i, j, NNF, [offset,offset], source_padded, target_image, target_size, offset);
                    if (NNF_val(1) ~= 0)
                        NNF(i,j,:) = NNF_val;
                    end
                end
            end
        end
    end

    toc

    fprintf("Done!\n");
end

function NNF_val = randomize(i, j, NNF, window_size, source_padded, target_image, target_size, offset)
    NNF_val = [0, 0, 0];
    i_offset_low = round(window_size(1));
    if i_offset_low >= i
        i_offset_low = i - 1;
    end
    i_offset_high = round(window_size(1));
    if i_offset_high + i > target_size(1)
        i_offset_high = target_size(1) - i;
    end
    j_offset_low = round(window_size(2));
    if j_offset_low >= j
        j_offset_low = j - 1;
    end
    j_offset_high = round(window_size(2));
    if j_offset_high + j > target_size(2)
        j_offset_high = target_size(2) - j;
    end

    NNF_offset = [randi([(-1*i_offset_low), i_offset_high],'uint8'), randi([(-1*j_offset_low), j_offset_high],'uint8')];
    NNF_index = [i+NNF_offset(1) j+NNF_offset(2)];

    source_pixel = [NNF(NNF_index(1),NNF_index(2),1) + offset, NNF(NNF_index(1),NNF_index(2),2) + offset];

    distance = patch_distance(i, j, source_padded, target_image, target_size, offset, source_pixel);
    if distance < NNF(i,j,3)
        NNF_val = [(source_pixel(1) - offset), (source_pixel(2) - offset) distance];
    end
end

function NNF_val = propogate(i, j, NNF, NNF_offset, source_padded, source_padded_size, target_image, target_size, offset)
    NNF_val = [0, 0, 0];
    %original_pixel = [i, j];

    NNF_index = [i+NNF_offset(1), j+NNF_offset(2)];

    %original_source_pixel = [NNF(NNF_index(1),NNF_index(2),1), NNF(NNF_index(1),NNF_index(2),2)];
    source_pixel = [NNF(NNF_index(1),NNF_index(2),1) - NNF_offset(1) + offset, NNF(NNF_index(1),NNF_index(2),2) - NNF_offset(2) + offset];

    if source_pixel(1) > source_padded_size(1) - offset
        source_pixel(1) = source_padded_size(1) - offset;
    elseif source_pixel(1) < offset + 1
        source_pixel(1) = offset + 1;
    end
    if source_pixel(2) > source_padded_size(2) - offset
        source_pixel(2) = source_padded_size(2) - offset;
    elseif source_pixel(2) < offset + 1
        source_pixel(2) = offset + 1;
    end

    distance = patch_distance(i, j, source_padded, target_image, target_size, offset, source_pixel);
    if distance < NNF(i,j,3)

        %current_distance = NNF(i,j,3);
      
        %assert(source_pixel(1) >= offset + 1, string(source_pixel(1)))
        %assert(source_pixel(1) <= source_padded_size(1) - offset, string(source_pixel(1)))
        %assert(source_pixel(2) >= offset + 1, string(source_pixel(2)))
        %assert(source_pixel(2) <= source_padded_size(2) - offset, string(source_pixel(1)))

        NNF_val = [(source_pixel(1) - offset), (source_pixel(2) - offset), distance];
    end
end


function dist = patch_distance(i, j, source_padded, target_image, target_size, offset, source_pixel)

    %assert (source_pixel(1) >= 1, string(source_pixel(1)))
    %assert (source_pixel(2) >= 1, string(source_pixel(2)))
    %assert (source_pixel(1) <= source_padded_size(1), string(source_pixel(1)))
    %assert (source_pixel(2) <= source_padded_size(2), string(source_pixel(2)))

    i_offset_low = offset;
    if i_offset_low >= i
        i_offset_low = i - 1;
    end
    i_offset_high = offset;
    if i_offset_high + i > target_size(1)
        i_offset_high = target_size(1) - i;
    end
    j_offset_low = offset;
    if j_offset_low >= j
        j_offset_low = j - 1;
    end
    j_offset_high = offset;
    if j_offset_high + j > target_size(2)
        j_offset_high = target_size(2) - j;
    end
    
    target_patch = target_image(i-i_offset_low:i+i_offset_high, j-j_offset_low:j+j_offset_high, :);
    source_patch = source_padded(source_pixel(1) - i_offset_low:source_pixel(1) + i_offset_high, source_pixel(2) - j_offset_low: source_pixel(2) + j_offset_high, :);
    
    dist = distance(target_patch, source_patch);
end

function dist = distance (patch1, patch2)
    patch_size = size(patch1);
    dist = sum(sum(sum((patch1 - patch2).^2))) / (patch_size(1) * patch_size(2));
end
    


% use the NNF to vote the source patches
function output = voteNNF(NNF, source_image, source_image_alpha)
    global patch_size;
    
    fprintf("Voting to reconstruct the final result...\n");
    
    source_size = size(source_image);
    target_size = size(NNF);

    % normalize NNF distances
   
    NNF(:,:,3) = NNF(:,:,3) ./ max(max(NNF(:,:,3)));

%     % get physical distances
%     physical_distances = zeros(target_size);
%     for i= 1:target_size(1)
%         for j= 1:target_size(2)
%             source_pixel = NNF(i, j, :);
%             physical_distances(i,j) = (i - source_pixel(2))^2 + (j - source_pixel(2))^2;
%         end
%     end
%     % normalize physical distances
%     physical_distances = physical_distances ./ max(max(physical_distances));

    offset = (patch_size - 1) / 2;
    
    % channel 1,2,3 : rgb
    % channel 4 : pixel count
    % channel 5 : distance count
    % channel 6 : neighbor likeness
    % channel 7 : distance from origin
    output = zeros(target_size(1) + (offset * 2), target_size(2) + (offset * 2), 7);
    
    % write your code here to reconstruct the output using source image
    % patches

    for i= 1:target_size(1)
        for j= 1:target_size(2)

            source_pixel = NNF(i, j, :);

            i_offset_low = offset;
            if i_offset_low >= source_pixel(1)
                i_offset_low = source_pixel(1) - 1;
            end
            i_offset_high = offset;
            if i_offset_high + source_pixel(1) > source_size(1)
                i_offset_high = source_size(1) - source_pixel(1);
            end
            j_offset_low = offset;
            if j_offset_low >= source_pixel(2)
                j_offset_low = source_pixel(2) - 1;
            end
            j_offset_high = offset;
            if j_offset_high + source_pixel(2) > source_size(2)
                j_offset_high = source_size(2) - source_pixel(2);
            end

            output_patch = output(i - i_offset_low + offset:i + i_offset_high + offset, j - j_offset_low + offset: j + j_offset_high + offset, :);

            source_patch = source_image(source_pixel(1) - i_offset_low:source_pixel(1) + i_offset_high, source_pixel(2) - j_offset_low: source_pixel(2) + j_offset_high, :);

            alpha_patch = source_image_alpha(source_pixel(1) - i_offset_low:source_pixel(1) + i_offset_high, source_pixel(2) - j_offset_low: source_pixel(2) + j_offset_high, :);

            neighbor_count = 0;
            total_neighbor_count = 0;

            if i > 1
                total_neighbor_count = total_neighbor_count + 1;
                if source_pixel(1) == NNF(i - 1, j, 1) + 1 && source_pixel(2) == NNF(i, j, 2)
                    neighbor_count = neighbor_count + 1;
                end
            end
            if i < target_size(1)
                total_neighbor_count = total_neighbor_count + 1;
                if source_pixel(1) == NNF(i + 1, j, 1) - 1 && source_pixel(2) == NNF(i, j, 2)
                    neighbor_count = neighbor_count + 1;
                end
            end
            if j > 1
                total_neighbor_count = total_neighbor_count + 1;
                if source_pixel(1) == NNF(i, j, 1) && source_pixel(2) == NNF(i, j - 1, 2) + 1
                    neighbor_count = neighbor_count + 1;
                end
            end
            if j < target_size(2)
                total_neighbor_count = total_neighbor_count + 1;
                if source_pixel(1) == NNF(i, j, 1) && source_pixel(2) == NNF(i, j + 1, 2) - 1
                    neighbor_count = neighbor_count + 1;
                end
            end

            neighbor_count = neighbor_count + 1;
            total_neighbor_count = total_neighbor_count + 1;

            % multiply by alpha
            source_patch = source_patch .* alpha_patch;

            % multiply by (1-distance)
            source_patch = source_patch * (1-source_pixel(3));

            % multiply by neighbor ratio
            source_patch = source_patch * (neighbor_count / total_neighbor_count);

%             % multiply by physical distance
%             source_patch = source_patch * (1-physical_distances(i,j));

            output_patch(:,:,1:3) = output_patch(:,:,1:3) + source_patch(:,:,1:3);

            % normalize image by pixel count + normalize by alpha
            output_patch(:,:,4) = output_patch(:,:,4) + alpha_patch(:,:,1);

            % add distance to count + normalize by alpha
            output_patch(:,:,5) = output_patch(:,:,5) + alpha_patch(:,:,1) * (1-source_pixel(3));

            % add neighbor ratio to count + normalize by alpha
            output_patch(:,:,6) = output_patch(:,:,6) + alpha_patch(:,:,1) * (neighbor_count / total_neighbor_count);

%            % add physical distance to count + normalize by alpha
%             output_patch(:,:,7) = output_patch(:,:,7) + alpha_patch(:,:,1) * (1-physical_distances(i,j));

            output(i - i_offset_low + offset:i + i_offset_high + offset, j - j_offset_low + offset: j + j_offset_high + offset, :) = output_patch;

        end
    end

    output = output(offset + 1:target_size(1) + offset, offset + 1:target_size(2) + offset, :);

%     % normalize physical distance by pixel count
%     output(:,:,7) = output(:,:,7) ./ output(:,:,4);
% 
%     % normalize by physical distance
%     output(:,:,1:3) = output(:,:,1:3) ./ output(:,:,7);

    % normalize neighbor count by pixel count
    output(:,:,6) = output(:,:,6) ./ output(:,:,4);

    % normalize by neighbor count
    output(:,:,1:3) = output(:,:,1:3) ./ output(:,:,6);

    % normalize distance by pixel count
    output(:,:,5) = output(:,:,5) ./ output(:,:,4);

    % normalize by distance on each pixel
    output(:,:,1:3) = output(:,:,1:3) ./ output(:,:,5);

    % remove small potential of NaN
    output(isnan(output))=0;

    % normalize by pixel count
    output = output(:,:,1:3) ./ output(:,:,4);

    assert(isequal(size(output,1:2), target_size(1:2)))
    
    fprintf("Done!\n");
end

