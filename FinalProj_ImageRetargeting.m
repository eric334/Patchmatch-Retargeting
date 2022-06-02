clear;
clc;
close all;

global patch_size;
patch_size = 11;

%% PART 2 - IMAGE RETARGETTING
%%% Declaring parameters for the retargeting
minImgSize = 50;                % lowest scale resolution size for min(w, h)
outSizeFactor = [.9, .95];		% the ration of output image
numScales = 30;                 % number of scales (disributed logarithmically)
coarse_retarget_steps = 25;
logarithmic_direction = 1      % -1 - more images toward source
                                % 0 - linear distribution
                                % 1 - more images toward target

%% Preparing data for the retargeting
image = imread('source7.jpg');
image_size = size(image)

targetSize = round(outSizeFactor .* [image_size(1), image_size(2)])
imageLab = rgb2lab(image); % Convert the source and target Images
imageLab = double(imageLab)/255;

target_size_coarse_final = 0;

if targetSize(1) < targetSize(2)
    target_size_coarse_final = round([minImgSize, minImgSize * (targetSize(2)/targetSize(1))]);
else
    target_size_coarse_final = round([minImgSize * (targetSize(1)/targetSize(2)), minImgSize]);
end

target_size_coarse_initial = round(target_size_coarse_final ./ outSizeFactor);

target_size_coarse_final
target_size_coarse_initial

target_scale = target_size_coarse_initial(1) / image_size(1)

source_images = cell(numScales);

log_scale = 0;
if logarithmic_direction == -1
    log_scale = logspace(1, 0, numScales);
    log_scale = log_scale / ((log_scale(end) - log_scale(1)) / (1 - target_scale));
    log_scale = flip(log_scale + (1-log_scale(end)));
    
elseif logarithmic_direction == 1
    log_scale = logspace(0, 1, numScales);
    log_scale = log_scale / ((log_scale(end) - log_scale(1)) / (1 - target_scale));
    log_scale = flip(log_scale + (1-log_scale(end)));
end

figure
for i=1:numScales
    scale = 0;
    if logarithmic_direction == 0
        scale = 1 - ((i-1) * (1 - target_scale) / (numScales-1));
    else
        scale = log_scale(i);
    end

    join (["resize step ", i, " scale ", scale])

    source_images{i} = imresize(imageLab,scale, "bicubic");
    imshow(lab2rgb(source_images{i} * 255))
end

% Gradual Scaling - iteratively icrease the relative resizing scale between the input and
% the output (working in the coarse level).
%% STEP 1 - do the retargeting at the coarsest level

source_image = source_images{numScales};
target_image = source_image;

for i=1:coarse_retarget_steps
    join (["retarget step", i, "/", coarse_retarget_steps])
    source_size = size(source_image);

    new_size = round(target_size_coarse_initial + (target_size_coarse_final - target_size_coarse_initial) / coarse_retarget_steps * i);

    if (source_size(1:2) == new_size(1:2))
        continue
    end

    target_image = imresize(target_image, [new_size(1), new_size(2)], "bicubic");

    target_image = voteNNF(patchMatchNNF(target_image, source_image), source_image);

    imshow(lab2rgb(target_image * 255))
    
end

"resolution refinement"


%% STEP 2 - do resolution refinment 
% (upsample for numScales times to get the desired resolution)


for i=2:numScales
    source_image = source_images{numScales - i + 1};

    source_size = size(source_image);

    new_size = round((target_size_coarse_final ./ target_size_coarse_initial) .* source_size(1:2))

    target_image = imresize(target_image, [new_size(1), new_size(2)], "bicubic");

    target_image = voteNNF(patchMatchNNF(target_image, source_image), source_image);

    imshow(lab2rgb(target_image * 255))

    join (["image scale", i, "/", numScales])
end

imwrite(lab2rgb(target_image * 255),"output.png")

"Done"


% computes the NNF between patches in the target image and those in the source image
function NNF = patchMatchNNF(target_image, source_image)
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

    NNF(:,:,1) = randi([1, source_size(1)],target_size(1),target_size(2),'uint8');
    NNF(:,:,2) = randi([1, source_size(2)],target_size(1),target_size(2),'uint8');
    
    iterations = 15

    for i=1:target_size(1)
        for j=1: target_size(2)
            source_pixel = [(NNF(i,j,1)+offset) (NNF(i,j,2)+offset)];
            NNF(i,j,3) = patch_distance(i,j, source_padded, target_image, target_size, offset, source_pixel);
        end
    end

    for iteration = 1: iterations

        %iteration

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

    %assert (source_pixel(1) - i_offset_low >= 1, string(source_pixel(1) - i_offset_low))
    %assert (source_pixel(1) + i_offset_high <= source_padded_size(1), string(source_pixel(1) + i_offset_high))
    %assert (source_pixel(2) - j_offset_low >= 1, string(source_pixel(2) - j_offset_low))
    %assert (source_pixel(2) + j_offset_high <= source_padded_size(2), string(source_pixel(2) + j_offset_high))

    target_patch = target_image(i-i_offset_low:i+i_offset_high, j-j_offset_low:j+j_offset_high, :);
    source_patch = source_padded(source_pixel(1) - i_offset_low:source_pixel(1) + i_offset_high, source_pixel(2) - j_offset_low: source_pixel(2) + j_offset_high, :);
    
    dist = distance(target_patch, source_patch);
end

function dist = distance (patch1, patch2)
    patch_size = size(patch1);
    dist = sum(sum(sum((patch1 - patch2).^2))) / (patch_size(1) * patch_size(2));
end


% use the NNF to vote the source patches
function output = voteNNF(NNF, source_image)
    global patch_size;
    
    fprintf("Voting to reconstruct the final result...\n");
    
    source_size = size(source_image);
    target_size = size(NNF);

    % normalize NNF distances
   
    NNF(:,:,3) = NNF(:,:,3) ./ max(max(NNF(:,:,3)));

    offset = (patch_size - 1) / 2;
    
    % channel 1,2,3 : rgb
    % channel 4 : pixel count
    % channel 5 : distance count
    % channel 6 : neighbor likeness
    % channel 7 : distance from origin
    output = zeros(target_size(1) + (offset * 2), target_size(2) + (offset * 2), 6);
    

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

            alpha_patch = 1;

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

            % multiply by (1-distance)
            source_patch = source_patch * (1-source_pixel(3));

            % multiply by neighbor ratio
            source_patch = source_patch * (neighbor_count / total_neighbor_count);

            output_patch(:,:,1:3) = output_patch(:,:,1:3) + source_patch(:,:,1:3);

            % normalize image by pixel count
            output_patch(:,:,4) = output_patch(:,:,4) + ones(size(source_patch(1), size(source_patch(2))));

            % add distance to count
            output_patch(:,:,5) = output_patch(:,:,5) + (1-source_pixel(3));

            % add neighbor ratio to count
            output_patch(:,:,6) = output_patch(:,:,6) + (neighbor_count / total_neighbor_count);

            output(i - i_offset_low + offset:i + i_offset_high + offset, j - j_offset_low + offset: j + j_offset_high + offset, :) = output_patch;

        end
    end

    output = output(offset + 1:target_size(1) + offset, offset + 1:target_size(2) + offset, :);

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



