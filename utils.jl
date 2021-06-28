
function screeplot(S)
    hold(false)
    colorscheme("light")
    plot(
        1:length(S),
        S,
        title = "Scree Plot",
        xlabel = "Singular Value IDs",
        ylabel = "Singular Values",
        grid = false,
    )
    hold(true)
    aspectratio(24 / 16)
    display(scatter(1:length(S), S, grid = false))
end

function convert_gray(data)
    m, n = size(data)
    img_array = temp = Array{Array{Float64,2},1}(undef, m)
    # reshaping and converting to grayscale 
    for i = 1:m
        temp = colorview(RGB, reshape(data[i, :], 4, 28, 28)[1:3, :, :] ./ 255)
        temp_gray = Float64.(Gray.(temp))
        img_array[i] = temp_gray
    end
    return img_array
end

function convert_IR(data)
    m, n = size(data)
    img_array = temp = Array{Array{Float64,2},1}(undef, m)
    # reshaping and converting to grayscale 
    for i = 1:m
        temp = reshape(Float64.((data[i, :]) ./ 255), 4, 28, 28)[4, :, :]
        img_array[i] = temp
    end
    return img_array
end


function extract_nsvdvals(images)
    recodedImg = Array{Float64}(undef, 0, 10)
    dims = size(images)
    for i = 1:dims[1]
        img_singular = @pipe images[i] |> svdvals(_)[1:10]'
        recodedImg = vcat(recodedImg, img_singular)
    end
    recodedImg = mapslices(normalize, recodedImg, dims = 1)
    # mapslices is a base function that transform the given dimensions of array A using function f.
    # try ?mapslices to learn more. 
    # un-normalized features will take more time to converge 
end


# reversing the one-hot encoding 
function reverse_onehot(y_raw)
    replace(
        sum(y_raw .* [4 1 1 1] .* [1 3 1 1] .* [1 1 2 1], dims = 2),
        4 => "barren_land",
        3 => "trees",
        2 => "grass_land",
        1 => "none",
    )
end

function reverse_onehot_test(y_raw)
    norm_vals = unique(mapslices(sum, mapslices(normalize, y_raw, dims = 1), dims = 2))

    y_train = replace(
        mapslices(sum, mapslices(normalize, y_raw, dims = 1), dims = 2),
        norm_vals[1] => "barren_land",
        norm_vals[2] => "none",
        norm_vals[3] => "grass_land",
        norm_vals[4] => "trees",
    )
    return y_train
end


function hist_extract(images)
    m = length(images)
    counts = Array{Int64}(undef, 0, 17)
    for i = 1:m
        _ , temp = build_histogram(Gray.(images[i]), 16)
        counts = vcat(counts, temp')
    end
    return counts[:,Not(1)]
end


function extract_color_info(data)
    m, n = size(data)
    features = DataFrame(
        Rₘ = Float32[], Bₘ = Float32[], Gₘ = Float32[],

    )
    # reshaping and converting to grayscale 
    for i = 1:m
        temp = colorview(RGB, reshape(data[i, :], 4, 28, 28)[1:3, :, :] ./ 255)
        temp_r_mean    = mean(red.(temp))
        temp_b_mean    = mean(blue.(temp))
        temp_g_mean    = mean(green.(temp))

        push!(features, [temp_r_mean, temp_b_mean, temp_g_mean])
    end
    return features
end


function NDVI(data)
    m, n = size(data)
    mean_ndvi = Array{Float64}(undef,m)
    # reshaping and converting to grayscale 
    for i = 1:m
        temp_red = red.(colorview(RGB, reshape(data[1, :], 4, 28, 28)[1:3, :, :] ./ 255))
        temp_IR = reshape(data[1, :], 4, 28, 28)[4, :, :]
        mean_ndvi[i] = mean((temp_IR - temp_red) / (temp_IR + temp_red))
    end
    return mean_ndvi
end