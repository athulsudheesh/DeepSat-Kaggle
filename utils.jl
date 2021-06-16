
function screeplot(S)
    hold(false)
    colorscheme("light")
    plot(1:length(S),S, title="Scree Plot", 
        xlabel = "Singular Value IDs",
        ylabel = "Singular Values", grid=false)
    hold(true)
    aspectratio(24/16)
    display(scatter(1:length(S),S,grid=false))
end

function convert_gray(data)
    m,n = size(data)
    img_array = temp = Array{Array{Float64,2},1}(undef,m)
# reshaping and converting to grayscale 
    for i in 1:m 
        temp = reshape(Float64.((data[i,:]) ./255),28,28,4)[:,:,1:3]
        temp_gray = (temp[:,:,1] + temp[:,:,2] + temp[:,:,3]) ./3 
        img_array[i] = temp_gray
    end
    return img_array
end

function extract_nsvdvals(images)
    recodedImg = Array{Float64}(undef, 0, 28) 
    dims = size(images)
    for i in 1:dims[1] 
        img_singular = @pipe images[i] |> svdvals(_)'
        recodedImg = vcat(recodedImg, img_singular)
    end
recodedImg = mapslices(normalize, recodedImg, dims=1) 
# mapslices is a base function that transform the given dimensions of array A using function f.
# try ?mapslices to learn more. 
end


# reversing the one-hot encoding 
function reverse_onehot(y_raw)
    norm_vals = unique(mapslices(sum,mapslices(normalize, y_raw,dims=1), dims=2))

    y_train = replace(mapslices(sum,mapslices(normalize, y_raw,dims=1), dims=2), 
            norm_vals[1]=> "barren_land",
            norm_vals[2] => "trees",
            norm_vals[3] => "none",
            norm_vals[4] => "grass_land")
    return y_train
end