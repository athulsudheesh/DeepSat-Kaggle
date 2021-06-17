
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
        temp = colorview(RGB,reshape(X_raw[i,:],4,28,28)[1:3,:,:] ./255)
        temp_gray = Float64.(Gray.(temp))
        img_array[i] = temp_gray
    end
    return img_array
end

function convert_IR(data)
    m,n = size(data)
    img_array = temp = Array{Array{Float64,2},1}(undef,m)
# reshaping and converting to grayscale 
    for i in 1:m 
        temp = reshape(Float64.((data[i,:]) ./255),4,28,28)[4,:,:]
        img_array[i] = temp
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
# un-normalized features will take more time to converge 
end


# reversing the one-hot encoding 
function reverse_onehot(y_raw)
    replace(sum(y_raw .* [4 1 1 1] .* [1 3 1 1] .* [1 1 2 1], dims=2), 
                4=>"barren_land", 3=>"trees", 2=>"grass_land", 1=>"none")
end

function reverse_onehot_test(y_raw)
    norm_vals = unique(mapslices(sum,mapslices(normalize, y_raw,dims=1), dims=2))

    y_train = replace(mapslices(sum,mapslices(normalize, y_raw,dims=1), dims=2), 
            norm_vals[1]=> "barren_land",
            norm_vals[2] => "none",
            norm_vals[3] => "grass_land",
            norm_vals[4] => "trees")
    return y_train
end