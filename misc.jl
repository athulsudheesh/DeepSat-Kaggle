using CSV, DataFrames
using Images
using ImageTransformations
using ImageView
using ImageCore
using Augmentor
using LinearAlgebra
using GRUtils
using JuliaDB
#======= utility functions =====# 

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

# Data Loading Pipelines =================# 
data = CSV.File("X_train_sat4.csv", limit = 100) |> DataFrame



function img_process(data)
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

