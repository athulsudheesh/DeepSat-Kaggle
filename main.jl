
include("packages.jl")
include("utils.jl")

# Data Loading Pipelines =================# 
X_raw = CSV.File(joinpath("data","X_train_sat4.csv"), 
                limit = 10000, threaded=false) |> DataFrame |> Matrix

y_raw = CSV.File(joinpath("data","y_train_sat4.csv"), 
                limit = 10000, threaded=false) |> DataFrame |> Matrix
images = convert_gray(X_raw)

# extract_nsvdvals takes the array of images, performs SVD and returns normalized singular values of each image 
X_train = extract_nsvdvals(images)
y_train = reverse_onehot(y_raw)

@sk_import linear_model: LogisticRegression
logistic_baseline = LogisticRegression(max_iter=1000000, 
            verbose=1,class_weight=:balanced, solver=:sag,
            penalty="none")

ScikitLearn.fit!(logistic_baseline,X_train,y_train)

ScikitLearn.score(logistic_baseline,X_train,y_train)
            