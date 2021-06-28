include("packages.jl")
include("utils.jl")

# Data Loading Pipelines =================# 
X_raw =
    CSV.File(joinpath("data", "X_train_sat4.csv"), limit = 10000, threaded = false) |>
    DataFrame |>
    Matrix

X_raw_test =
    CSV.File(joinpath("data", "X_test_sat4.csv"), limit = 10000, threaded = false) |>
    DataFrame |>
    Matrix

y_raw =
    CSV.File(joinpath("data", "y_train_sat4.csv"), limit = 10000, threaded = false) |>
    DataFrame |>
    Matrix

y_raw_test =
    CSV.File(joinpath("data", "y_test_sat4.csv"), limit = 10000, threaded = false) |>
    DataFrame |>
    Matrix

images = convert_gray(X_raw) # convert_gray  function is defined in utils.jl
images_test = convert_gray(X_raw_test) 


IR = convert_IR(X_raw)
y_train = reverse_onehot(y_raw) # reverse_onehot  function is defined in utils.jl
y_test = reverse_onehot(y_raw_test)

hist_train = hist_extract(images)#extract_nsvdvals function is defined in utils.jl
svd_train = extract_nsvdvals(images)
X_train = [hist_train svd_train]
X_test = hist_extract(images_test)


@sk_import linear_model:LogisticRegression
logistic_baseline = LogisticRegression(
    max_iter = 1000000,
    verbose = 1,
    class_weight = :balanced,
    solver = :saga,
)
ScikitLearn.fit!(logistic_baseline, X_train, y_train)
ScikitLearn.score(logistic_baseline, X_train, y_train)
ScikitLearn.score(logistic_baseline, X_test, y_test)

