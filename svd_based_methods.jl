using Base: Float64, project_names

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

# extract_nsvdvals takes the array of images, performs SVD and returns normalized singular values of each image 
X_train = extract_nsvdvals(images) #extract_nsvdvals function is defined in utils.jl
X_test = extract_nsvdvals(images_test)

y_train = reverse_onehot(y_raw) # reverse_onehot  function is defined in utils.jl
y_test = reverse_onehot(y_raw_test)

X_features = Matrix(extract_color_info(X_raw))
X_features_test = Matrix(extract_color_info(X_raw_test))

# Baseline Logisitic Regression Model
@sk_import linear_model:LogisticRegression
logistic_baseline = LogisticRegression(
    max_iter = 1000000,
    verbose = 1,
    class_weight = :balanced,
    solver = :sag,
    penalty = "none",
)
ScikitLearn.fit!(logistic_baseline, [X_train X_features], y_train)
ScikitLearn.score(logistic_baseline, [X_train X_features], y_train) # training set accuracy = 0.7841  
ScikitLearn.score(logistic_baseline, [X_test X_features_test X_dct], y_test)
cross_val_score(logistic_baseline, X_train, y_train)
# Accuracy with IR image: 0.704



@sk_import tree:DecisionTreeClassifier
tree = DecisionTreeClassifier(class_weight = :balanced)
ScikitLearn.fit!(tree, [X_train X_features kmeans.labels_], y_train)
ScikitLearn.score(tree, [X_train X_features kmeans.labels_], y_train)



@sk_import ensemble:BaggingClassifier
@sk_import svm:SVC
clf = BaggingClassifier(
    base_estimator = SVC(),
    n_estimators = 100,
    random_state = 100,
    bootstrap = true,
    oob_score = true,
    verbose = 1,
)
ScikitLearn.fit!(clf, X_train, y_train)
ScikitLearn.score(clf, X_train, y_train) # 84.24 accuracy 
ScikitLearn.score(clf, X_test, y_test) # 27.58 accuracy

ndvi = NDVI(X_raw)