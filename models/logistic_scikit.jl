include(joinpath(pwd(),"src", "packages.jl"))
include(joinpath(pwd(),"src","utils.jl"))

# Loading the train & test CSV (only the first n rows in each CSV)
path = "data"; n = 10000;
X_raw, y_raw, X_raw_test, y_raw_test = load_raw_data(path,n) 

images_train = convert_gray(X_raw) # convert_gray function is defined in src/utils.jl
images_test = convert_gray(X_raw_test)
labels_train = reverse_onehot(y_raw) #reverse_onehot  function is defined in src/utils.jl
labels_test = reverse_onehot(y_raw_test)

# Feature Engineering 
svd_features_train = extract_nsvdvals(images_train) #extract_nsvdvals function is defined in utils.jl
svd_feratures_test = extract_nsvdvals(images_test)
color_features_train = Matrix(extract_color_info(X_raw))
color_features_test = Matrix(extract_color_info(X_raw_test))
X_train = [svd_features_train color_features_train]
X_test = [svd_feratures_test color_features_test]

# Model Definition & Fiting 
@sk_import linear_model:LogisticRegression
logistic_baseline = LogisticRegression(
    max_iter = 1000000,
    verbose = 1,
    class_weight = :balanced,
    solver = :sag,
    penalty = "none",
)
ScikitLearn.fit!(logistic_baseline, X_train, labels_train)

# Model Accuracy Metrics 
ScikitLearn.score(logistic_baseline, X_train, labels_train) # training set accuracy = 86.65%  
ScikitLearn.score(logistic_baseline, X_test, labels_test)   # test set accuracy = 86.86%
