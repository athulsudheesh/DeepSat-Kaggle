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
ScikitLearn.fit!(clf, X_train, labels_train)
ScikitLearn.score(clf, X_train, labels_train) # 86.37 accuracy 
ScikitLearn.score(clf, X_test, labels_test) # 86.49 accuracy