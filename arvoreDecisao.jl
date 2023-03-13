using RDatasets
iris = dataset("datasets", "iris")

using Random
Random.seed!(42)
train_index = rand(1:size(iris, 1), Int(size(iris, 1) * 0.7))
train_data = iris[train_index, :]
test_data = iris[setdiff(1:size(iris, 1), train_index), :]


using DecisionTree
model = DecisionTreeClassifier(max_depth=3)
fit!(model, train_data[:, 1:4], train_data[:, 5])

predictions = predict(model, test_data[:, 1:4])
accuracy = sum(predictions .== test_data[:, 5]) / length(predictions)
println("Acurácia: $accuracy")

