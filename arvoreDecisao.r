data(iris)

set.seed(42)
train_index <- sample(1:nrow(iris), nrow(iris)*0.7)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

library(rpart)
model <- rpart(Species ~ ., data = train_data, method = "class")

predictions <- predict(model, test_data, type = "class")
accuracy <- mean(predictions == test_data$Species)
print(paste("Acurácia:", accuracy))
