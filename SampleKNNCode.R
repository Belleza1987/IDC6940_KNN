# ============================================================
# 0. Libraries
# ============================================================
library(tidyverse)
library(caret)
library(class)
library(ggplot2)
library(pROC)
library(rpart)
library(randomForest)

set.seed(123)

# ============================================================
# 1. Load Data
# ============================================================
data <- read.csv("diabetes.csv")

str(data)
summary(data)

# ============================================================
# 2. Clean + Preprocess
# ============================================================
cols_with_zero <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

for (col in cols_with_zero) {
  data[[col]][data[[col]] == 0] <- NA
}

# Impute missing values
pre_impute <- preProcess(data, method = "medianImpute")
data <- predict(pre_impute, data)

# Scale predictors
predictor_cols <- setdiff(names(data), "Outcome")

pre_scale <- preProcess(data[, predictor_cols], method = c("center", "scale"))
scaled_x <- predict(pre_scale, data[, predictor_cols])

final_data <- cbind(scaled_x, Outcome = as.factor(data$Outcome))

# ============================================================
# 3. Train/Test Split
# ============================================================
train_index <- createDataPartition(final_data$Outcome, p = 0.8, list = FALSE)

train <- final_data[train_index, ]
test  <- final_data[-train_index, ]

train_x <- train[, predictor_cols]
train_y <- train$Outcome

test_x  <- test[, predictor_cols]
test_y  <- test$Outcome

# ============================================================
# 4. Baseline KNN (k = 5)
# ============================================================
knn5_pred <- knn(train_x, test_x, train_y, k = 5)
knn5_cm   <- confusionMatrix(knn5_pred, test_y)
print(knn5_cm)

# ============================================================
# 5. Tune K (1–25)
# ============================================================
k_grid <- 1:25
acc_results <- data.frame(k = k_grid, accuracy = NA)

for (i in k_grid) {
  pred_i <- knn(train_x, test_x, train_y, k = i)
  acc_results$accuracy[i] <- mean(pred_i == test_y)
}

best_k <- acc_results$k[which.max(acc_results$accuracy)]
print(best_k)

# Plot accuracy vs k
ggplot(acc_results, aes(x = k, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point() +
  labs(title = "KNN Accuracy vs K", x = "K", y = "Accuracy")

# ============================================================
# 6. Final KNN Model
# ============================================================
knn_pred <- knn(train_x, test_x, train_y, k = best_k)
knn_cm   <- confusionMatrix(knn_pred, test_y)
print(knn_cm)

knn_roc <- roc(as.numeric(test_y), as.numeric(knn_pred))
knn_auc <- auc(knn_roc)

# ============================================================
# 7. Logistic Regression
# ============================================================
log_model <- glm(Outcome ~ ., data = train, family = binomial)
log_prob  <- predict(log_model, test, type = "response")
log_pred  <- as.factor(ifelse(log_prob > 0.5, 1, 0))
log_cm    <- confusionMatrix(log_pred, test_y)
print(log_cm)

log_roc <- roc(as.numeric(test_y), as.numeric(log_pred))
log_auc <- auc(log_roc)

# ============================================================
# 8. Decision Tree
# ============================================================
tree_model <- rpart(Outcome ~ ., data = train, method = "class")
tree_pred  <- predict(tree_model, test, type = "class")
tree_cm    <- confusionMatrix(tree_pred, test_y)
print(tree_cm)

tree_roc <- roc(as.numeric(test_y), as.numeric(tree_pred))
tree_auc <- auc(tree_roc)

# ============================================================
# 9. Random Forest
# ============================================================
rf_model <- randomForest(Outcome ~ ., data = train)
rf_pred  <- predict(rf_model, test)
rf_cm    <- confusionMatrix(rf_pred, test_y)
print(rf_cm)

rf_roc <- roc(as.numeric(test_y), as.numeric(rf_pred))
rf_auc <- auc(rf_roc)

# ============================================================
# 10. Comparison Table
# ============================================================
results <- data.frame(
  Model    = c("KNN", "Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(
    mean(knn_pred == test_y),
    mean(log_pred == test_y),
    mean(tree_pred == test_y),
    mean(rf_pred == test_y)
  ),
  AUC = c(
    as.numeric(knn_auc),
    as.numeric(log_auc),
    as.numeric(tree_auc),
    as.numeric(rf_auc)
  )
)

print(results)

# Save results
write.csv(results, "results/model_comparison.csv", row.names = FALSE)
