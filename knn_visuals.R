###############################################
# KNN Diabetes Prediction - Updated Training Script
# Author: Jennifer Becerra
# IDC6940 - Capstone Project
###############################################

library(tidyverse)
library(caret)
library(pROC)

#------------------------------------------------
# 1. Load Data
#------------------------------------------------
df <- read.csv("diabetes.csv")

#------------------------------------------------
# 2. Replace impossible zeros with NA
#------------------------------------------------
cols_with_zero <- c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")
df[cols_with_zero] <- lapply(df[cols_with_zero], function(x) ifelse(x == 0, NA, x))

#------------------------------------------------
# 3. Separate predictors and outcome
#------------------------------------------------
predictors <- df %>% select(-Outcome)
outcome <- df$Outcome

#------------------------------------------------
# 4. Preprocess ONLY predictors (impute + scale)
#------------------------------------------------
preProc <- preProcess(predictors, method = c("medianImpute", "center", "scale"))
predictors_processed <- predict(preProc, predictors)

#------------------------------------------------
# 5. Recombine processed predictors + outcome
#------------------------------------------------
df_processed <- bind_cols(predictors_processed, Outcome = outcome)
df_processed <- na.omit(df_processed)

#------------------------------------------------
# 6. Train/Test Split (80/20)
#------------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(df_processed$Outcome, p = 0.8, list = FALSE)
train <- df_processed[trainIndex, ]
test  <- df_processed[-trainIndex, ]

train$Outcome <- factor(train$Outcome, levels = c(0,1), labels = c("No","Yes"))
test$Outcome  <- factor(test$Outcome,  levels = c(0,1), labels = c("No","Yes"))

#------------------------------------------------
# 7. Cross-Validation Setup
#------------------------------------------------
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

#------------------------------------------------
# 8. Train KNN Model Across k = 1 to 25
#------------------------------------------------
set.seed(123)
knn_model <- train(
  Outcome ~ .,
  data = train,
  method = "knn",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = expand.grid(k = 1:25)
)

# Best k value chosen by cross-validation
best_k <- knn_model$bestTune$k
print(paste("Best k:", best_k))

#------------------------------------------------
# 9. Plot Tuning Curve (Shows Why k = 23 is Best)
#------------------------------------------------
plot(knn_model, main = "KNN Tuning Curve (k = 1 to 25)")

#------------------------------------------------
# 10. Evaluate Model on Test Set
#------------------------------------------------
pred <- predict(knn_model, test)
prob <- predict(knn_model, test, type = "prob")[, "Yes"]

cm <- confusionMatrix(pred, test$Outcome)
auc <- roc(response = ifelse(test$Outcome == "Yes", 1, 0), predictor = prob)$auc

print(cm)
print(paste("AUC:", round(auc, 4)))

#------------------------------------------------
# 11. Save Updated Confusion Matrix Plot
#------------------------------------------------
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted","Actual","Freq")

p_cm <- ggplot(cm_df, aes(Predicted, Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, color = "white") +
  scale_fill_gradient(low = "#2CA6A4", high = "#0A2342") +
  labs(title = paste0("KNN Confusion Matrix (k = ", best_k, ")")) +
  theme_minimal(base_size = 16)

ggsave("visuals/confusion_matrix.png", p_cm, width = 10, height = 6, dpi = 150)


#------------------------------------------------
# 12. Save Feature Influence Plot
#------------------------------------------------
feat_names <- names(train)[names(train) != "Outcome"]
importance <- data.frame(
  Feature = feat_names,
  Score = sapply(train[feat_names], sd, na.rm = TRUE)
)

p_imp <- ggplot(importance, aes(x = reorder(Feature, Score), y = Score)) +
  geom_col(fill = "#0A2342") +
  coord_flip() +
  labs(title = "Feature Influence (Standardized)", x = "Feature", y = "Relative Influence") +
  theme_minimal(base_size = 16)

ggsave("visuals/feature_influence.png", p_imp, width = 10, height = 6, dpi = 150)


#------------------------------------------------
# 13. Save Workflow Diagram (optional)
#------------------------------------------------
library(DiagrammeR)
library(DiagrammeRsvg)

workflow <- grViz("
digraph workflow {
  graph [layout = dot, rankdir = TB]
  node [shape = box, style = filled, fillcolor = '#0A2342', fontcolor='white']
  A [label = 'Dataset']
  B [label = 'Preprocessing']
  C [label = 'Normalization']
  D [label = 'Train/Test Split']
  E [label = 'KNN Modeling']
  F [label = 'Evaluation']
  G [label = 'Conclusion']
  A -> B -> C -> D -> E -> F -> G
}
")

writeLines(export_svg(workflow), "visuals/workflow_diagram.svg")



