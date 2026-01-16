#SAMPLE CODE FROM https://developer.ibm.com/tutorials/awb-implementing-knn-in-r/

#install.packages("caret")
#install.packages("httr")
#install.packages("readxl")
#install.packages("dplyr")
#install.packages("mt")

library(mt)
library(caret)
library(httr)
library(readxl)
library(dplyr)

GET("https://query.data.world/s/qmc5py3qjodoyglqcny7ihdowxwepf?dws=00000", write_disk(tf <- tempfile(fileext = ".xlsx")))
df <- data.frame(read_excel(tf))

head(df)

# Checking how balance is with the dependent variable
prop.table(table(df$Diabetes))
#This isn't quite balanced, but it's very common for medical data sets like this to be unbalanced. 
#It can be important to correctly account for the actual frequency of the condition in the population. 

#We'll select the non-demographic columns to begin and create a reduced data set:
df_reduced <- df[c("Diabetes", "Cholesterol", "Glucose", "BMI", "Waist.hip.ratio", "HDL.Chol", "Chol.HDL.ratio", "Systolic.BP", "Diastolic.BP", "Weight")]

#kNN is sensitive to scaling (of the variables). 
#it's helpful to first standardize the variables. Do this using the preProcess function of the caret package:
preproc_reduced <- preProcess(df_reduced[,2:10], method = c("scale", "center"))
df_standardized <- predict(preproc_reduced, df_reduced[, 2:10])

#Now we have scaled and centered values for our data.frame:
summary(df_standardized)

#Apply those scaled and centered values to our data using the bind_cols method from dpylr to normalize all of our data.
df_stdize <- bind_cols(Diabetes = df_reduced$Diabetes, df_standardized)
#Now we have a data set which will work well for KNN! 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#First create a test/train split so that we can train a KNN model. 
#use the createDataPartition from caret to randomly select 80% of the data to use for the training set and 20% of the data to use for the test set. 
#Then, create two different data.frames using partition point returned by createDataPartition.
set.seed(42)
# p indicates how much of the data should go into the first created partition
param_split<- createDataPartition(df_stdize$Diabetes, times = 1, p = 0.8, list = FALSE)
train_df <- df_stdize[param_split, ]
test_df <- df_stdize[-param_split, ]

#Train the KNN. 

#The trainControl method is a helper function for all the nuances of the KNN train method. It allows you to easily set up the KNN training parameters. 
trnctrl_df <- trainControl(method = "cv", number = 10)
#method: This is the resampling method used. Using "cv" or cross validation. 
#number: The number of folds because we're using cv. 

#use all variables to predict Diabetes. 
#then specify the training data, that we're training a KNN, and pass the training hyperparameters that we built with trainControl.
model_knn_df <- train(Diabetes ~ ., data = train_df, method = "knn", 
                      trControl = trnctrl_df, 
                      tuneLength = 10)

model_knn_df
#This returns a series of KNN models and their associated metrics and tells us that the optimal number of nearest neighbors to be considered is 11 and the model accuracy for that K is 0.9198387. 
#That's good but the Kappa score is 0.6138024, which can indicate that our model is not predicting the two classes with an even accuracy. 
#Remember that most of the records in our data set are not patients with diabetes. 
#Our model might simply guess that no patients have diabetes and achieve fairly high accuracy that way.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#we can find that simpler models have better accuracy. 
#To see which variables are having the most impact on our outcome variable, we can use fs.anova to check the analysis of variance for each variable and use that in a second KNN.
df_stdize$Diabetes = ifelse(df_stdize$Diabetes == "Diabetes", 1, 0)
fs <- fs.anova(df_stdize, df_stdize$Diabetes)
print(fs$fs.order)
#We can see that "Glucose" is the most important feature in our data set for predicting diabetes.

#two things here to improve our performance. 
#First, we'll balance our data set so that we have equal numbers of records with and without diabetes. 
#Second, we'll use just the Glucose readings to calculate the distance to our 'neighbors' in KNN:
positive_cases <- nrow(train_df[train_df$Diabetes == 'Diabetes',])

balanced = rbind(train_df[sample(which(train_df$Diabetes == 'No diabetes'), positive_cases),], train_df[train_df$Diabetes == 'Diabetes',])

trnctrl_df <- trainControl(method = "cv", number = 10)
model_knn_df_simple <- train(Diabetes ~ Glucose, data = balanced, method = "knn", 
                             trControl = trnctrl_df, 
                             tuneLength = 10)

model_knn_df_simple
#simpler model seems to have better performance. 
#With a K value of 15, we see an accuracy of 0.93 and a Kappa of 0.86. 
#Both of these scores are higher than we had seen with the larger model and the Kappa of 0.86 indicates that we are predicting both classes better although still not perfectly.
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Since we held out a testing set of 20% of our training data, we have an easy way to evaluate both of our models. 

#pass the model to predict and check the results against the actual test labels. 
#use a confusion matrix to assess how well our model predicts the test data:
predict_knn <- predict(model_knn_df, test_df)
confusionMatrix(as.factor(predict_knn), as.factor(test_df$Diabetes), positive = "Diabetes")

#Our everything model does a good job of identifying patients without diabetes but a poor job of identifying patients with diabetes. 
#As seen in the confusion matrix, the model only accurately identifies 25% of the records labeled 'Diabetes'.

predict_knn_simple <- predict(model_knn_df_simple, test_df)
confusionMatrix(as.factor(predict_knn_simple), as.factor(test_df$Diabetes), positive = "Diabetes")
#As we can see, our overall accuracy rate has fallen slightly, but our ability to predict the diabetes label is much much better. 
#This model correctly predicts 10 of the 12 true 'diabetes' records. 
#On the negative side though, the model mis-classifies 9 of the 66 'no diabetes' records as 'diabetes'. 
#This is a downside of training our KNN on a balanced data set: when we introduce new data points from our test set, 
#our model is a little too over-eager to label them 'diabetes' when they aren't.