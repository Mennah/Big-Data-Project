library(randomForest)
library(party)
library(caret)
library(rpart)
library(rpart.plot)
library(ROCR)
library(e1071)

setwd("D:/El Gam3a/Year Four/Second Term/BigData/Labs/S18-BDA-LAB6/Lab 6 - Linear Regression")
d <- read.csv("data.csv")

############RandomForest#############
# Create the forest.
output.forest <- randomForest(hotel_cluster ~ pc2 + pc4 + pc9 + pc6 + pc7 + pc8,data = d)

# View the forest results.
print(output.forest)

###############SVM###################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)

svm_Linear <- train(V14 ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred

#############DecisionTree############

#Build the tree to "fit" the model
fit <- rpart(hotel_cluster ~ pc2 + pc4 + pc9 + pc6 + pc7 + pc8,
             method="class", 
             data=play_decision,
             control=rpart.control(minsplit=2, maxdepth = 3),
             parms=list(split='information'))
#split='information' : means split on "information gain" 

#plot the tree
rpart.plot(fit, type = 4, extra = 1)

summary(fit)
