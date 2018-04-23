library (factoextra)
library (plyr)
library (caret)
library (stringr)
library(rattle)
library(NbClust)
library(cluster)
library(HSAUR)
library(Boruta)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ROCR)
library(e1071)
library(xgboost)
library(BBmisc)
library(klaR)
library(FactoMineR)
library(reshape2)
library (nnet)
library (gbm)
#clean environment 
rm(list=ls())
#set working directory
setwd("E:/BigData")
#============================================Part 1: Reading and Merging Datasets=========================================#

#Read Training and Destinations Datasets
dfm <- read.csv("E:/BigData/train/train_sample.csv")
dest <- read.csv("E:/BigData/destinations/destinations.csv")

#Getting principle component analysis to choose the most important columns from Destinations Dataset
dest2 <- prcomp(x = dest, scale = TRUE)
screeplot(dest2)

#Choosing The first three columns as representatives of the Destinations Dataset
destinations<-cbind(dest$srch_destination_id,dest2$x[,1:3])

#Changing type of Destinations matrix to Dataframe
destinations <- as.data.frame(destinations)
head(destinations, 20)
nrow (destinations)
u <- unique(dest$srch_destination_id)
length (u)

#Changing names of columns in Destinations Dataset
colnames(destinations) <- c("srch_destination_id", "PC1", "PC2", "PC3")
names (destinations)

#Joining Destinations and Training Datasets
dfm <- join(dfm, destinations, by = "srch_destination_id", type = "left", match = "all")
#============================================Part 2: Data Preparation=========================================#

#Since test data contains only booking events then we will remove those booking = false
dfm <- dfm[!(dfm$is_booking %in% c(0)),]

#Checking the presence of any NULLS in data
lapply (dfm, function(dfm) sum (is.na(dfm)))

#Date Time Field
dfm$Date  <- as.Date(dfm$date_time)
dfm$Year  <- as.numeric(format(dfm$Date, format = "%Y"))
dfm$Month <- as.numeric(format(dfm$Date, format = "%m"))
dfm$Day   <- as.numeric(format(dfm$Date, format = "%d"))

#Check in Date
dfm$CheckInDate  <- as.Date(dfm$srch_ci)
dfm$CheckInYear  <- as.numeric(format(dfm$CheckInDate, format = "%Y"))
dfm$CheckInMonth <- as.numeric(format(dfm$CheckInDate, format = "%m"))
dfm$CheckInDay   <- as.numeric(format(dfm$CheckInDate, format = "%d"))

#Check out Date
dfm$CheckOutDate  <- as.Date(dfm$srch_co)
dfm$CheckOutYear  <- as.numeric(format(dfm$CheckOutDate, format = "%Y"))
dfm$CheckOutMonth <- as.numeric(format(dfm$CheckOutDate, format = "%m"))
dfm$CheckOutDay   <- as.numeric(format(dfm$CheckOutDate, format = "%d"))

#Calculate length of stay 
dfm$LengthOfStay <- (dfm$CheckOutDate - dfm$CheckInDate)

#remove rows with nulls after the conversion
dfm<-dfm[!(is.na(dfm$CheckInDay)),]
dfm<-dfm[!(is.na(dfm$CheckInMonth)),]
dfm<-dfm[!(is.na(dfm$CheckInYear)),]
dfm<-dfm[!(is.na(dfm$CheckOutDay)),]
dfm<-dfm[!(is.na(dfm$CheckOutMonth)),]
dfm<-dfm[!(is.na(dfm$CheckOutYear)),]
dfm<-dfm[!(is.na(dfm$LengthOfStay)),]

#replace nulls by mean in orig_destination_distance, PC1, PC2, PC3
WithoutNA <- dfm$orig_destination_distance[!is.na(dfm$orig_destination_distance)]
m <- mean (WithoutNA)
dfm$orig_destination_distance <- ifelse(is.na(dfm$orig_destination_distance), m, dfm$orig_destination_distance)
sum (is.na(dfm$orig_destination_distance))

WithoutNA1 <- dfm$PC1[!is.na(dfm$PC1)]
m1 <- mean (WithoutNA1)
dfm$PC1 <- ifelse(is.na(dfm$PC1), m1, dfm$PC1)
sum (is.na(dfm$PC1))

WithoutNA2 <- dfm$PC2[!is.na(dfm$PC2)]
m2 <- mean (WithoutNA2)
dfm$PC2 <- ifelse(is.na(dfm$PC2), m2, dfm$PC2)
sum (is.na(dfm$PC2))

WithoutNA3 <- dfm$PC3[!is.na(dfm$PC3)]
m3 <- mean (WithoutNA3)
dfm$PC3 <- ifelse(is.na(dfm$PC3), m3, dfm$PC3)
sum (is.na(dfm$PC3))

sub <- c ("site_name" ,"posa_continent", "is_mobile", "is_package", "is_booking", "channel", "hotel_continent",
          "hotel_cluster", "Year", "Month", "Day", "CheckInMonth", "CheckInDay", "CheckOutMonth", "CheckOutDay", 
          "user_location_country", "user_location_region", "user_location_city", "srch_destination_id", 
          "srch_destination_type_id", "hotel_country", "hotel_market")
dfm[sub]  <- lapply (dfm[sub],  factor)

#negative values??
x <- factor (dfm$LengthOfStay)
levels (x)
#assuming user made a mistake and replaced check in date wwith check out and that is not possible in the website then 
#these records are faulty and we would remove them
dfm<-dfm[!(dfm$LengthOfStay < 0),]


#Remove unwanted columns 
drops <- c("date_time","Date", "srch_ci", "srch_co", "CheckInDate", "CheckOutDate")
dfm <- dfm[ , !(names(dfm) %in% drops)]

#Clustering
#first use a dataframe without the predicted variable
clustDFM <- subset(dfm, select = -c(hotel_cluster, is_booking, cnt, user_id))
#wdiff <- (nrow(clustDFM)-1)*sum(apply(clustDFM,2,var))
wdiff  <- 0
for (i in 1:15) 
{
  xx <- sum(kmodes(clustDFM,i, iter.max = 5)$withindiff)
  wdiff[i] <- ifelse(!is.na (xx), xx, 0)
}
plot(1:15, wdiff, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

#then best number of clusters -> 13
clust <- kmodes(clustDFM,5, iter.max = 10)


hotel_cluster <- clust$cluster

#============================================Part 3: Feature Selection =======================================#
#boruta
boruta_output <- Boruta(dataset$hotel_cluster ~ ., data= dataset , doTrace=2, pValue = 0.00000001)
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  # plot variable importance


#gradient boost
myboost=gbm(hotel_cluster ~ . ,data = clustDFM ,distribution = "multinomial",n.trees = 500,
            shrinkage = 0.01, interaction.depth = 4)


summary(myboost) #Summary gives a table of Variable Importance and a plot of Variable Importance

mydata <- subset (clustDFM, select =  c(CheckInYear, site_name, srch_adults_cnt, CheckInMonth, CheckOutYear, 
                                         user_location_region, CheckOutMonth))

mydata <- as.data.frame(mydata)
#============================================Part 4: Splitting Data===========================================#
set.seed(3033)
intrain  <- createDataPartition(mydata$hotel_cluster, p=0.7, list = FALSE)
train <- mydata[intrain,]
test  <- mydata[-intrain,]

hotel_cluster <- train$hotel_cluster
train <- subset(train, select = -c(hotel_cluster))
hotel_cluster_test <- test$hotel_cluster
test <- subset(test, select = -c(hotel_cluster))
#============================================Part 5: Prediction===============================================#

#-----------------------------------------------Naive Bayes-----------------------------------------------#

# train a naive bayes model
model <- naiveBayes(hotel_cluster ~., data=train, method="class")
# make predictions
predictions <- predict(model, test)
# summarize results
#mat <- confusionMatrix(predictions, test$hotel_cluster)
#print(mat)

tab <- table(predictions, hotel_cluster_test)
acc1 <- sum(diag(tab))/sum(tab)
print(acc1)
#----------------------------------------------DecisionTree-----------------------------------------------#

#Build the tree to "fit" the model
model <- rpart(hotel_cluster ~., data=train, method="class")
# make predictions
predictions <- predict(model, test)
# summarize results
tab <- table(predictions, hotel_cluster_test)
acc2 <- sum(diag(tab))/sum(tab)
print (acc2)

#-------------------------------------------Logistic Regression-------------------------------------------#

model <- multinom(train$hotel_cluster ~ train$PC2 + train$PC4 + train$PC6 + train$PC7 + train$PC8 + train$PC9, data = train)
# make predictions
predictions <- predict(model, test)
# summarize results
tab <- table(predictions, test$hotel_cluster)
acc3 <- sum(diag(tab))/sum(tab)

#----------------------------------------------Random Forest----------------------------------------------#

model <- randomForest(hotel_cluster ~.,data = train)
# make predictions
predictions <- predict(model, test)
# summarize results
tab <- table(predictions, hotel_cluster_test)
acc4 <- sum(diag(tab))/sum(tab)
print (acc4)
#------------------------------------------------XGBoost--------------------------------------------------#
label <- as.numeric(hotel_cluster)
data <-  as.matrix(train)

model <- xgboost(data = data, label = label, nround = 2, objective = "binary:logistic")
# make predictions
predictions <- predict(model, test)
# summarize results
tab <- table(predictions, hotel_cluster_test)
acc5 <- sum(diag(tab))/sum(tab)
print (acc5)
#--------------------------------------------------SVM----------------------------------------------------#
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
set.seed(3233)
svm_Linear <- train(train$hotel_cluster ~ train$PC2 + train$PC4 + train$PC6 + train$PC7 + train$PC8 + train$PC9, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

# make predictions
predictions <- predict(model, test)
# summarize results
tab <- table(predictions, test$hotel_cluster)
acc6 <- sum(diag(tab))/sum(tab)
