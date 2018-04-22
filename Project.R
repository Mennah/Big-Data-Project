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

#dfm <- as.data.frame(dfm)

sub <- c ("site_name" ,"posa_continent", "is_mobile", "is_package", "is_booking", "channel", "hotel_continent",
          "hotel_cluster", "Year", "Month", "Day", "CheckInMonth", "CheckInDay", 
          "CheckOutMonth", "CheckOutDay")
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
wdiff <- (nrow(clustDFM)-1)*sum(apply(clustDFM,2,var))
for (i in 5:30) wdiff[i] <- sum(kmodes(clustDFM,i, iter.max = 5)$withindiff)
plot(1:30, wdiff, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

#then best number of clusters -> 13
clust <- kmodes(clustDFM,13, iter.max = 10)
dfm$cluster <- clust$cluster

#selecting independent variables
NumericVar <-subset (dfm, select =  c(site_name, user_location_country, user_location_region, user_location_city,       
                                         orig_destination_distance, srch_adults_cnt, srch_children_cnt,srch_rm_cnt, 
                                         srch_destination_id, srch_destination_type_id, 
                                         hotel_country, hotel_market,PC1,PC2, PC3,LengthOfStay))

CatVar <- subset (dfm, select = c(posa_continent, is_mobile, is_package, channel, hotel_continent,
                                      Year, Month, Day, CheckInMonth, CheckInDay, CheckOutMonth, CheckOutDay))

#Normalizing variables
NumericVar <- lapply (NumericVar, as.numeric)
#NumericVar <- lapply (NumericVar, function (NumericVar) normalize(NumericVar, method = "standardize", range = c(0, 1), margin = 1L, on.constant = "quiet"))
NumericVar <- as.data.frame(NumericVar)

#applying PCA on them to reduce dimensionality
FinalNumeric <- prcomp(x = NumericVar, scale. = TRUE)
screeplot(FinalNumeric)

#Choosing The first three columns as representatives of the Destinations Dataset
Numericdata<-cbind(dfm$user_id,FinalNumeric$x[,1:3])
Numericdata <- as.data.frame(Numericdata)

dataset <- CatVar
dataset$PC1 <- Numericdata$PC1
dataset$PC2 <- Numericdata$PC2
dataset$PC3 <- Numericdata$PC3


dataset$hotel_cluster <- dfm$cluster

dataset <- as.data.frame(dataset)

#============================================Part 3: Feature Selection =======================================#
#boruta
boruta_output <- Boruta(dataset$hotel_cluster ~ ., data= dataset , doTrace=2)
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  # plot variable importance

data <- subset (dataset, select =  c(posa_continent, is_mobile, is_package, channel, hotel_continent,
                                     Year, Month, Day, CheckInMonth, CheckInDay, CheckOutMonth,
                                     CheckOutDay, PC1, PC2, PC3, hotel_cluster))
 
#============================================Part 4: Splitting Data===========================================#
set.seed(3033)
intrain  <- createDataPartition(data$hotel_cluster, p=0.7, list = FALSE)
train <- dataset[intrain,]
test  <- dataset[intrain,]
#============================================Part 5: Prediction===============================================#
############Logistic Regression#########





############RandomForest#############
# Create the forest.
#output.forest <- randomForest(dataset$hotel_cluster ~ dataset$PC2 + dataset$PC4 + dataset$PC9 + dataset$PC6 + dataset$PC7 + dataset$PC8,data = dataset)

# View the forest results.
#print(output.forest)

###############SVM###################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)

svm_Linear <- train(dataset$hotel_cluster ~ dataset$PC2 , method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = testing)
test_pred

#############DecisionTree############

#Build the tree to "fit" the model
fit <- rpart(dataset$hotel_cluster ~ dataset$PC2,
             method="class", 
             data=dataset,
             control=rpart.control(minsplit=2, maxdepth = 10),
             parms=list(split='information'))
#split='information' : means split on "information gain" 

#plot the tree
rpart.plot(fit, type = 4, extra = 1)

summary(fit)

#############################XGBoost###########################################
bstSparse <- xgboost(data = dataset, label = dataset$hotel_cluster, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")












                   
                   