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
#clean environment 
rm(list=ls())
#set working directory
setwd("E:/BigData")
#============================================Part 1: Reading and Merging Datasets=========================================#

#Read Training and Destinations Datasets
dfm <- read.csv("E:/BigData/train/train_sample.csv")
dest <- read.csv("E:/BigData/destinations/destinations.csv")

#Analyzing Datasets
nrow (dfm)
factor (dfm$hotel_cluster)
hist(dfm$hotel_cluster)
head (dfm)

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
write.csv(dfm, file = "dfm.csv")

#============================================Part 2: Splitting Data===========================================#
set.seed(3033)
intrain  <- createDataPartition(dfm$hotel_cluster, p=0.7, list = FALSE)
train <- dfm[intrain,]
test  <- dfm[intrain,]
dfm   <- train
#============================================Part 3: Data Preparation=========================================#
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

#sub <- c ("site_name", "posa_continent", "is_mobile", "is_package", "is_booking", "channel", "hotel_continent", "hotel_country")
#dfm[sub]  <- lapply (dfm[sub],  factor)
#sub2 <- c ("hotel_market", "hotel_cluster", "Year", "Month", "Day", "CheckInMonth", "CheckInDay", 
#           "CheckOutMonth", "CheckOutDay")
#dfm[sub2] <- lapply (dfm[sub2], factor)
#sapply (dfm , class)

#negative values??
x <- factor (dfm$LengthOfStay)
levels (x)
#assuming user made a mistake and replaced check in date wwith check out and that is not possible in the website then 
#these records are faulty and we would remove them
dfm<-dfm[!(dfm$LengthOfStay < 0),]


#Remove unwanted columns 
drops <- c("date_time","Date", "srch_ci", "srch_co", "CheckInDate", "CheckOutDate")
dfm <- dfm[ , !(names(dfm) %in% drops)]

#Checking the correlation between the dependent variable and all other variables except those not present in test data
#IndependentVar <- c ("user_id" ,"site_name", "posa_continent", "user_location_country", "user_location_region", "user_location_city",       
#                     "orig_destination_distance","is_mobile" ,"is_package", "channel",                  
#                     "srch_adults_cnt", "srch_children_cnt","srch_rm_cnt", "srch_destination_id", "srch_destination_type_id", 
#                     "hotel_continent", "hotel_country", "hotel_market", "PC1","PC2", "PC3", "Year", 
#                     "Month", "Day", "CheckInYear", "CheckInMonth", "CheckInDay",               
#                     "CheckOutYear","CheckOutMonth","CheckOutDay", "LengthOfStay" )

#selecting independent variables
IndependentVar <-subset (dfm, select =  c(user_id, site_name, posa_continent, user_location_country, user_location_region, user_location_city,       
                                         orig_destination_distance,is_mobile ,is_package, channel,                  
                                         srch_adults_cnt, srch_children_cnt,srch_rm_cnt, srch_destination_id, srch_destination_type_id, 
                                         hotel_continent, hotel_country, hotel_market,PC1,PC2, PC3, Year, 
                                         Month,Day, CheckInYear, CheckInMonth, CheckInDay,               
                                         CheckOutYear,CheckOutMonth,CheckOutDay, LengthOfStay))

IndependentVar <- lapply (IndependentVar, as.numeric)
lapply (IndependentVar, class)
IndependentVar <- as.data.frame(IndependentVar)
#applying PCA on them to reduce dimensionality
IndepVar <- prcomp(x = IndependentVar, scale. = TRUE)
screeplot(IndepVar)

#Choosing The first three columns as representatives of the Destinations Dataset
FinalVariables<-cbind(IndependentVar$user_id,IndepVar$x[,1:10])

#Changing type of Destinations matrix to Dataframe
FinalVariables <- as.data.frame(FinalVariables)

dfm$hotel_cluster <- as.factor(dfm$hotel_cluster)

#Measure Variables Importance
model <- glm(formula = dfm$hotel_cluster ~ ., family = "binomial", data = FinalVariables)
varImp(model)
#pc2, pc4, pc9, pc6, pc7, pc8


dataset <- subset (FinalVariables, select =  c(V1, PC2, PC4, PC6, PC7, PC8, PC9))
dataset$hotel_cluster <- dfm$hotel_cluster                  

#===================================================Part 4: Prediction===========================================#
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












                   
                   