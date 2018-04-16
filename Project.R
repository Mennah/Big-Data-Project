library(factoextra)
library (plyr)
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

#============================================Part 2: Data Preparation=========================================#
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

        
#Remove unwanted columns 
drops <- c("date_time","Date", "srch_ci", "srch_co", "CheckInDate", "CheckOutDate")
dfm <- dfm[ , !(names(dfm) %in% drops)]




















