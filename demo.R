library(ggplot2); library(caret); library(randomForest); set.seed(2322)

setwd("Desktop/frank material/PML/PMLfinal/")
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
source("pml_write_files.R")

# identify is the column names between training and testing are same or not
sameName <- colnames(testing) == colnames(training)
# display the unmatched column names
colnames(training)[sameName==FALSE]
colnames(testing)[sameName==FALSE]

# define the dimension of training and testing data
dimTr <- dim(training); dimTe <- dim(testing)
head(colnames(training))
training <- training[,6:dimTr[2]]
testing <- testing[,6:dimTe[2]]

# create a vector of logicals for whether the predictor is a near zero variance predictor
uselessCol <- nearZeroVar(training, saveMetrics=TRUE)$nzv
for ( i in 1:dim(training)[2] )
  if (sum(is.na(training[,i]))/dimTr[1] > 0.8)
    uselessCol[i] <- TRUE

# delete the useless columns
training <- training[, uselessCol==FALSE]
testing <- testing[, uselessCol==FALSE]

# split data into two parts
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
dataTrain <- training[inTrain,]
dataCV <- training[-inTrain,]

# plot the data
qplot(accel_forearm_x, accel_forearm_y, col=classe, data=dataTrain)
qplot(accel_forearm_z, total_accel_forearm, col=classe, data=dataTrain)

# apply classification tree
modRP <- train(classe ~ ., method="rpart", data = dataTrain)
RP <- confusionMatrix(predict(modRP, dataCV), dataCV$classe)
RP

# apply GBM algorithm
modGBM <- train(classe ~ ., method="gbm", data=dataTrain)
GBM <- confusionMatrix(predict(modGBM, dataCV), dataCV$classe)
GBM

# apply random forest algorithm
rfTrain <- dataCV; rfCV <- dataTrain
modRF <- train(classe ~ ., method = 'rf', data = rfTrain, prox=TRUE,
               trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
RF <- confusionMatrix(predict(modRF, rfCV), rfCV$classe)
RF
# draw the correction plot
rfCV$predRight <- rfCV$classe ==  predict(modRF, rfCV)
qplot(accel_forearm_x, accel_forearm_y, col=predRight, data=rfCV)

# sum up all the methods
FinalOutcome <- data.frame(RP$overall[1], GBM$overall[1], RF$overall[1])
colnames(FinalOutcome) <- c("RP overall", "GBM overall", "RF overall")
FinalOutcome

# show the out-of-sample error
outOfSamErr <- 1-FinalOutcome
rownames(outOfSamErr) <- "Out of sample error"
outOfSamErr

# predict the classe of testing data
predTesting <- predict(modRF, testing)
predTesting
