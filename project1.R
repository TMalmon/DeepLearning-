
library(data.table)
library(ggplot2)
library(WeightedROC)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


gradientDecent <- function( X, y, stepSize, maxIter )
{
  weightVector <- rep( 0, ncol(X) )
  weightMatrix <- matrix( 0, ncol( X ), maxIter)
  
  gradient <- 0
  
  #each iteration
  for( i in 1:maxIter )
  {
    prediction <- X%*%weightVector
    gradient <- colMeans(-as.numeric(y)*X / as.numeric(1+exp(y*(prediction))))

    weightVector = weightVector - stepSize * gradient
      
    weightMatrix[,i] = weightVector
  }
  
  return (weightMatrix)
}

#Sigmoid function
sigmoid <- function(z)
{
  g <- 1/(1+exp(-z))
  return(g)
}


#Error calculation
errorPercent <- function( predictions, answersTilde)
{
  ## too slow
  
  #count <- 0
  #for( i in 1: length(predictions) )
  #{
  #  if (round( sigmoid(predictions[i])) == answers[i])
  #    count = count+1
  #}
  #return ( 100*(1-count/length(predictions)) )
  
  predictions <- ifelse(predictions>0, 1, -1)
  100*colMeans(as.vector(answersTilde) != predictions)
}

# small math helper to remove clutter
meanLogisticLoss <- function ( predictions, answersTilde )
{
  colMeans(log(1+exp(-as.vector(answersTilde) * predictions)))
}




set.seed(2)
#----------------------------
# Spam Data Analysis
#----------------------------

X <- fread("spam.data.txt")

#data breakdown
sample <- sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = FALSE, prob = NULL)
train <- X[sample, ]
remaining  <- X[-sample, ]

sample <- sample.int(n = nrow(remaining), size = floor(.5*nrow(remaining)), replace = FALSE)
test <- remaining[sample, ]
validation  <- remaining[-sample, ]

trainY = train[,58]
trainYtilde <- ifelse(trainY == 1, 1, -1)
train = train[,-c(58)]
train = scale( train )

validationY = validation[,58]
validationYtilde <- ifelse(validationY == 1, 1, -1)
validation = validation[,-c(58)]
validation = scale( validation )

testY = test[,58]
testYtilde <- ifelse(testY == 1, 1, -1)
test = test[,-c(58)]
test = scale( test )

#set analytics
trainCount <- table( trainY )
validCount <- table( validationY )
testCount <- table( testY )

counts <- t(cbind(trainCount,validCount,testCount))
print(counts)

# Setup gradient descent
maxIterations <- 400
stepSize <- 0.7

#Retrieve weights and make predictions
weights <- gradientDecent(train , trainYtilde , stepSize , maxIterations)

predictionValid <- validation%*%weights
predictionTrain <- train%*%weights


#Plotting error percentage
trainErrorArray <- errorPercent(predictionTrain, trainYtilde)
validErrorArray <- errorPercent(predictionValid, validationYtilde)

plot( c(1:maxIterations), trainErrorArray, type = "l", col = "black", ylim=c(7,13)
      , ylab="Error Percent", xlab="Iterations")
  lines( c(1:maxIterations), validErrorArray, type = "l", col="red")
  legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)


#Plotting logistic loss
trainLoss <- meanLogisticLoss(predictionTrain, trainYtilde )
validLoss <- meanLogisticLoss(predictionValid, validationYtilde )

plot( c(1:maxIterations), trainLoss, type = "l", col = "black", ylim=c(0.2,0.5)
      , ylab="Logistic Loss", xlab="Iterations")
  lines( c(1:maxIterations), validLoss, type = "l", col="red")
  points( which.min(trainLoss) , trainLoss[which.min(trainLoss)], col="black")
  points( which.min(validLoss) , validLoss[which.min(validLoss)], col="red")
  legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)

  #best iteration
print(which.min(validLoss))
  
  
#error analytics
baseline <- ifelse(names(trainCount)[which.max(trainCount)]==1, 1, -1)
baseline <- rep(baseline, length(trainYtilde) )
bestPrediction <- train%*%weights[,which.min(trainLoss)]

baseTrain <- errorPercent( as.matrix(baseline), trainYtilde)
bestTrain <- errorPercent(bestPrediction, trainYtilde)


baseline <- ifelse(names(validCount)[which.max(validCount)]==1, 1, -1)
baseline <- rep(baseline, length(validationYtilde) )
bestPrediction <- validation%*%weights[,which.min(trainLoss)]

baseValid <- errorPercent(as.matrix(baseline), validationYtilde)
bestValid <- errorPercent(bestPrediction, validationYtilde)


baseline <- ifelse(names(testCount)[which.max(testCount)]==1, 1, -1)
baseline <- rep(baseline, length(testYtilde) )
bestPrediction <- test%*%weights[,which.min(trainLoss)]

baseTest <- errorPercent(as.matrix(baseline), testYtilde)
bestTest <- errorPercent(bestPrediction, testYtilde)

data.table( set = c( "Train", "Valid", "Test"), Baseline = c(baseTrain, 
            baseValid, baseTest), Best = c( bestTrain, bestValid, bestTest))



bestPredictionTilde = ifelse(bestPrediction > 0,1,-1)


falsePositives <- bestPredictionTilde == 1 & testYtilde == -1
negatives <- testYtilde == -1
truePositives <- bestPredictionTilde == 1 & testYtilde == 1
positives <- testYtilde == 1
TPR = sum(truePositives)/sum(positives)
FPR = sum(falsePositives)/sum(negatives)


baselineFalsePositives <- baseline == 1 & testYtilde == -1
baselineNegatives <- testYtilde == -1
baselineTruePositives <- baseline == 1 & testYtilde == 1
baselinePositives <- baseline == 1
baselineTPR = 0#sum(baselineTruePositives)/sum(baselinePositives)
baselineFPR = 0#sum(baselineFalsePositives)/sum(baselineNegatives)




weightedroc <- WeightedROC::WeightedROC(bestPredictionTilde, testYtilde)
baselineWeightedroc <- WeightedROC::WeightedROC(baseline, testYtilde)


ggplot()+
  geom_path(aes(
    FPR, TPR), data = weightedroc)+
  scale_fill_manual(values = c(predicted="white"))+
  coord_equal()+
  geom_point(
    aes(
      FPR,TPR,fill=TPRvsFPR ),
    data = data.table( TPRvsFPR="predicted"))+
  geom_path(aes(
    baselineFPR, baselineTPR), data = baselineWeightedroc)



#----------------------------
# SAHeart Data Analysis
#----------------------------

X <- fread("SAheart.data.txt")
  
#data breakdown
sample <- sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = FALSE, prob = NULL)
train <- X[sample, ]
remaining  <- X[-sample, ]
  
sample <- sample.int(n = nrow(remaining), size = floor(.5*nrow(remaining)), replace = FALSE)
test <- remaining[sample, ]
validation  <- remaining[-sample, ]
  
trainY = train[,11]
trainYtilde <- ifelse(trainY == 1, 1, -1)
train = train[,-c(11)]
train = scale( train )
  
validationY = validation[,11]
validationYtilde <- ifelse(validationY == 1, 1, -1)
validation = validation[,-c(11)]
validation = scale( validation )
  
testY = test[,11]
testYtilde <- ifelse(testY == 1, 1, -1)
test = test[,-c(11)]
test = scale( test )

#set analytics
trainCount <- table( trainY )
validCount <- table( validationY )
testCount <- table( testY )

counts <- t(cbind(trainCount,validCount,testCount))
print(counts)


# Setup gradient descent
maxIterations <- 50
stepSize <- .4

#Retrieve weights and make predictions
weights <- gradientDecent(train , trainYtilde , stepSize , maxIterations)

predictionValid <- validation%*%weights
predictionTrain <- train%*%weights


#Plotting error percentage
trainErrorArray <- errorPercent(predictionTrain, trainYtilde)
validErrorArray <- errorPercent(predictionValid, validationYtilde)

plot( c(1:maxIterations), trainErrorArray, type = "l", col = "black", ylim=c(27,36)
      , ylab="Error Percent", xlab="Iterations")
lines( c(1:maxIterations), validErrorArray, type = "l", col="red")
legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)


#Plotting logistic loss
trainLoss <- meanLogisticLoss(predictionTrain, trainYtilde )
validLoss <- meanLogisticLoss(predictionValid, validationYtilde )

plot( c(1:maxIterations), trainLoss, type = "l", col = "black", ylim=c(0.5,0.8)
      , ylab="Logistic Loss", xlab="Iterations")
lines( c(1:maxIterations), validLoss, type = "l", col="red")
points( which.min(trainLoss) , trainLoss[which.min(trainLoss)], col="black")
points( which.min(validLoss) , validLoss[which.min(validLoss)], col="red")
legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)

#best iteration
print(which.min(validLoss))


#error analytics
baseline <- ifelse(names(trainCount)[which.max(trainCount)]==1, 1, -1)
baseline <- rep(baseline, length(trainYtilde) )
bestPrediction <- train%*%weights[,which.min(trainLoss)]

baseTrain <- errorPercent( as.matrix(baseline), trainYtilde)
bestTrain <- errorPercent(bestPrediction, trainYtilde)


baseline <- ifelse(names(validCount)[which.max(validCount)]==1, 1, -1)
baseline <- rep(baseline, length(validationYtilde) )
bestPrediction <- validation%*%weights[,which.min(trainLoss)]

baseValid <- errorPercent(as.matrix(baseline), validationYtilde)
bestValid <- errorPercent(bestPrediction, validationYtilde)


baseline <- ifelse(names(testCount)[which.max(testCount)]==1, 1, -1)
baseline <- rep(baseline, length(testYtilde) )
bestPrediction <- test%*%weights[,which.min(trainLoss)]

baseTest <- errorPercent(as.matrix(baseline), testYtilde)
bestTest <- errorPercent(bestPrediction, testYtilde)

data.table( set = c( "Train", "Valid", "Test"), Baseline = c(baseTrain, 
            baseValid, baseTest), Best = c( bestTrain, bestValid, bestTest))



bestPredictionTilde = ifelse(bestPrediction > 0,1,-1)


falsePositives <- bestPredictionTilde == 1 & testYtilde == -1
negatives <- testYtilde == -1
truePositives <- bestPredictionTilde == 1 & testYtilde == 1
positives <- testYtilde == 1
TPR = sum(truePositives)/sum(positives)
FPR = sum(falsePositives)/sum(negatives)


baselineFalsePositives <- baseline == 1 & testYtilde == -1
baselineNegatives <- testYtilde == -1
baselineTruePositives <- baseline == 1 & testYtilde == 1
baselinePositives <- baseline == 1
baselineTPR = 0#sum(baselineTruePositives)/sum(baselinePositives)
baselineFPR = 0#sum(baselineFalsePositives)/sum(baselineNegatives)



weightedroc <- WeightedROC::WeightedROC(bestPredictionTilde, testYtilde)
baselineWeightedroc <- WeightedROC::WeightedROC(baseline, testYtilde)


ggplot()+
  geom_path(aes(
    FPR, TPR), data = weightedroc)+
  scale_fill_manual(values = c(predicted="white"))+
  coord_equal()+
  geom_point(
    aes(
      FPR,TPR,fill=TPRvsFPR ),
    data = data.table( TPRvsFPR="predicted"))+
  geom_path(aes(
    baselineFPR, baselineTPR), data = baselineWeightedroc)



#----------------------------
# zip.train Data Analysis
#----------------------------

X <- fread("zip.train")


#filter non 0-1 entries
X =  X[X[[1]]==0 | X[[1]]==1, ]

#do some breakdown now to remove NaN's
y = X[,1]
yTilde = ifelse(y == 1, 1, -1)

X = X[,-c(1)]
X = scale(X)
X = X[,complete.cases(t(X))]


#data split
sample <- sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = FALSE, prob = NULL)
train <- X[sample, ]
trainYtilde <- yTilde[sample,]
remaining  <- X[-sample, ]
remainingYtilde <- yTilde[-sample,]

sample <- sample.int(n = nrow(remaining), size = floor(.5*nrow(remaining)), replace = FALSE)
test <- remaining[sample, ]
testYtilde <- remainingYtilde[sample ]
validation  <- remaining[-sample, ]
validationYtilde <- remainingYtilde[-sample]


#set analytics
trainCount <- table( trainY )
validCount <- table( validationY )
testCount <- table( testY )

counts <- t(cbind(trainCount,validCount,testCount))
print(counts)


# Setup gradient descent
maxIterations <- 20000
stepSize <- 0.0001

#Retrieve weights and make predictions
weights <- gradientDecent(train , trainYtilde , stepSize , maxIterations)

predictionValid <- validation%*%weights
predictionTrain <- train%*%weights


#Plotting error percentage
trainErrorArray <- errorPercent(predictionTrain, trainYtilde)
validErrorArray <- errorPercent(predictionValid, validationYtilde)

plot( c(1:maxIterations), trainErrorArray, type = "l", col = "black", ylim=c(0,2)
      , ylab="Error Percent", xlab="Iterations")
lines( c(1:maxIterations), validErrorArray, type = "l", col="red")
legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)


#Plotting logistic loss
trainLoss <- meanLogisticLoss(predictionTrain, trainYtilde )
validLoss <- meanLogisticLoss(predictionValid, validationYtilde )


plot( c(1:maxIterations), trainLoss, type = "l", col = "black", ylim=c(0.01,0.7)
      , ylab="Logistic Loss", xlab="Iterations")
lines( c(1:maxIterations), validLoss, type = "l", col="red")
points( which.min(trainLoss) , trainLoss[which.min(trainLoss)], col="black")
points( which.min(validLoss) , validLoss[which.min(validLoss)], col="red")
legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)

#best iteration
print(which.min(validLoss))


#error analytics
baseline <- ifelse(names(trainCount)[which.max(trainCount)]==1, 1, -1)
baseline <- rep(baseline, length(trainYtilde) )
bestPrediction <- train%*%weights[,which.min(trainLoss)]

baseTrain <- errorPercent( as.matrix(baseline), trainYtilde)
bestTrain <- errorPercent(bestPrediction, trainYtilde)


baseline <- ifelse(names(validCount)[which.max(validCount)]==1, 1, -1)
baseline <- rep(baseline, length(validationYtilde) )
bestPrediction <- validation%*%weights[,which.min(trainLoss)]

baseValid <- errorPercent(as.matrix(baseline), validationYtilde)
bestValid <- errorPercent(bestPrediction, validationYtilde)


baseline <- ifelse(names(testCount)[which.max(testCount)]==1, 1, -1)
baseline <- rep(baseline, length(testYtilde) )
bestPrediction <- test%*%weights[,which.min(trainLoss)]

baseTest <- errorPercent(as.matrix(baseline), testYtilde)
bestTest <- errorPercent(bestPrediction, testYtilde)

data.table( set = c( "Train", "Valid", "Test"), Baseline = c(baseTrain, 
            baseValid, baseTest), Best = c( bestTrain, bestValid, bestTest))


bestPredictionTilde = ifelse(bestPrediction > 0,1,-1)


falsePositives <- bestPredictionTilde == 1 & testYtilde == -1
negatives <- testYtilde == -1
truePositives <- bestPredictionTilde == 1 & testYtilde == 1
positives <- testYtilde == 1
TPR = sum(truePositives)/sum(positives)
FPR = sum(falsePositives)/sum(negatives)


baselineFalsePositives <- baseline == 1 & testYtilde == -1
baselineNegatives <- testYtilde == -1
baselineTruePositives <- baseline == 1 & testYtilde == 1
baselinePositives <- baseline == 1
baselineTPR = 0#sum(baselineTruePositives)/sum(baselinePositives)
baselineFPR = 0#sum(baselineFalsePositives)/sum(baselineNegatives)



weightedroc <- WeightedROC::WeightedROC(bestPredictionTilde, testYtilde)
baselineWeightedroc <- WeightedROC::WeightedROC(baseline, testYtilde)


ggplot()+
  geom_path(aes(
    FPR, TPR), data = weightedroc)+
  scale_fill_manual(values = c(predicted="white"))+
  coord_equal()+
  geom_point(
    aes(
      FPR,TPR,fill=TPRvsFPR ),
    data = data.table( TPRvsFPR="predicted"))+
  geom_path(aes(
    baselineFPR, baselineTPR), data = baselineWeightedroc)






