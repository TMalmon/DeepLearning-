
library(data.table)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


gradientDecent <- function( X, y, stepSize, maxIter )
{
  weightVector <- rep( 0, ncol(X) )
  weightMatrix <- matrix( 0, ncol( X ), maxIter)
  
  gradient <- 0
  
  #each iteration
  for( i in 1:maxIter )
  {
    #each parameter
    for( j in 1: ncol( X ) )
    {
      gradient <- (1/nrow(X)) * sum(((X%*%weightVector)- y)*X[,j])

      weightVector[j] = weightVector[j] - stepSize * gradient
      
    }
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
errorPercent <- function( predictions, answers)
{
  count <- 0
  for( i in 1: length(predictions) )
  {
    if (round( sigmoid(predictions[i])) == answers[i])
      count = count+1
  }
  return ( 100*(1-count/length(predictions)) )
}


#----------------------------
# Spam Data Analysis
#----------------------------

X <- fread("spam.data.txt")

set.seed(1)


sample <- sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = FALSE, prob = NULL)
train <- X[sample, ]
test  <- X[-sample, ]

sample <- sample.int(n = nrow(test), size = floor(.5*nrow(test)), replace = FALSE)
test <- X[sample, ]
validation  <- X[-sample, ]

trainY = train[,58]
train = train[,-c(58)]
train = scale( train )

validationY = validation[,58]
validation = validation[,-c(58)]
validation = scale( validation )

maxIterations <- 50
stepSize <- 0.1

weights <- gradientDecent(train , trainY , stepSize , maxIterations)

predictionValid <- validation%*%weights
predictionTrain <- train%*%weights

print( errorPercent(predictionTrain[,1] , trainY) )
print( errorPercent(predictionValid[,1] , validationY) )


trainErrorArray <- rep( 0, maxIterations)
for( i in 1:maxIterations)
{
  trainErrorArray[i] = errorPercent(predictionTrain[,i] , trainY )
}

validErrorArray <- rep( 0, maxIterations)
for( i in 1:maxIterations)
{
  validErrorArray[i] = errorPercent(predictionValid[,i] , validationY )
}

plot( c(1:maxIterations), trainErrorArray, type = "l", col = "black", ylim=c(9,20)
      , ylab="Error Percent", xlab="Iterations")

lines( c(1:maxIterations), validErrorArray, type = "l", col="red")

legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)



#----------------------------
# SAHeart Data Analysis
#----------------------------

X <- fread("SAheart.data.txt")

set.seed(1)


sample <- sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = FALSE, prob = NULL)
train <- X[sample, ]
test  <- X[-sample, ]

sample <- sample.int(n = nrow(test), size = floor(.5*nrow(test)), replace = FALSE)
test <- X[sample, ]
validation  <- X[-sample, ]

trainY = train[,11]
train = train[,-c(11)]
train = scale( train )

validationY = validation[,11]
validation = validation[,-c(11)]
validation = scale( validation )

maxIterations <- 200
stepSize <- 0.01

weights <- gradientDecent(train , trainY , stepSize , maxIterations)

predictionValid <- validation%*%weights
predictionTrain <- train%*%weights

print( errorPercent(predictionTrain[,1] , trainY) )
print( errorPercent(predictionValid[,1] , validationY) )


trainErrorArray <- rep( 0, maxIterations)
for( i in 1:maxIterations)
{
  trainErrorArray[i] = errorPercent(predictionTrain[,i] , trainY )
}

validErrorArray <- rep( 0, maxIterations)
for( i in 1:maxIterations)
{
  validErrorArray[i] = errorPercent(predictionValid[,i] , validationY )
}

plot( c(1:maxIterations), trainErrorArray, type = "l", col = "black", ylim=c(9,50)
      , ylab="Error Percent", xlab="Iterations")

lines( c(1:maxIterations), validErrorArray, type = "l", col="red")

legend("topright",
       c("Train","Validation"),
       fill=c("black","red")
)



#----------------------------
#
#----------------------------



