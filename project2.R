library(data.table)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# predicts the most occuring label on everything
baseline <- function( x_mat, y_vec, x_test, y_test )
{
  zeros <- length(y_vec[ y_vec == 0 ])
  ones <- length(y_vec[ y_vec == 1 ])
  baseline = max( zeros, ones )
  baselinePred <- vector(mode= "numeric" , length = nrow(y_test))+baseline
  
  error <- colMeans(y_test == baselinePred)
  
  return (error)
}

# runs nncv with 1 neighbor
nearest_1_neighbors <- function( x_mat, y_vec, x_new,
                                 num_folds = 5 )
{
  return (nearest_neighbors_cv(x_mat, y_vec, x_new,
                               num_folds, 1))
}


# runs an algorithm with a fold vector to get an average error
k_fold_cv <- function(x_mat, y_vec, compute_predictions, fold_vec) {
  error_vec <- vector(length = length(unique(fold_vec)))
  
  for (fold in unique(fold_vec)) {
    x_new <- x_mat[fold_vec == fold, ]
    y_new <- y_vec[fold_vec == fold]
    
    x_train <- x_mat[fold_vec != fold, ]
    y_train <- y_vec[fold_vec != fold]
    

    pred_new <- compute_predictions(x_train, y_train, x_new)
    
    loss <- colMeans(y_new == as.vector(pred_new))
    
    error_vec[fold] <- loss
  }
  return(error_vec)
}

# runs nn alg on training set with folds then give error
# rate on x_new
nearest_neighbors_cv <- function(x_mat, y_vec, x_new,
                                 num_folds = 5, max_neighbors = 20) {
  
  validation_fold_vec <- sample(rep(1:num_folds, l = nrow(x_mat)))
  error_mat <- matrix(nrow = num_folds, ncol = max_neighbors)
  mean_error_vec <- vector(length = max_neighbors)
  
  
  for (num_neighbors in 1:max_neighbors) {
    wrap_knn <- function(x_mat,y_vec, x_new, k)class::knn(x_mat,as.matrix(x_new),as.matrix(y_vec), k = max_neighbors)
    
    error <- k_fold_cv(x_mat, y_vec,
                       compute_predictions = wrap_knn,
                       validation_fold_vec)
    error_mat[, num_neighbors] <- error
  }
  mean_error_vec <- colMeans(error_mat)
  best_neighbors <- which.min(mean_error_vec)
  
  best_pred <- class::knn(x_mat, as.matrix(x_new), as.matrix(y_vec), k = best_neighbors)
  
  results_list <- list(best_pred, mean_error_vec, error_mat)
  names(results_list) <- c("best", "mean_error_vec", "error_mat")
  
  return(results_list)
}

# read in data
spam_datatable <- data.table::fread("spam.data")

# split data into usable elements
x <- spam_datatable[, -58]
x_scale <- scale(x)

y <- spam_datatable[, 58]

# run nncv for an error rate to graph
result <- nearest_neighbors_cv(x_scale, y, t(vector(length = ncol(x))))
ggplot() +
  geom_line(aes(
    c(1:20), result$mean_error_vec),
    data = as.data.table(result$mean_error_vec ), size = 3)+
  geom_point( aes( which.min(result$mean_error_vec), min(result$mean_error_vec)  ), size = 7)

# create new test fold vector
test_fold_vec <- sample(rep(1:4, l = nrow(x_scale)))

# make a table to display of zeros and ones in each fold
fold_list<- list()
for (fold in unique(test_fold_vec))
{
  fold_vals <- as.vector(y)[ fold == test_fold_vec ]
  zeros <- fold_vals[ as.vector(fold_vals==0) ]
  ones <- fold_vals[ as.vector(fold_vals==1) ]
  
  fold_list[[fold]] <- ( list( zeros = length(t(zeros)), ones = length(t(ones))) )
}
do.call(rbind, fold_list)





# run kfoldcv with different algs to prove nncv is best
err.dt.list <- list()
## assign folds.
for(test.fold in 1:4){
  ## split into train/test sets.
  x_new <- x_scale[test_fold_vec == test.fold, ]
  y_new <- y[test_fold_vec == test.fold]
  
  x_train <- x_scale[test_fold_vec != test.fold, ]
  y_train <- y[test_fold_vec != test.fold]
  
  wrap_knncv <- function(x_train,y_train, x_new)nearest_neighbors_cv(x_train,as.matrix(x_new),as.matrix(y_train))
  wrap_1nn <- function(x_train,y_train, x_new)nearest_1_neighbors(x_train,as.matrix(x_new),as.matrix(y_train))
  wrap_baseline <- function(x_train,y_train, x_new)baseline(x_train,as.matrix(x_new),as.matrix(y_train))
  
  for(algorithm in c("wrap_knncv", "wrap_1nn", "wrap_baseline")){
    ## run algorithm and store test error.
    error.percent <- k_fold_cv( x_train, y_train, compute_predictions = algorithm, test_fold_vec)
    err.dt.list[[paste(test.fold, algorithm)]] <- data.table(
      test.fold, algorithm, error.percent)
  }
}
err.dt <- do.call(rbind, err.dt.list)


