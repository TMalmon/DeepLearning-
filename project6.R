library(keras)
library(data.table)
library(ggplot2)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read in data
zip_train <- data.table::fread("zip.train")

x <- zip_train[, -1]
y <- zip_train[, 1]

#mnist <- keras::dataset_mnist()

#x <- mnist$train$x
#y <-mnist$train$y
#y <- to_categorical(y, 10)

img_size = 16


fold_vec <- vector(mode = "logical", length = nrow(x))
fold_vec <- sample(c(1,2,3,4,5), nrow(x), replace = TRUE)


is_train <- vector(mode = "logical", length = nrow(x))
is_train <- sample(c(TRUE, FALSE), nrow(x), replace = TRUE, prob = c(0.8, 0.2))


x_train_by_fold <- list()
y_train_by_fold <- list()
x_test_by_fold <- list()
y_test_by_fold <- list()


for (fold in unique(fold_vec) ) {
  x_by_fold <- x[fold_vec == fold,]
  y_by_fold <- y[fold_vec == fold ]
  
  x_train_by_fold[[fold]] = x_by_fold[is_train[fold_vec == fold],]
  y_train_by_fold[[fold]] = y_by_fold[is_train[fold_vec == fold] ]
  
  x_test_by_fold[[fold]] = x_by_fold[!is_train[fold_vec == fold],]
  y_test_by_fold[[fold]] = y_by_fold[!is_train[fold_vec == fold] ]
}




conv_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(img_size,img_size,1)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')


conv_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(lr = 0.001),
  metrics = c('accuracy')
)

dense_model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(img_size,img_size,1))%>%
  layer_dense(784, activation = "relu") %>%
  layer_dense(270, activation = "relu") %>%
  layer_dense(270, activation = "relu") %>%
  layer_dense(128, activation = "relu") %>%
  layer_dense(10, activation = "softmax")

dense_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(lr = 0.001),
  metrics = c('accuracy')
)


epochs = 20
batch_size = 100

conv_fold_history <- list()
for (fold in unique(fold_vec) ){
  
  x_train = x_train_by_fold[[fold]]
  x_train = as.matrix(x_train)
  x_train = array_reshape(x_train, c( nrow(x_train),img_size,img_size, 1) )
  
  history <- conv_model %>% fit(
    x_train, as.matrix(y_train_by_fold[[fold]]),
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2
  )
  conv_fold_history[[fold]] <- history$metrics
  
}


dense_fold_history <- list()
for (fold in unique( fold_vec ) ){
  
  x_train = x_train_by_fold[[fold]]
  x_train = as.matrix(x_train)
  x_train = array_reshape(x_train, c( nrow(x_train),img_size,img_size, 1) )
  
  history <- dense_model %>% fit(
    x_train, as.matrix(y_train_by_fold[[fold]]),
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2
  )
  dense_fold_history[[fold]] <- history$metrics
}

best_epochs_conv <- list()
best_epochs_dense <- list()
for (fold in unique( fold_vec ) ){
  best_epochs_conv[[fold]] <- which.min(conv_fold_history$fold)
  best_epochs_dense[[fold]] <- which.min(dense_fold_history$fold)
}



for (fold in unique( fold_vec ) ){
  x_train = x_train_by_fold[[fold]]
  x_train = as.matrix(x_train)
  x_train = array_reshape(x_train, c( nrow(x_train),img_size,img_size, 1) )
  
  history <- conv_model %>% fit(
    x_train, as.matrix(y_train_by_fold[[fold]]),
    batch_size = batch_size,
    epochs = best_epochs_conv[[fold]],
    validation_split = 0.0
  )
  conv_fold_history[[fold]] <- history$metrics
  
  
  x_train = x_train_by_fold[[fold]]
  x_train = as.matrix(x_train)
  x_train = array_reshape(x_train, c( nrow(x_train),img_size,img_size, 1) )
  
  history <- dense_model %>% fit(
    x_train, as.matrix(y_train_by_fold[[fold]]),
    batch_size = batch_size,
    epochs = best_epochs_dense[[fold]],
    validation_split = 0.0
  )
  dense_fold_history[[fold]] <- history$metrics
  
  
  conv_model %>%
    evaluate( array_reshape(as.matrix(x[!is_train,]), c(nrow(x[!is_train,]), img_size,img_size,1)), as.matrix(y[!is_train]) )
  
  dense_model %>%
    evaluate( array_reshape(as.matrix(x[!is_train,]), c(nrow(x[!is_train,]), img_size,img_size,1)), as.matrix(y[!is_train]) )
  
  # accuracy of baseline prediction
  y_tab <- table(y[is_train])
  y_baseline <- as.integer(names(which.max(y_tab)))
  mean(y[!is_train] == y_baseline)
}



