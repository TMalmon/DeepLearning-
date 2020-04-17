library(keras)
library(data.table)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# read in data
spam_datatable <- data.table::fread("spam.data")


# split data into usable elements
x <- spam_datatable[, -58]
x_scale <- scale(x)

y <- spam_datatable[, 58]

is_train <- vector(mode = "logical", length = nrow(x))
is_train <- sample(c(TRUE, FALSE), nrow(x), replace = TRUE, prob = c(0.8, 0.2))


# training models
model_units <- c(10,100,1000)
num_models <- length(model_units)

models <- list()
metric_list <- list()
for( model_num in 1:num_models){
  
  model <- keras_model_sequential() %>%
  layer_dense( units = model_units[model_num], activation = "sigmoid", input_shape = c(ncol(x))) %>%
  layer_dense(1, activation = "sigmoid")
  
  model %>%
    compile(
      loss= "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  history <- model %>%
    fit(
      x = x_scale[is_train,],
      y = as.matrix(y[is_train]),
      epochs = 100,
      validation_split = 0.4,
      verbose = 2,
      view_metrics = FALSE
    )
  
  metric <- do.call(data.table::data.table, history$metrics)
  metric[, epoch := 1:.N]
  metric[, model_num := model_num]
  
  models[[model_num]] <- model
  metric_list[[model_num]] <- metric
}

metrics <- do.call(rbind, metric_list)


# calc min val loss and epoch by model num
mins <- metrics[, .(
  val_loss = min(val_loss),
  epoch = which.min(val_loss)
), by = model_num]


# plot val loss for each model
ggplot()+
  geom_line(aes(
    x=epoch, y=val_loss, group = model_num, color = model_num),
    data=metrics)+
  geom_line(aes(
    x=epoch, y=loss, group = model_num, color = model_num, linetype = "dashed"),
    data=metrics)+
  geom_point(aes(
    x=epoch, y=val_loss, color = model_num),
    data=metrics)+
  geom_point(aes(
    x=epoch, y=val_loss, color = model_num, size = 5),
    data=mins)


# re train with best epoch
best_epochs <- mins$epoch
best_runs <- list()

for( model_num in 1:length(models)) {
  history <- models[[model_num]] %>%
    fit(
      x = x_scale[is_train,],
      y = as.matrix(y[is_train]),
      epochs = best_epochs[model_num],
      validation_split = 0,
      verbose = 2,
      view_metrics = FALSE
    )
  
  metric <- do.call(data.table::data.table, history$metrics)
  metric[, epoch := 1:.N]
  metric[, model_num := model_num]
  
  best_runs[[model_num]] <- metric
}
best_runs <- do.call(rbind, best_runs)


# evaluate each model with test set
for(model_num in 1:length(models)){
  models[[model_num]] %>%
    evaluate( x_scale[!is_train,], as.matrix(y[!is_train]))
}

# accuracy of baseline prediction
y_tab <- table(y[is_train])
y_baseline <- as.integer(names(which.max(y_tab)))
mean(y[!is_train] == y_baseline)


