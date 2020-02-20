
library(data.table)
library(ggplot2)


k_fold_cv <- function(x_mat, y_vec, compute_predictions, fold_vec) {
    error_vec <- vector(length = length(unique(fold_vec)))

    for (fold in unique(fold_vec)) {
        x_new <- x_mat[fold_vec == fold, ]
        y_new <- y_vec[fold_vec == fold]

        x_train <- x_mat[fold_vec != fold, ]
        y_train <- y_vec[fold_vec != fold]

        pred_new <- compute_predictions(x_train, y_train, x_new)

        loss <- (y_new == pred_new) / length(y_new)

        error_vec[fold] <- loss
    }
    return(error_vec)
}


nearest_neighbors_cv <- function(x_mat, y_vec, x_new,
                                 num_folds = 5, max_neighbors = 20) {
    validation_fold_vec <- sample(rep(1:num_folds, l = nrow(x_mat)))
    error_mat <- matrix(num_folds, max_neighbors)
    mean_error_vec <- vector(length = max_neighbors)

    for (num_neighbors in 1:max_neighbors) {
        error <- k_fold_cv(x_mat, y_vec,
            compute_predictions =
                class::knn(x_mat, x_new, y_vec[, 1], k = num_neighbors),
                                                validation_fold_vec)
        error_mat[, num_neighbors] <- error
    }
    mean_error_vec <- colMeans(error_mat)
    best_neighbors <- min(mean_error_vec)

    best_pred <- class::knn(x_mat, y_vec, k = best_neighbors)
    best_x_new <- best_pred %*% x_new

    results_list <- list(best_x_new, mean_error_vec, error_mat)
    names(results_list) <- c("best", "mean_error_vec", "error_mat")

    return(results_list)
}


spam_datatable <- data.table::fread("spam.data")

x <- spam_datatable[, -58]
x_scale <- scale(x)

y <- spam_datatable[, 58]

result <- nearest_neighbors_cv(x_scale, y, vector(length = ncol(x)))
ggplot() +
  geom_line(aes(
    neighbors, error.percent, group = validation.fold),
    data = result$mean_error_vec)