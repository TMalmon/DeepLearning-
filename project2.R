
library(data.table)
library(ggplot2)


k_fold_cv <- function(x_mat, y_vec, compute_predictions, fold_vec) {

}


nearest_neighbors_cv <- function(x_mat, y_vec, x_new,
                                 num_folds = 5, max_neighbors = 20) {
    validation_fold_vec <- sample(rep(1:num_folds, l = nrow(x_mat)))
    error_mat <- matrix(num_folds, max_neighbors)
    mean_error_vec <- vector(length = max_neighbors)

    for (i in 1:max_neighbors) {
        error <- k_fold_cv(x_mat, y_vec,
            compute_predictions =
                class::knn(i), validation_fold_vec
        )
        error_mat[, i] <- error
    }
    mean_error_vec <- colMeans(error_mat)
    best_neighbors <- min(mean_error_vec)

}