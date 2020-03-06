#libraries
library(data.table)
library(ggplot2)

#functions
n_net_one_split <- function (x_mat, y_vec, step_size, max_epochs,
                            n_hidden_units, is_subtrain) {

    x_sub <- x_mat[is_subtrain]
    x_vali <- x_mat[-is_subtrain]

    y_sub <- y_vec[is_subtrain]
    y_vali <- y_vec[-is_subtrain]
    y_vali_tilde <- ifelse(y_vali == 1, 1, -1)

    weight_mat <- rnorm(matrix(nrow = ncol(x_mat), ncol = n_hidden_units))
    weight_vec <- vector(length = n_hidden_units)

    for( epoch = 1 in 1:max_epochs ) {
        for( obs = 1 in 1:x_sub ) {
            #step in negative gradient for weight_mat and weight_vec
        }

        predictions <- weight_mat %*% x_vali
        loss_values <-  log(1+exp(-y_vali_tilde*predictions))
    }

    return (list("loss" = loss_values, "weight_mat" = weight_mat, "weight_vec" = weight_vec))
}

# start data analysis
#---------------------------------------------------------
spam_datatable <- data.table::fread("spam.data")

x <- spam_datatable[, -58]
x_scale <- scale(x)

y <- spam_datatable[, 58]

is_train <- vector(mode = "logical", length = nrow(x))
is_train <- sample(c(TRUE, FALSE), nrow(x), replace = TRUE, prob = c(0.8, 0.2))


is_subtrain <- vector(mode = "logical", length = ncol(is_train))
is_subtrain <- sample(c(TRUE,FALSE), ncol(is_train), replace = TRUE, prob = c(0.6, 0.4))

max_epoch <- numeric(1000)

best_epochs <- numeric(0)

nNetOneSplit <- n_net_one_split(x_scale, y, .1, max_epoch, 64, is_subtrain)

best_epochs <- n_net_one_split(x_scale, y, .1, best_epochs, 64, is_subtrain = TRUE)



