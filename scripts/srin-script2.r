## experimenting with models

library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(e1071)
library(glmnet)

# load preprocessing (use existing functions from given script)
source('scripts/preprocessing.R')

# load data
load('data/claims-raw.RData')
cat("Binary classes:", levels(claims_raw$bclass), "\n")
cat("Multi classes:", levels(claims_raw$mclass), "\n")
cat("Total samples:", nrow(claims_raw), "\n\n")

claims_clean <- claims_raw %>% parse_data()
save(claims_clean, file = 'data/claims-clean-task4.RData')

# split data
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8, strata = bclass)

train_df <- training(partitions)
test_df <- testing(partitions)
cat("Training samples:", nrow(train_df), "\n")
cat("Test samples:", nrow(test_df), "\n\n")

# tf-idf features

train_features <- train_df %>% 
  select(.id, text_clean, bclass, mclass) %>%
  nlp_fn()

test_features <- test_df %>% 
  select(.id, text_clean, bclass, mclass) %>%
  nlp_fn()

common_cols <- intersect(names(train_features), names(test_features))
train_features <- train_features %>% select(all_of(common_cols))
test_features <- test_features %>% select(all_of(common_cols))

missing_cols <- setdiff(names(train_features), names(test_features))
for(col in missing_cols) {
  test_features[[col]] <- 0
}
test_features <- test_features %>% select(all_of(names(train_features)))

cols_to_remove <- intersect(c(".id", "bclass", "mclass"), names(train_features))
X_train <- train_features %>% select(-all_of(cols_to_remove)) %>% as.matrix()
X_test <- test_features %>% select(-all_of(cols_to_remove)) %>% as.matrix()
y_train_binary <- train_features$bclass
y_test_binary <- test_features$bclass

train_df_aligned <- train_df %>% 
  select(.id, mclass) %>%
  right_join(train_features %>% select(.id), by = ".id")
test_df_aligned <- test_df %>% 
  select(.id, mclass) %>%
  right_join(test_features %>% select(.id), by = ".id")

y_train_multi <- train_df_aligned$mclass
y_test_multi <- test_df_aligned$mclass

cat("Aligned training samples:", nrow(X_train), "\n")
cat("Aligned test samples:", nrow(X_test), "\n\n")

# svm model

# binary SVM
cat("Training Binary SVM...\n")
svm_binary <- svm(x = X_train, 
                  y = y_train_binary,
                  kernel = "radial",
                  cost = 1)

svm_binary_preds <- predict(svm_binary, X_test)
svm_binary_acc <- mean(svm_binary_preds == y_test_binary)
cat("Binary SVM Accuracy:", round(svm_binary_acc, 4), "\n")

# multiclass SVM
cat("Training Multiclass SVM...\n")
svm_multi <- svm(x = X_train, 
                 y = y_train_multi,
                 kernel = "radial",
                 cost = 1)

svm_multi_preds <- predict(svm_multi, X_test)
svm_multi_acc <- mean(svm_multi_preds == y_test_multi)
cat("Multiclass SVM Accuracy:", round(svm_multi_acc, 4), "\n")

# regularized logistic regression

# binary - ridge regression
cat("Training Binary Logistic Regression (Ridge)...\n")
y_lr_train_binary <- as.numeric(y_train_binary) - 1
y_lr_test_binary <- as.numeric(y_test_binary) - 1

cv_binary <- cv.glmnet(X_train, y_lr_train_binary, 
                       family = "binomial", 
                       alpha = 0,  # Ridge
                       nfolds = 5)

lr_binary_probs <- predict(cv_binary, X_test, 
                           s = "lambda.min", 
                           type = "response")
lr_binary_preds <- ifelse(lr_binary_probs > 0.5, 
                          levels(y_train_binary)[2], 
                          levels(y_train_binary)[1])
lr_binary_acc <- mean(lr_binary_preds == y_test_binary)
cat("Binary Logistic Regression Accuracy:", round(lr_binary_acc, 4), "\n")

# multiclass - multinomial
cat("Training Multiclass Logistic Regression (Ridge)...\n")

cv_multi <- cv.glmnet(X_train, y_train_multi, 
                      family = "multinomial",
                      alpha = 0,
                      nfolds = 5)

lr_multi_preds <- predict(cv_multi, X_test, 
                          s = "lambda.min", 
                          type = "class")
lr_multi_acc <- mean(lr_multi_preds == y_test_multi)
cat("Multiclass Logistic Regression Accuracy:", round(lr_multi_acc, 4), "\n")

# RNN
train_text_aligned <- train_df %>% 
  right_join(train_features %>% select(.id), by = ".id") %>%
  pull(text_clean)

test_text_aligned <- test_df %>% 
  right_join(test_features %>% select(.id), by = ".id") %>%
  pull(text_clean)

# binary labels
rnn_train_labels_binary <- as.numeric(y_train_binary) - 1
rnn_test_labels_binary <- as.numeric(y_test_binary) - 1

# multi labels
rnn_train_labels_multi <- as.numeric(y_train_multi) - 1
rnn_test_labels_multi <- as.numeric(y_test_multi) - 1
n_classes <- length(unique(rnn_train_labels_multi))

# vectorization
max_tokens <- 5000
sequence_length <- 200

text_vectorization <- layer_text_vectorization(
  max_tokens = max_tokens,
  output_mode = "int",
  output_sequence_length = sequence_length
)
text_vectorization %>% adapt(train_text_aligned)

# binary RNN
cat("Training Binary RNN...\n")
rnn_binary <- keras_model_sequential() %>%
  text_vectorization() %>%
  layer_embedding(input_dim = max_tokens, output_dim = 64) %>%
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(1, activation = 'sigmoid')

rnn_binary %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('accuracy')
)

history_binary <- rnn_binary %>%
  fit(train_text_aligned, rnn_train_labels_binary,
      epochs = 10,
      batch_size = 32,
      validation_split = 0.2,
      verbose = 1)

rnn_binary_probs <- predict(rnn_binary, test_text_aligned) %>% as.numeric()
rnn_binary_preds <- (rnn_binary_probs > 0.5) %>% as.integer()
rnn_binary_acc <- mean(rnn_binary_preds == rnn_test_labels_binary)
cat("Binary RNN Accuracy:", round(rnn_binary_acc, 4), "\n")

# multiclass RNN
cat("Training Multiclass RNN...\n")
rnn_multi <- keras_model_sequential() %>%
  text_vectorization() %>%
  layer_embedding(input_dim = max_tokens, output_dim = 64) %>%
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(n_classes, activation = 'softmax')

rnn_multi %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('accuracy')
)

history_multi <- rnn_multi %>%
  fit(train_text_aligned, rnn_train_labels_multi,
      epochs = 10,
      batch_size = 32,
      validation_split = 0.2,
      verbose = 1)

rnn_multi_probs <- predict(rnn_multi, test_text_aligned)
rnn_multi_preds <- apply(rnn_multi_probs, 1, which.max) - 1
rnn_multi_acc <- mean(rnn_multi_preds == rnn_test_labels_multi)
cat("Multiclass RNN Accuracy:", round(rnn_multi_acc, 4), "\n")

## results

results <- tibble(
  Model = c("SVM", "Regularized Logistic Regression", "RNN"),
  Binary_Accuracy = c(svm_binary_acc, lr_binary_acc, rnn_binary_acc),
  Multiclass_Accuracy = c(svm_multi_acc, lr_multi_acc, rnn_multi_acc)
)
print(results)

best_binary <- results$Model[which.max(results$Binary_Accuracy)]
best_multi <- results$Model[which.max(results$Multiclass_Accuracy)]

cat("\nBest Binary Model:", best_binary, 
    "(", round(max(results$Binary_Accuracy), 4), ")\n")
cat("Best Multiclass Model:", best_multi, 
    "(", round(max(results$Multiclass_Accuracy), 4), ")\n")

# best binary model: regularized logistic regression (0.7718)
# best multiclass model: RNN (0.6026)

cat("\nSaving models...\n")
save(cv_binary, cv_multi, file = "results/models-lr.RData")
rnn_multi$save("results/model-rnn-multi.keras")
save(common_cols, file = "results/feature-info.RData")


