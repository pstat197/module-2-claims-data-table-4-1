# predict on test data
library(tidyverse)
library(keras)
library(tensorflow)
library(glmnet)

source('scripts/preprocessing.R')

# load test data
load('data/claims-test.RData')
load('data/claims-raw.RData')
binary_labels <- levels(claims_raw$bclass)
multi_labels <- levels(claims_raw$mclass)

# preprocess
test_clean <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean)

# binary predictions: logistic regression
load('results/models-lr.RData')
load('results/feature-info.RData')
test_features <- test_clean %>%
  mutate(bclass = NA, mclass = NA) %>%
  nlp_fn()

for(col in common_cols) {
  if(!col %in% names(test_features)) {
    test_features[[col]] <- 0
  }
}
test_features <- test_features %>% select(all_of(common_cols))

cols_to_remove <- intersect(c(".id", "bclass", "mclass"), names(test_features))
X_test <- test_features %>% select(-all_of(cols_to_remove)) %>% as.matrix()

# predict
lr_binary_probs <- predict(cv_binary, X_test, s = "lambda.min", type = "response")
binary_preds <- factor(ifelse(lr_binary_probs > 0.5, binary_labels[2], binary_labels[1]))

cat("Binary predictions complete!\n")
cat("Distribution:\n")
print(table(binary_preds))

## multiclass prediction: RNN
cat("Loading RNN model...\n")
rnn_multi <- load_model("results/model-rnn-multi.keras")
test_text <- test_clean %>% pull(text_clean)

# predict
multi_probs <- predict(rnn_multi, test_text)
multi_preds_numeric <- apply(multi_probs, 1, which.max)
multi_preds <- factor(multi_preds_numeric, levels = 1:length(multi_labels), labels = multi_labels)

cat("Multiclass predictions complete!\n")
cat("Distribution:\n")
print(table(multi_preds))

binary_df <- test_features %>%
  select(.id) %>%
  bind_cols(bclass.pred = binary_preds)

multi_df <- test_clean %>%
  select(.id) %>%
  bind_cols(mclass.pred = multi_preds)

pred_df <- multi_df %>%
  left_join(binary_df, by = ".id")

missing_binary <- sum(is.na(pred_df$bclass.pred))
if(missing_binary > 0) {
  cat("Warning:", missing_binary, "samples missing binary predictions\n")
  cat("Filling with most common class...\n")
  most_common <- names(sort(table(binary_preds), decreasing = TRUE))[1]
  pred_df <- pred_df %>%
    mutate(bclass.pred = if_else(is.na(bclass.pred), 
                                 factor(most_common, levels = binary_labels), 
                                 bclass.pred))
}

pred_df <- pred_df %>%
  select(.id, bclass.pred, mclass.pred)


cat("Total predictions:", nrow(pred_df), "\n\n")

cat("Binary class distribution:\n")
print(table(pred_df$bclass.pred))

cat("\nMulticlass distribution:\n")
print(table(pred_df$mclass.pred))

cat("\nMissing values check:\n")
cat("Binary NAs:", sum(is.na(pred_df$bclass.pred)), "\n")
cat("Multiclass NAs:", sum(is.na(pred_df$mclass.pred)), "\n")

group_number <- 4
output_file <- paste0('results/preds-group', 4, '.RData')
save(pred_df, file = output_file)
