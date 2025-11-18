# Load local preprocessing script
source('C:/Users/ktchi/OneDrive/Desktop/pstat197/module-2-claims-data-table-4-1/scripts/preprocessing.R')

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))

# Libraries
library(tidyverse)
library(tidymodels)
library(tidytext)
library(Matrix)
library(glmnet)
library(tokenizers)
library(textstem)
library(modelr)
library(qdapRegex)
library(rvest)
library(stopwords)
library(sparsesvd)

# Load raw data
load("C:/Users/ktchi/OneDrive/Desktop/pstat197/module-2-claims-data-table-4-1/data/claims-clean-example.RData")

set.seed(11152025)

unigram_dtm <- nlp_fn(claims_clean)

unigram_dtm

word_split <- unigram_dtm %>% initial_split(prop = 0.8)

word_split

train_dtm <- training(word_split) %>% select(-.id, -bclass)
train_labels <- training(word_split) %>% select(.id, bclass)

test_dtm <- testing(word_split) %>% select(-.id, -bclass)
test_labels <- testing(word_split) %>% select(.id, bclass)

# PCA on unigram DTM
proj_words <- projection_fn(train_dtm, .prop = 0.8)
train_words_pc <- proj_words$data
test_words_pc <- reproject_fn(test_dtm, proj_words)

# Elastic net logistic regression
x_train_words <- as.matrix(train_words_pc)
y_train_words <- train_labels$bclass

alpha_enet <- 0.3

fit_words <- glmnet(
  x = x_train_words,
  y = y_train_words,
  family = "binomial",
  alpha = alpha_enet
)

cv_words <- cv.glmnet(
  x = x_train_words,
  y = y_train_words,
  family = "binomial",
  alpha = alpha_enet
)

lambda_words <- cv_words$lambda.min

# Extract LOG-ODDS 
logodds_words_train <- as.numeric(
  predict(fit_words, newx = x_train_words, s = lambda_words, type = "link")
)

logodds_words_test <- as.numeric(
  predict(fit_words, newx = as.matrix(test_words_pc), s = lambda_words, type = "link")
)

# Bigram tokenizer 
nlp_fn_bigram <- function(parsed_df) {
  
  # 1. Tokenize into words with order preserved
  tokens <- parsed_df %>%
    unnest_tokens(
      output = token,
      input = text_clean,
      token = "words"
    ) %>%
    mutate(token_lem = lemmatize_words(token)) %>%
    filter(str_length(token_lem) > 2)
  
  # 2. Recombine token sequence per document in correct order
  text_seq <- tokens %>%
    group_by(.id, bclass) %>%
    summarise(text_lem = str_c(token_lem, collapse = " "), .groups = "drop")
  
  # 3. Bigram creation
  bigram_df <- text_seq %>%
    unnest_tokens(
      output = bigram,
      input = text_lem,
      token = "ngrams",
      n = 2
    ) %>%
    separate(bigram, c("w1", "w2"), sep = " ", remove = FALSE) %>%
    filter(!w1 %in% stop_words$word, !w2 %in% stop_words$word) %>%
    select(.id, bclass, bigram)
  
  # 4. TF-IDF bigram DTM
  bigram_dtm <- bigram_df %>%
    count(.id, bclass, bigram) %>%
    bind_tf_idf(term = bigram, document = .id, n = n) %>%
    pivot_wider(
      id_cols = c(.id, bclass),
      names_from = bigram,
      values_from = tf_idf,
      values_fill = 0
    )
  
  return(bigram_dtm)
}



# Create full bigram DTM 
bigram_full <- nlp_fn_bigram(claims_clean)

# Split by IDs
train_ids <- train_labels$.id
test_ids <- test_labels$.id

train_bigram <- bigram_full %>% filter(.id %in% train_ids) %>% arrange(factor(.id, levels=train_ids))
test_bigram  <- bigram_full %>% filter(.id %in% test_ids) %>% arrange(factor(.id, levels=test_ids))

# Remove columns for PCA
train_bigram_mat <- train_bigram %>% select(-.id, -bclass)
test_bigram_mat  <- test_bigram %>% select(-.id, -bclass)

# PCA on bigram DTM 
proj_bigram <- projection_fn(train_bigram_mat, .prop = 0.8)
train_bigram_pc <- proj_bigram$data
test_bigram_pc <- reproject_fn(test_bigram_mat, proj_bigram)

# final combined model
train_combined <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(
    word_logodds = logodds_words_train,
    train_bigram_pc
  )

x_train_comb <- train_combined %>% select(-bclass) %>% as.matrix()
y_train_comb <- train_combined$bclass

# Fit combined elastic net 
fit_combined <- glmnet(
  x = x_train_comb,
  y = y_train_comb,
  family = "binomial",
  alpha = alpha_enet
)

cv_comb <- cv.glmnet(
  x = x_train_comb,
  y = y_train_comb,
  family = "binomial",
  alpha = alpha_enet
)

lambda_comb <- cv_comb$lambda.min

lambda_comb
