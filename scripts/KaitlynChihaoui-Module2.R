# Preliminary Task 2
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(tokenizers)
library(textstem)

# source preprocessing script
source('./scripts/preprocessing.R')

# load in claims data
load("C:/Users/ktchi/OneDrive/Desktop/pstat197/module-2-claims-data-table-4-1/data/claims-raw.RData")
View(claims_raw)

# function for bigram tokenization
nlp_fn_bigram <- function(parse_data.out){
  # First, lemmatize all words
  words_lemmatized <- parse_data.out %>%
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2)
  
  # Reconstruct text from lemmatized words for bigram creation
  text_lemmatized <- words_lemmatized %>%
    group_by(.id, bclass) %>%
    summarize(text_lem = str_c(token.lem, collapse = ' '), .groups = 'drop')
  
  # Create bigrams from lemmatized text
  out <- text_lemmatized %>%
    unnest_tokens(output = bigram, 
                  input = text_lem, 
                  token = 'ngrams',
                  n = 2) %>%
    # Remove bigrams containing stopwords
    separate(bigram, c('word1', 'word2'), sep = ' ', remove = FALSE) %>%
    filter(!word1 %in% stop_words$word,
           !word2 %in% stop_words$word) %>%
    select(-word1, -word2) %>%
    # Count and calculate TF-IDF
    count(.id, bclass, bigram, name = 'n') %>%
    bind_tf_idf(term = bigram, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'bigram',
                values_from = 'tf_idf',
                values_fill = 0)
  
  return(out)
}

# Helper function to create bigram DTM from already-split data
create_bigram_dtm <- function(.data, parse_data.out){
  # Get the .id values from the split data
  id_subset <- .data %>% pull(.id)
  
  # Filter parsed data to match the subset
  parsed_subset <- parse_data.out %>%
    filter(.id %in% id_subset)
  
  # Apply bigram tokenization
  bigram_dtm <- nlp_fn_bigram(parsed_subset)
  
  return(bigram_dtm)
}

# split data
set.seed(11152025)
word_split <- claims_raw %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(word_split) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(word_split) %>%
  select(.id, bclass, mclass)

# same, training set
train_dtm <- training(word_split) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(word_split) %>%
  select(.id, bclass, mclass)

# PCA
# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.8)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc

# regression
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

#fit <- glm(bclass ~., data = train, family = "binomial")

# store predictors and response as matrix and vector
x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

# fit enet model
alpha_enet <- 0.3
fit_reg_words <- glmnet(x = x_train, 
                        y = y_train, 
                        family = 'binomial',
                        alpha = alpha_enet)

# choose a strength (penalty/lambda value) by cross-validation
set.seed(11172025)
cvout_words <- cv.glmnet(x = x_train, 
                   y = y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength (penalty)
lambda_opt <- cvout_words$lambda.min

# view results
lambda_opt

# training set log-odds
log_odds_words_train <- predict(fit_reg_words,
                                s = lambda_opt,
                                newx = x_train,
                                type = 'link')

# test-set log odds
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)
x_test <- as.matrix(test_dtm_projected)

log_odds_words_test <- predict(fit_reg_words, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'link')

# create bigram dtm
parsed_data <- parse_data(claims_raw)
claims_bigram <- nlp_fn_bigram(parsed_data)

# Get train/test IDs
train_ids <- training(word_split) %>% pull(.id)
test_ids <- testing(word_split) %>% pull(.id)

# Split bigram data using the same IDs
train_bigram_dtm <- claims_bigram %>%
  filter(.id %in% train_ids) %>%
  select(-.id, -bclass)

test_bigram_dtm <- claims_bigram %>%
  filter(.id %in% test_ids) %>%
  select(-.id, -bclass)

# pca on bigram dtm
proj_out_bigram <- projection_fn(.dtm = train_bigram_dtm, .prop = 0.8)
train_bigram_projected <- proj_out_bigram$data

# project test bigrams
test_bigram_projected <- reproject_fn(.dtm = test_bigram_dtm, proj_out_bigram)

# Combine word log-odds with bigram PCs for training
train_combined <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(
    word_logodds = as.numeric(log_odds_words_train),
    train_bigram_projected
  )

# Fit combined model
x_train_combined <- train_combined %>% select(-bclass) %>% as.matrix()
y_train_combined <- train_combined %>% pull(bclass)

set.seed(11172025)
fit_combined <- glmnet(x = x_train_combined, 
                       y = y_train_combined, 
                       family = 'binomial',
                       alpha = alpha_enet)

# Cross-validation for combined model
cvout_combined <- cv.glmnet(x = x_train_combined, 
                            y = y_train_combined, 
                            family = 'binomial',
                            alpha = alpha_enet)
lambda_opt_combined <- cvout_combined$lambda.min