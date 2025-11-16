getwd()
setwd("/Users/alexdieter/Downloads/Courses/197/team assignment_three/module-2-claims-data-table-4-1")
library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(tidymodels)

############################################################
#Copy of preprocessing.R FUNCTIONS
############################################################
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}



#############################################################
# My code
#############################################################
#same function but including headers
parse_fn_headers <- function(html_string){
  
  raw_text <- read_html(html_string) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%   # <-- NEW
    html_text2() %>%
    str_c(collapse = " ")
  
  clean_text <- raw_text %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  
  return(clean_text)
}

parse_data_headers <- function(df){
  df %>%
    filter(str_detect(text_tmp, "<!")) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_headers(text_tmp)) %>%
    unnest(text_clean)
}



load("data/claims-raw.RData")    


claims_clean_baseline <- claims_raw %>%
  parse_data()

claims_clean_headers <- claims_raw %>%
  parse_data_headers()


tfidf_baseline <- nlp_fn(claims_clean_baseline)
tfidf_headers  <- nlp_fn(claims_clean_headers)


run_logistic_pcr <- function(tfidf_df, n_comp = 50){
  set.seed(11172025)
  
  split_obj <- initial_split(tfidf_df, prop = 0.8, strata = bclass)
  train_df  <- training(split_obj)
  test_df   <- testing(split_obj)
  
  rec <- recipe(bclass ~ ., data = train_df) %>%
    step_rm(.id) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_predictors()) %>%
    step_pca(all_predictors(), num_comp = n_comp)
  
  log_spec <- logistic_reg(mode = "classification") %>%
    set_engine("glm")
  
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(log_spec)
  
  wf_fit <- last_fit(wf, split_obj)
  
  acc <- collect_metrics(wf_fit) %>%
    filter(.metric == "accuracy") %>%
    pull(.estimate)
  
  return(acc)
}


acc_baseline <- run_logistic_pcr(tfidf_baseline)
acc_headers  <- run_logistic_pcr(tfidf_headers)

acc_baseline
acc_headers