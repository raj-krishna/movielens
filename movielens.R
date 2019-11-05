##################################
# Create edx set, validation set #
##################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

dim(edx)

set.seed(1, sample.kind="Rounding")

# creating additional train and test set from edx data
# obtaining test index with 20% of the edx data
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
temp <- edx[test_index, ]
train_set <- edx[-test_index, ]

# Ensuring movieId and userId in train set are also in test set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into test set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(temp, removed, test_index)

RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings) ^ 2))
}
#1. Naive Model: RMSE using average movie rating

# Calculate the average
mu <- mean(train_set$rating)

# Evaluate the performance
naive_rmse <- RMSE(test_set$rating, mu)

# Store the result
rmse_results <- data_frame(Method = "Naive model", RMSE = round(naive_rmse,7), Improvement = "NA")
rmse_results

#2. Movie Effect Model: Modeling movie effect (b_i)

# Calculate the movie effect (b_i)
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

# Obtain the prediction of movie ratings
me_rating <- mu + test_set %>%
  left_join(movie_avgs, by = "movieId") %>% # Combining movie averages with the test set
  pull(b_i)

# Evaluate the performance
model_1_rmse <- RMSE(test_set$rating, me_rating)

# Calculate the improvement in percentage
improvement <- round((naive_rmse - model_1_rmse) * 100 / naive_rmse, 4)

# Append the result for visual comparison
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method = "Movie effect model", 
                                     RMSE = round(model_1_rmse, 7),
                                     Improvement = as.character(improvement)))
rmse_results

#3. Movie and user effect model: Modeling movie effect and user effect (b_i + b_u)

# Calculate the user effect (b_u)
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%  # Combining movie averages with the train set
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# Obtain the prediction of movie ratings
ue_rating <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>% # Combining movie averages with the test set
  left_join(user_avgs, by = "userId") %>% # Combining user averages with the test set
  mutate(uer = mu + b_i + b_u)%>%
  pull(uer)

# Evaluate the performance
model_2_rmse <- RMSE(test_set$rating, ue_rating)

# Calculate the improvement in percentage
improvement <- round((model_1_rmse - model_2_rmse) * 100 / model_1_rmse, 4)

# Append the result for visual comparison
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method = "Movie + User effect model",              
                                     RMSE = round(model_2_rmse, 7),
                                     Improvement = as.character(improvement)))

rmse_results

#4. Movie, user and time effect model: Modeling movie effect, user effect, 
# and time effect (b_i + b_u + b_t)

# Mutate test set to create a date column which would be used for the join 
test_set <- test_set %>%
  mutate(date = lubridate::round_date(lubridate::as_datetime(timestamp), unit = "week"))

# Calculate the time effect (b_t)
time_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%  # Combining movie averages with the train set
  left_join(user_avgs, by = "userId") %>% # Combining user averages with the train set
  mutate(date = lubridate::round_date(lubridate::as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarise(b_t = mean(rating - mu - b_i - b_u))

# Obtain the prediction of movie ratings
te_rating <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%  # Combining movie averages with the test set
  left_join(user_avgs, by = "userId") %>% # Combining user averages with the test set
  left_join(time_avgs, by = "date") %>%
  mutate(ter = mu + b_i + b_u + b_t)%>%
  pull(ter)

# Evaluate the performance
model_3_rmse <- RMSE(test_set$rating, te_rating)

# Calculate the improvement in percentage
improvement = round((model_2_rmse - model_3_rmse) * 100 / model_2_rmse, 4)

# Append the result for visual comparison
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method = "Movie + User + Time effect model",
                                     RMSE = round(model_3_rmse, 7),
                                     Improvement = as.character(improvement)))
rmse_results

#5. Modeling movie effect, user effect and genre effect (b_i + b_u + b_g)

# Separate the each pipe-delimited genre into it's own row
genre_train_set <- train_set %>%
  separate_rows(genres, sep = "\\|")

genre_test_set <- test_set %>%
  separate_rows(genres, sep = "\\|")

# Calculate the genre effect (b_g)
genre_avgs <- genre_train_set %>%
  # Combining movie averages with the genre train set
  left_join(movie_avgs, by = "movieId") %>%  
  # Combining user averages with the genre train set
  left_join(user_avgs, by = "userId") %>% 
  group_by(genres) %>%
  summarise(b_g = mean(rating - mu - b_i - b_u))

# Obtain the prediction of movie ratings
ge_rating <- genre_test_set %>%
  left_join(movie_avgs, by = "movieId") %>% # Combining movie averages with the genre test set
  left_join(user_avgs, by = "userId") %>% # Combining user averages with the genre test set
  left_join(genre_avgs, by ="genres") %>% # Combining genre averages with the genre test set
  mutate(ger = mu + b_i + b_u + b_g)%>%
  pull(ger)

# Evaluate the performance
model_4_rmse <- RMSE(genre_test_set$rating, ge_rating)

# Calculate the improvement in percentage
improvement <- round((model_2_rmse - model_4_rmse) * 100 / model_2_rmse, 4)

# Append the result for visual comparison
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method = "Movie + User + Genre effect model", 
                                     RMSE = round(model_4_rmse, 7),
                                     Improvement = as.character(improvement)))

knitr::kable(rmse_results)


#6. Penalized Least Square

# Create a set of possible lambdas 
lambdas <- seq(0, 10, 0.25)
lambdas

mu <- mean(genre_train_set$rating)

# Using cross validation to obtain the lambda that minimizes the RMSE
# Calculate the RMSE for each of the effects or each lambda
rmses <- sapply(lambdas, function(lambda){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))
  
  b_g <- genre_train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda))
  
  # Predict the ratings based on the penalized effects
  predicted_ratings <- genre_test_set %>%
    left_join(b_i, by = "movieId") %>% # Combining movie averages with genre test set
    left_join(b_u, by = "userId") %>% # Combining user averages with the genre test set
    left_join(b_g, by ="genres") %>%
    mutate(ger = mu + b_i + b_u + b_g)%>%
    pull(ger)
  
  return(RMSE(predicted_ratings, genre_test_set$rating))
  
})


# Determine which value of lambda provides the minimum RMSE
lambda <- lambdas[which.min(rmses)]
lambda


# Calculate the improvement in percentage
improvement <- round((model_4_rmse - min(rmses)) * 100 / model_4_rmse, 4)

# Append the result for visual comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Movie + User + Genre Effect Model",  
                                     RMSE = round(min(rmses), 7),
                                     Improvement = as.character(improvement)))


rmse_results %>% knitr::kable()

ggplot(data_frame(lambdas, rmses)) +
  geom_point(aes(lambdas, rmses), color = "red") +
  labs(title = "Penalized Least Squares", 
       subtitle = "Lambda that minimizes RMSE") +
  theme(plot.title = element_text(color = "red", face = "bold", hjust = 0.5),
        plot.subtitle = element_text(color = "blue", size = 10, hjust = 0.5)) +
  gghighlight::gghighlight(lambdas == lambdas[which.min(rmses)]) +
  geom_segment(x = -1, xend = lambdas[which.min(rmses)], 
               y = min(rmses), yend = min(rmses), 
               color = "red", lty = 2) +
  geom_segment(x = lambdas[which.min(rmses)], xend = lambdas[which.min(rmses)], 
               y = 0, yend = min(rmses), 
               color = "red", lty = 2) +
  ggrepel::geom_label_repel(aes(lambdas, rmses), label = lambda, 
                            fill = "red", color = "white", 
                            fontface = "bold", hjust = 1, vjust = 1)



## Final Results
# Separate the each pipe-delimited genre into it's own row

genre_validation <- validation %>%
  separate_rows(genres, sep = "\\|")

genre_edx <- edx %>%
  separate_rows(genres, sep = "\\|")


# Using the full dataset to identify the lambda for the final model
lambdas <- seq(0, 10, 0.25)
lambdas


# Calculating the various penalized effects based on the lambda that minimizes the RMSE
# on the full edx dataset

rmses <- sapply(lambdas, function(lambda){
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))
  
  b_g <- genre_edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda))
  
  # Obtain the prediction of movie ratings
  predicted_ratings <- genre_validation %>%
    left_join(b_i, by = "movieId") %>% # Combining movie averages with the validation set
    left_join(b_u, by = "userId") %>% # Combining user averages with the validation set
    left_join(b_g, by ="genres") %>% # Combining genre averages with the validation set
    mutate(ger = mu + b_i + b_u + b_g)%>%
    pull(ger)
  
  return(RMSE(predicted_ratings, genre_validation$rating))
})

# Determine which value of lambda provides the minimum RMSE
lambda <- lambdas[which.min(rmses)]
lambda

ggplot(data_frame(lambdas, rmses)) +
  geom_point(aes(lambdas, rmses), color = "red") +
  labs(title = "Penalized Least Squares", 
       subtitle = "Lambda that minimizes RMSE") +
  theme(plot.title = element_text(color = "red", face = "bold", hjust = 0.5),
        plot.subtitle = element_text(color = "blue", size = 10, hjust = 0.5)) +
  gghighlight::gghighlight(lambdas == lambdas[which.min(rmses)]) +
  geom_segment(x = -1, xend = lambdas[which.min(rmses)], 
               y = min(rmses), yend = min(rmses), 
               color = "red", lty = 2) +
  geom_segment(x = lambdas[which.min(rmses)], xend = lambdas[which.min(rmses)], 
               y = 0, yend = min(rmses), 
               color = "red", lty = 2) +
  ggrepel::geom_label_repel(aes(lambdas, rmses), label = lambda, 
                            fill = "red", color = "white", 
                            fontface = "bold", hjust = 1, vjust = 1)

# Display the final RMSE
final_rmse <- data_frame(Method = "Regularized Movie + User + Genre Effect", 
                         RMSE = round(min(rmses),7), lambda = lambda)

final_rmse %>% knitr::kable()
