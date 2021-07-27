
if (!require("pacman")) install.packages("pacman") 
p_load(rsample, dplyr, caTools, caret, e1071)


# Read the data file
real_es <- read.csv("realEstate.csv")

# Define input variables
X = real_es[,2:9]
# Define target variable
y = real_es[,10]


colnames(X)
# "full_sq"    "life_sq"    "floor"      "max_floor"  "material"   "build_year" "num_room"   "kitch_sq"  


# normalizing all input variables 
# Normalize the inputs 
norm.values <- preProcess(X, method=c("center", "scale")) 
X.norm <- predict(norm.values, X) # Normalized input 

# create a function that takes a variable and returns the same variables as "low", "med", "high" 
binning_func <- function(x){ 
  binx <- cut(x, c((min(x)), (mean(x)-1*sd(x)), (mean(x)+1*sd(x)), max(x)), labels=c("Low", "Med", "High"))
  return(binx)
}

# applied the function to all the columns in the X.norm dataframe and saved all of them in X_binned dataframe
X_binned <- data.frame(apply(X.norm, 2, binning_func))



df <- cbind(X_binned, y)

set.seed(101) # random seed for replicating 
# train test split
sample = sample.split(df$y, SplitRatio = 0.80)# random sample of 80%
train <- df[sample==TRUE,] # training dataset
test <-  df[sample==FALSE,] # testing dataset


# Perform Naive Bayes on 8 predictors 
Naive_Bayes_Model=naiveBayes(y ~ ., data=train)
Naive_Bayes_Model$tables

# Accuracy on training set 
train_predictions = predict(Naive_Bayes_Model, train[,-9]) # Removed the ninth column (target variable)
confusionMatrix(train_predictions, train$y)# comparing actual price classes and predicted price classes in the train dataset
# Accuracy - 0.71

# Accuracy on the test set
test_predictions = predict(Naive_Bayes_Model, test[,-9])# removes the ninth column (target variable)
confusionMatrix(test_predictions, test$y)# comparing actual price classes and predicted price classes in the test dataset
# Accuracy - 0.71



#######################################################---END---############################################################

