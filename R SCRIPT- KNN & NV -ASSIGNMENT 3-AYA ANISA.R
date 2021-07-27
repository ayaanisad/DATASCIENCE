
############################### KNN METHOD  ######################################

if (!require("pacman")) install.packages("pacman") 
p_load(rsample, dplyr, caTools, caret, e1071, FNN) 


dim(attrition)

# Subsetted dataframe 
attrition.df <- attrition[,c("Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany","YearsInCurrentRole", 
                             "YearsSinceLastPromotion", "YearsWithCurrManager", "Attrition")] 

### scale it so that different variables can be brought to the same scale
# get all the input variables in one dataframe X
X <- attrition.df[,1:9]   # selecting all input variables 

# Normalize the inputs 
norm.values <- preProcess(X, method=c("center", "scale")) 
X.norm <- predict(norm.values, X) # Normalized input 

y <- attrition.df$Attrition # target variables

set.seed(101)

# train test split for  prediction accuracy
sample = sample.split(attrition.df$Attrition, SplitRatio = 0.80)# random sampel 80% 
X_train = subset(X.norm, sample==TRUE) # input training
X_test = subset(X.norm, sample==FALSE) # input  prediction accuracy

y_train = subset(y,sample==TRUE) # training
y_test = subset(y, sample==FALSE)# prediction accuracy

#  k =3 
nn <- knn(train = X_train, test=X_test, cl = y_train, k=3)
summary(nn)
confusionMatrix(nn, y_test)$overall[1]

confusionMatrix(nn, y_test) # evaluate other metrics

# check accuracy as various values of K between 1 and 20. 

# define a dataframe to save accuracy for different values of K
accuracy.df <- data.frame(k = seq(1, 20, 1), accuracy = rep(0, 20))
accuracy.df 
# compute knn for different k on validation by looping 
for(i in 1:20) { # we will loop through K= 1 to 20
  knn.pred <- knn(train = X_train, test=X_test, cl = y_train, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, y_test)$overall[1]
}

plot(accuracy.df) # plot accuracy for different values of K 
lines(accuracy.df)

which.max(accuracy.df$accuracy) # optimal K
# K = 20
# Evaluate model at K =20
accuracy.df[20,]


################################################---END---##########################################################


############################################ NAIVE BAYES METHOD  ######################################

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


