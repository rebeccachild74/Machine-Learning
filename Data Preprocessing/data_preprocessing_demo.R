# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age) 

# Checks to see if the age is nan. If true, it sets the missing
# values in the age colunm to the average of the rest of the columns,
# including the missing values.if False, set the age accordingly

# Repeat for the salary column

dataset$Salary = ifelse(is.na(dataset$Salary), 
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'), # c is a vector in R
                         labels = c(1, 2, 3)) # France = 1, Spain = 2, Germany = 3

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No', 'Yes'),
                         labels = c(0, 1)) # Yes = 1, No = 2

# Splitting the dataset into the Training and Testing Set
# Installing a library (seein in packages tab)
#install.packages('caTools')

# Setting seed to recieve the same results as demo
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) # Ratio here is portion of training data

training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])