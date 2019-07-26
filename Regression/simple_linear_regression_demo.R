# Simple Linear Regression 

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set

regressor = lm(formula = Salary ~ YearsExperience,
               training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualize the Training set results
# install.packages('ggplot2')

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') + 
  ylab('Salary')

# Visualize the Testing set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') + 
  ylab('Salary')
