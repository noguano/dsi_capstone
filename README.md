# DSI US-5 ATX capstone project

# Problem Statement
Telstra, the largest telecommunications company in Australia,  is on a journey to enhance the customer experience - ensuring everyone in the company is putting customers first. In terms of its expansive network, this means continuously advancing how it predicts the scope and timing of service disruptions. Telstra wants to see how you would help it drive customer advocacy by developing a more advanced predictive model for service disruptions and to help it better serve its customers.

# Overview
In this Kaggle competition, the challenge is to predict Telstra network's fault severity at a time at a particular location based on the log data available. Each row in the main dataset (train.csv, test.csv) represents a location and a time point. They are identified by the "id" column, which is the key "id" used in other data files. 

Fault severity has 3 categories: 0,1,2 (0 meaning no fault, 1 meaning only a few, and 2 meaning many). 

Different types of features are extracted from log files and other sources: event_type.csv, log_feature.csv, resource_type.csv, severity_type.csv. 

Note: “severity_type” is a feature extracted from the log files (in severity_type.csv). Often this is a severity type of a warning message coming from the log. "severity_type" is categorical. It does not have an ordering. “fault_severity” is a measurement of actual reported faults from users of the network and is the target variable (in train.csv).



# Methods and Models

For this project, I intend to conduct a cross-validated grid search of several classification models; including, ExtraTrees, 

# Risks and Assumptions



# Success Criteria

Submissions are evaluated using the multi-class logarithmic loss. Each data row has been labeled with one true class. For each row, you must submit a set of predicted probabilities (one for every fault severity). The formula is then,

logloss=−1N∑i=1N∑j=1Myijlog(pij),
where N is the number of rows in the test set, M is the number of fault severity classes,  log is the natural logarithm, yij is 1 if observation i belongs to class j and 0 otherwise, and pij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with max(min(p,1−10−15),10−15).

# Data Source

The Kaggle competition provides the following data:

|  Name      | Description | 
|------------|-------------|
| train.csv  | the training set for fault severity |  
| test.csv   | the test set for fault severity |
| sample_submission.csv | a sample submission file in the correct format |
| event_type.csv  | event type related to the main dataset |
| log_feature.csv | features extracted from log files |
| resource_type.csv | type of resource related to the main dataset |
| severity_type.csv | severity type of a warning message coming from the log |

