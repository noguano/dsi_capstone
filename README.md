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

# Features

One of the first things I noticed in the provided data is the lack of equipment and connection information. In my experience, the equipment used to provide service (RAN, router, switch, shelf, blade, cross-connect, transceiver, etc.) had the highest probability of causing service interruptions, followed closely by the software or firmware operating that equipment. Another huge factor is human (operator) error (fat fingers, mis-reading procedures, not thoroughly testing before deploying, etc). However, in a public competition the service provider doesn't want to tip consumers over the edge by admitting how their equipment, software, and employees cause the vast majority of outages. Instead, we are left to sift through the haystacks of data to find other minutiae, such as:
1.	Location – location data could point to factors I mentioned above, so this my be our best bet
2.	Weather – harsh weather conditions often impact service, especially metallic loop services
3.	Time (of day, month, year) – temporal data gives us an indicator of service usage for peak hours
4.	Maintenance – planned service could have more impact than intended
5.	Power failure – although rare in telecommunications, power outages do happen
6.	Natural disaster – line breaks and flooding impact service
7.	Demand fluctuation – certain areas are more susceptible to interruptions based on fluctuations in demand
Several of these factors are share a linear relationship, for example, a hurricane (or typhoon) may cause flooding that causes a power outage and a spike in demand. Another example is a music festival, where tens of thousands of consumers try to simultaneously connect while the service provider has not scaled equipment to meet that level of demand. 

# Exploratory Data analysis

Since we are tasked with determining the probabilities of three outcomes, this is a multi-classification problem. 
