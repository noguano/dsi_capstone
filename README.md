# DSI US-5 ATX capstone project

# Problem Statement
Telstra, the largest telecommunications company in Australia,  is on a journey to enhance the customer experience - ensuring everyone in the company is putting customers first. In terms of its expansive network, this means continuously advancing how it predicts the scope and timing of service disruptions. Telstra wants to see how you would help it drive customer advocacy by developing a more advanced predictive model for service disruptions and to help it better serve its customers.

# Overview
In this Kaggle competition, the challenge is to predict Telstra network's fault severity at a particular location and time based on the log data available. Each row in the main dataset (train.csv, test.csv) represents a location and a time point. Location and time are coded into the "id" column, which is the key "id" used in other data files. 

Fault severity has 3 categories: 0,1,2 (0 meaning low, 1 meaning medium, and 2 meaning high). This is normally set by each network element in the system, and polled via syslogd on a central server. I guess Telstra hadn't heard of splunk yet.

Different types of features are extracted from log files and other sources: event_type.csv, log_feature.csv, resource_type.csv, severity_type.csv. 

Note: “severity_type” is a feature extracted from the log files (in severity_type.csv). Often this is a severity type of a warning message coming from the log. "severity_type" is categorical. It does not have an ordering. Whereas, “fault_severity” is a measurement of actual reported faults from network elements and is our target variable (in train.csv and test.csv).

# Methods and Models

For this project, I intend to conduct a cross-validated grid search using my favorite classifier - XGBClassifier.

# Success Criteria

Submissions are evaluated using the multi-class logarithmic loss (log_loss). Each data row has been labeled with one true class. For each row, we must submit a set of predicted probabilities (one for every fault severity). The formula is then,

$$logloss=−1N∑i=1N∑j=1Myijlog(pij)$$,

where N is the number of rows in the test set, M is the number of fault severity classes,  log is the natural logarithm, yij is 1 if observation i belongs to class j and 0 otherwise, and pij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with max(min(p,1−10−15),10−15).

Formatting the submission file presented several challenges. Most significantly, separating the classes into three columns meant that the predictions had to be created in three vectors. 

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

Note: I had two corrupted data files when I downloaded the kaggle zip of all files. That set me back several hours as I dug through my bode to find the bug. Then, as a last resort, I decided to diff the original files with individual files I downloaded from kaggle. The moral of this story is don't trust that kaggle has clean or even non-corrupt files. I believe that is a bug, as there was no pedagogic reason for the corrupt files (other than, perhaps, to annoy me, but that really doesn't teach a good lesson, does it?).

# Features

One of the first things I noticed in the provided data is the lack of equipment and connection information. In my experience, the equipment used to provide service (RAN, router, switch, shelf, blade, cross-connect, transceiver, etc.) had the highest probability of causing service interruptions, followed closely by the software or firmware operating that equipment. Another huge factor is human (operator) error (fat fingers, mis-reading procedures, not thoroughly testing before deploying, etc). However, in a public competition the service provider doesn't want to tip consumers over the edge by admitting how their equipment, software, and employees cause the vast majority of outages. Instead, we are left to sift through the haystacks of data to find other minutiae, such as:
1.	Location – location data could point to factors I mentioned above, so this my be our best bet
2.	Weather – harsh weather conditions often impact service, especially metallic loop services
3.	Time (of day, month, year) – temporal data gives us an indicator of service usage for peak hours
4.	Maintenance – planned service could have more impact than intended
5.	Power failure – although rare in telecommunications, power outages do happen
6.	Natural disaster – line breaks and flooding impact service
7.	Demand fluctuation – certain areas are more susceptible to interruptions based on fluctuations in demand
Several of these factors share a linear relationship, for example, a hurricane (or typhoon) may cause flooding that causes a power outage and a spike in demand. Another example is a music festival, where tens of thousands of consumers try to simultaneously connect while the service provider has not scaled equipment to meet that level of demand. 

What we are left with, in this competition, are four tables of encoded numbers, keyed to our train and test sets. The game is afoot!

# Exploratory Data analysis

Since we are tasked with determining the probabilities of three outcomes, this is a multi-classification problem. The classes are very imbalanced, so splitting them up makes sense. We'll have to reverse engineer all data in the supporting files, as we are not given any pertinent information about them. This is where having a decent amount of domain experience comes in handy. In the notebood, I describe the step by step process.

# Feature Engineering

Probably the most challenging, and certainly the most time consuming, part of this competition was creating features from the reverse engineered data. I found a sequential relationship in the location variable, which indicated a temporal encoding. Also, I used a tried and true method of creating features by aggregating less important features, and summarizing other features that had a categorical effect on the target. As can be seen in my notebook, some features only required the top 20 or 40 by count. Scaling was also challenging in this competition because there were two sets of features that required scaling. The first set required standard, Gaussian, scaling, whereas, the second set required robust scaling by percentile to keep the logarithmic scale intact. 

# Modeling

The model I chose for this competition is the XGBClassifier, which I have used in other competitions to great effect. Using a grid search with cross validation, I was able to get the best estimators with the best parameters to get the best log_loss score. 
