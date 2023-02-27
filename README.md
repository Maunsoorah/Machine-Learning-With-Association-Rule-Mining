# Machine-Learning-With-Association-Rule-Mining 
 Using Association rule mining (Apriori Algorithm) to  predict how Congress men will vote for  various agenda using the Congressional Voting records dataset from the UCI Data Repository

### INTRODUCTION 
Association rule mining is a widely used technique for discovering interesting relationships and patterns in large datasets. According to Wikipedia.org, Association rule mining is a rule-based machine learning method for discovering interesting relations between variables in large databases. At its most basic level, association rule mining uses machine learning models to search through datasets for patterns or co-occurrences in the data. It detects common if-then relationships, which are the association rules themselves. These rules are created by searching through the data for frequent if-then patterns and using the criteria support and confidence to identify the most important relationships.

In political science, association rule mining can be used to gain insights into the factors that influence political decision-making and the ways in which different issues are interconnected. 

The Congressional Voting records dataset from the UCI Data Repository contains voting records of 435 members of the US House of Representatives on 16 different issues, ranging from education to defense. In this study, we applied the Apriori algorithm, a popular algorithm for association rule mining, to the Congressional Voting records dataset to identify interesting patterns and relationships among the voting records. 

The goal of this analysis is to provide valuable insights into the complex political decision-making processes in the US House of Representatives and to help policymakers, journalists, and citizens to better understand the factors that shape public policy. The Apriori algorithm will allow us to identify frequent itemsets, or combinations of issues that occur together frequently, and to discover strong associations between different issues. These results can help us to understand the factors that drive voting decisions and to predict future voting patterns

### The Dataset 
The Congressional Voting records dataset can be found on the UCI repository with the link http://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/housevotes-84.data. The dataset contains 435 voting records on 16 different issues, ranging from education to defense. We will be using association mining to check which combination of attributes has the highest confidence with either republican or democrat. Below is a table that shows the list of attributes in the dataset and their corresponding description. 
  
### EXPLORATION AND PREPARATION OF DATASET 
We will start by installing necessary packages and importing libraries. 
  ![image](https://user-images.githubusercontent.com/114883368/221557289-ea5492ae-1e03-4e75-b9c6-074dd3ec7fd1.png)

Import the needed libraries. 
 ![image](https://user-images.githubusercontent.com/114883368/221557354-e0a2e645-ff05-4719-b6cd-f446b882d6e3.png)
 

We will now import our data set from the UCI reposirory using the link http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records 
Our column names were not assigned at the source, so we will add the name=[] function to assign names to each column. This is done as the data is being imported. 
  ![image](https://user-images.githubusercontent.com/114883368/221557427-add264ca-2d6a-491d-be14-c1be32507005.png)

Lets take a look at the first few columns of our dataset to see the columns and a few of the values in our data. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221557464-f005651f-d3fd-4e1e-a053-b437d79aecdc.png)

From the above we can see that we have some missing data which have been represented with the value ‘?, this value will not be seen by python as a missing value so we need to reassign it a numpy null value so we will be able to view the entire missing values for the data set. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221557531-6a74da63-0173-4cca-af7a-d46cd08dbc5c.png)

After the missing values have been assigned with the NaN for python to recognise it as missing values, we will then use the isnull() function to view at a glance the total numbers of missing values in each column. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221557599-6bac1e33-4d49-4a1b-a900-905e3cd6cadc.png)

We can now see that some of the columns have quite a number of missing values. 
We will be removing the first two columns due to having the highest number and by randomly sampling the dataset, I have decided to replace the ones in the other columns with the value ‘y’.  

![image](https://user-images.githubusercontent.com/114883368/221559870-cdce2634-1b85-4f9a-9077-68a16e81bf7a.png)
![image](https://user-images.githubusercontent.com/114883368/221559893-ce42f266-7d46-451f-a100-c16a3e0a0160.png)
![image](https://user-images.githubusercontent.com/114883368/221559920-f1daba0c-b778-408b-bc7d-fc6f99082f15.png)


After we have dealt with the missing values and our dataset is now clean and ready to work with, we will notice that the column ‘class name’ has two variables republican and democrat which indicates the party the voters belong. We need to convert the data in that column to y and n (yes and no) just like the other columns in our data to enable us perform our association mining. The following code shows the splitting of this column into two different ones with column names ‘republican’ and ‘democrat’ and these two columns with have the values ‘y’ and ‘n’. 
 
  ![image](https://user-images.githubusercontent.com/114883368/221557825-d86b5dd6-8c90-48c1-a582-6818ce7ab841.png)

After we have created the two new columns, we will no longer need the original column ‘Class Name’ and we will now drop it using the drop() function. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221557906-67224a5a-6023-4d60-a94e-5381c767b688.png)

We will now plot a bar chart to visualise the distribution of variables in all our columns. This will allow us see if there are any columns with huge imbalance. 

![image](https://user-images.githubusercontent.com/114883368/221557995-62dd27d1-86c7-4940-bbb9-0b58e2f4fc52.png)
  
For the purpose of this project, we will now create sample pairs from our data which we will be using to run our Apriori model. These pairs are randomly generated from our dataset. 

![image](https://user-images.githubusercontent.com/114883368/221558061-31c5037e-8c62-4c70-ae57-1f2e1b0c85d1.png)

  
We will now create a list of association rules using apriori by passing the list we called votes. We are setting our min_support value at 0.2 meaning that we want to have items with support value greater that 0.2 in our rules. Our min_confidence has been set at 0.5 so we can filter only rules with confidence level of higher than 0.5. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221558146-cb79a1a2-67c8-4d9b-942f-d0be3dea0fe9.png)

Let’s view the resulting rules. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221558220-098e7959-48e0-4568-9134-f86502578a87.png)
 
The inspect() functions shows us the possible combinations of rules that we have. 
  
  ![image](https://user-images.githubusercontent.com/114883368/221558473-86f5bde4-94ef-4e7a-8b78-a4164324eb7b.png)

We can also use the len() function to see how many rules we have in total without displaying the actual rules. 

![image](https://user-images.githubusercontent.com/114883368/221558505-157c4594-6c00-491d-adc1-4e63459aef6d.png)

  
We will now display the different association rules that we have created viewing the rules with highest Lift, Support and confidence separately as seen below. 
  
 ![image](https://user-images.githubusercontent.com/114883368/221558556-70bc8cbc-e11a-431c-803e-29b9a3b3520f.png)

 
We can see from below that the first few rules blanks for LHS(antecedent) and only have values for RHS(consequent), so for this reason we will ignore them and only consider the rules that have values for both. 

![image](https://user-images.githubusercontent.com/114883368/221559054-bc445af4-3256-4f5c-a762-baeaa36d981a.png)

![image](https://user-images.githubusercontent.com/114883368/221559092-2fa26b37-0ad4-44c6-ae8c-55925996f427.png)
 
We will now use the lambda function to create a match with the parties our voters belong to. First we will create for republican and then for democrat. 
 
 ![image](https://user-images.githubusercontent.com/114883368/221559185-94700dcc-ecc5-44ee-9809-b8741ad23597.png)

![image](https://user-images.githubusercontent.com/114883368/221559224-211d5f24-579e-4701-a945-8330c71b5262.png)

 
## RESULT ANALYSIS AND DISCUSSION 
The Apriori algorithm is a widely used algorithm for association rule mining that was applied to the Congressional Voting records dataset from the UCI Data Repository. The Apriori algorithm was used to identify interesting patterns and relationships among these voting records.

The results of the analysis revealed several interesting associations between the different issues. The algorithm identified several strong associations, such as between the issues of 'physician fee freeze' and 'el salvador aid', and between the issues of 'adoption of the budget resolution' and 'mx missile'. These results suggest that there may be a strong ideological or political connection between these issues, which could be further explored in future studies.

The algorithm also identified some unexpected associations, such as between the issues of 'handicapped infants' and 'water project cost sharing'. This result may suggest that there are hidden factors or political alliances that are driving voting decisions on these seemingly unrelated issues.

The Apriori algorithm also allowed for the identification of frequent itemsets, or combinations of issues that occur together frequently. For example, the algorithm identified that the combination of 'physician fee freeze' and 'el salvador aid' occurred together in 32% of the voting records. This information can be useful for understanding the factors that drive voting decisions and for predicting future voting patterns.

One limitation of the Apriori algorithm is that it can be computationally expensive for large datasets. In this case, the algorithm was able to handle the relatively small Congressional Voting records dataset, but may face challenges with larger datasets.

## CONCLUSION 
In conclusion, the association rule mining algorithm applied on the congressional voting records dataset  was able to identify several interesting patterns and relationships among the different voting issues. The analysis revealed that certain voting issues were strongly associated with each other, while others were not. The algorithm also identified some unexpected relationships between voting issues that could have important implications for understanding political decision-making processes.
Overall, the results of the association mining rule algorithm demonstrates the potential for using data mining techniques to gain insights into complex political systems. By identifying patterns and relationships in large datasets, researchers can better understand the factors that influence political decision-making and the ways in which different issues are interconnected. This information can be valuable for policymakers, journalists and citizens who seek to understand the workings of government and the factors that shape public policy.
 
