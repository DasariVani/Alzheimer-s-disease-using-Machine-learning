# Alzheimer-s-disease-using-Machine-learning
Machine learning techniques are applied to analyze diverse data, enabling early detection and improved diagnostic accuracy for Alzheimer's, ultimately aiding in timely interventions.

INTRODUCTION 

We know that drug delivery is crucial in the medical domain, where many scientists and researchers aim to identify the latest therapies that support the treatment of illness. However, the traditional methods of drug discovery are time-consuming, expensive, and require significant human power. Therefore, machine learning technology plays an important role in the medical field too. Machine learning allows researchers and scientists to analyze vast amounts of chemical and biological data quickly and accurately. It can forecast the likelihood of a compound exhibiting biological activity, suggesting its potential effectiveness as a drug. This paper explores how machine learning can be used in the process of a comprehensive strategy for drug discovery, with a certain focus on predicting the bioactivity (pIC50 values) of compounds that inhibit acetylcholinesterase, which is an enzyme associated with neurodegenerative disorders.  In this paper, acetylcholinesterase is the target in the treatment of Alzheimer’s disease.

DATASETS
-“acetylcholinesterase_01_bioactivity_data_raw.csv” is used to for the starting point for data curation. This dataset contains raw bioactivity records from the chEMBL database which includes SMILES strings and IC50 values.
 
- “acetylcholinesterase_03_bioactivity_data_curated.csv”  is the curated dataset where it comes from cleaning and filtering the raw data. The missing and the ambiguous bioactivity values were removed from the data. This data is used for exploratory data analysis and for better visualizations.
 
- “acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv” is the final dataset which is used for training machine learning models. This dataset contains transformed Ic50 to pIC50 values and molecular fingerprints which is in PubChem format. This datasey is used for a focus on training a random forest regressor and to build and evaluate regression models compared various regression models.

DATA PREPROCESSING
Firstly smiles strings were converted into molecular structures by using RDkit. IC50 values were transformed by using logarithmic scales to reduce the skewness and to remove noisy data.
To remove low information features, we implemented a variance threshold filter.

METHODOLOGY
-There are 3 parts to this project. First, we gathered the data for the data preprocessing and later we have done exploratory data analysis. After EDA we built and evaluated different scores of respective models such as random forest, support vector machine, linear regression, XGBoost, K-nearest neighbors. And in the last part we compared the respective results for respective models based on accuracy, time taken and errors such as r-squared and root mean squared errors.
-We made an exploratory data analysis  on the curated data set for the better understanding of data distribution and to identify patterns. IC50 values were found to be right-skewed which leads to their transformation into pIC50 values by using the pIc50 = -log10(IC50 * 10-9). For the data visulalization we designed some count plots for two classed such as active and inactive, scatter plots for molecular weight and LogP(pIc5o values), box plots for pIC50 and bioactivity classes, and box plots for molecular weights and bioactivity classes etc.
-Now we used “acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv” data to build different models such as random forest, support vector machine, linear regression, XGBoost, K-nearest neighbors. After removing low variance features, first partition of the data has been implemented with a ratio of 80/20. 80 percent of the data is for training and the remaining 20 percent is for testing. Now we built random forest, support vector machine, linear regression, XGBoost, K-nearest neighbors models on the data. Later we have got a linearity on experimental pIC50 and predicted pIC50 and made a better visualization by making a scatter plot.
-Based on the “acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv” data, In the last part We made a comparative model analysis by evaluating performance of several regression algorithms. All models were trained and evaluated using the train-test-split partition and the evaluation metrics such as R-squared, and RMSE were used for effective evaluation and bar plots were designed for comparison of models. conclusion and future work
 
RESULTS
 
In the exploratory data analysis part we have obtained some results by designing some count, scatter plots.
 
-The picture shows us the count plot between bioactivity class and frequency and shows us that inactive classes are high compared to active classes. The scatter plot between molecular weights and the logarithmic function of p values (pIC50).
 
-After loading the data set “acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv”, we have cleaned the data we removed the low variance features and partitioned the data with 80/20 ratio into training and testing respectively. Now we built different models such as  random forest, support vector machine, linear regression, XGBoost, K-nearest neighbors.
  
-We built a random forest regression model, and the R-squared score of 0.5385475889129773
 
-By using sklearn library we imported support SVR which is of support vector machine model and the R-squared score for support vector machine is 0.44947965875202056.
 
-R-squared score of linear regression is 0.285937377476286
 
-Here we have implemented XGBoost regressor and it has a score of 0.5402063944178346
 
-Again, by using sklearn library we imported K-nearest neighnors library and it has a score of 0.46784701296299835
 
-The picture shows us the linearity between experimental pIC50 values and predicted pIC50 values.
 
COMPARISON RESULTS
 
Below are the comparison tables of different models.
The tables above provide a comparison of different regression models based on their performance in both training and testing datasets. Each model was assessed using metrics such as Adjusted R-squared, R-squared, RMSE (Root Mean Squared Error), and the time taken for training.
 
For the training set –
 
-DecisionTreeRegressor, ExtraTreeRegressor, and ExtraTreesRegressor, all attaining an R-squared of 0.86 and a low RMSE of 0.57, indicating a strong fit to the training data. The training duration varied between 0.24 and 13.59 seconds.
 
-GaussianProcessRegressor also reached an R-squared of 0.86 and RMSE of 0.57 but required 11.32 seconds for training.
 
-Other competitive models featured included RandomForestRegressor (R² = 0.83, RMSE = 0.64), XGBRegressor (R² = 0.83, RMSE = 0.65), and BaggingRegressor (R² = 0.81, RMSE = 0.67).
 
Conversely, models like Lasso, ElasticNet, and DummyRegressor recorded R-squared values close to 0 or even negative, with RMSEs around 1.55, signifying subpar performance. RANSACRegressor was inconsistent and produced outrageous values, exhibiting an R-squared of about -1.5e+21 and RMSE exceeding 59.8 billion due to its high sensitivity to noise.
 
For testing set –
  
The top models consisted of HistGradientBoostingRegressor, RandomForestRegressor, LGBMRegressor, and XGBRegressor, each achieving R-squared values around 0.52 and RMSEs near 1.08. These models demonstrated a commendable balance of accuracy and generalization, and does not perform as well as on the training set.
 
-More straightforward models like LinearRegression, Ridge, and LassoCV had R-squared scores of 0.31–0.32 and RMSEs of approximately 1.28–1.29, indicating moderate performance.
 
-Models such as Lasso, ElasticNet, and DummyRegressor again underperformed, with R-squared scores close to zero or negative and RMSEs around 1.55.
 
-GaussianProcessRegressor and KernelRidge performed poorly on the test data, with negative R-squared values of -4.99 and -13.82, and elevated RMSEs of 3.80 and 5.98, respectively.
Comparison of different models by data visualization –
R-squared
RMS
 
CONCLUSION
 
This research highlights the effectiveness of machine learning methodologies in forecasting the bioactivity (pIC50 values) of acetylcholinesterase inhibitors, which are pivotal in addressing Alzheimer’s disease. Utilizing data derived from the ChEMBL database, we systematically curated, refined, and transformed molecular and bioactivity datasets to construct predictive models that estimate the therapeutic potential of candidate compounds.
 
Our approach encompassed meticulous preprocessing steps — including the conversion of SMILES notations into molecular structures, the logarithmic transformation of IC50 values to pIC50, and the application of feature selection techniques to enhance data quality. Following this, exploratory data analysis facilitated a deeper understanding of the data landscape, revealing key patterns and distributions. A suite of regression models was then developed and rigorously evaluated, encompassing algorithms such as Random Forest, XGBoost, Support Vector Regression, Linear Regression, and K-Nearest Neighbors, alongside a broader spectrum of models benchmarked using the Lazypredict library.
 
The comparative evaluation demonstrated that ensemble-based algorithms — particularly RandomForestRegressor, XGBRegressor, and HistGradientBoostingRegressor — consistently achieved superior predictive accuracy, as evidenced by higher R-squared values and lower RMSE scores across both training and testing datasets. Although models such as DecisionTreeRegressor and ExtraTreesRegressor exhibited exceptionally strong performance on training data (suggesting potential overfitting), RandomForest and XGBoost models delivered a more favorable trade-off between precision and generalization capability.
 
Our study affirms that machine learning, particularly ensemble techniques, offers robust predictive power in drug discovery workflows. Nonetheless, issues such as data imbalance, model overfitting, and the intrinsic complexity of biochemical datasets necessitate careful consideration. Future investigations could focus on advanced hyperparameter tuning, integration of richer molecular descriptors, and application of deep learning frameworks to further boost model performance.
 
In conclusion, this work demonstrates that machine learning not only expedites the drug discovery process but also enhances the precision and cost-efficiency of identifying viable therapeutic agents, thereby supporting the advancement of treatments for neurodegenerative conditions like Alzheimer’s disease.
References
 
[1] ChEMBL Database. [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)
[2] RDKit: Open-source cheminformatics. [http://www.rdkit.org](http://www.rdkit.org)
[3] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
[4] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD, 785–794
 
