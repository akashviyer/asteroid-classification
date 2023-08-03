# End-to-End Asteroid Classification Project By Akash Iyer

## Introduction
This README provides an overview of a personal, end-to-end data science project I completed over Summer 2023. It covers the entire data lifecycle, from data acquisition and exploration to model building and deployment.

## Objectives and Goals
The primary objective of this project was to apply machine learning and data analysis methods to an asteroid dataset to understand what factors can be used to predict the spectral type of Asteroids in our solar system. I wanted to apply what I have learned in my classes as well as in my own studies to a full-length data science project. I also implemented modular coding, CI/CD pipelines, logging, and exception handling.
## Asteroid Spectral Types
Asteroid spectral types are classifications used to categorize asteroids based on their spectra, which represent the unique patterns of light they emit or reflect at different wavelengths. Spectral types provide valuable information about the composition and surface properties of asteroids.

Machine learning can be useful in classifying asteroid spectral types for many reasons, including:

**Large Datasets:** Advancements in astronomical observations and instrumentations have led to a much larger volume of asteroid data. Machine learning can efficiently process and classify large datasets, allowing us to analyze vast numbers of asteroids - quicker than past surveys conducted by teams of researchers, that took longer.

**Complex Spectral Patterns:**
Spectral types involve intricate patterns that may be challenging for humans to interpret manually. Machine learning algorithms can discern subtle patterns and relationships within the data that human eyes might miss.

Overall, machine learning can provide greater insight, precision, and support to researchers looking to accurately classify asteroids.

## Data Sources
The data used for this project is obtained from various reliable sources. Here are the main datasets used:

1. **Dataset 1:**
   - Name: NASA Jet Propulsion Laboratory Small-Body Database Query
   - Source: https://ssd.jpl.nasa.gov/tools/sbdb_query.html
   - Format: CSV
   - Description: This database has tabular data on millions of asteroids and comets, with physical, orbital, and more properties.

2. **Dataset 2:**
   - Source: https://pds.nasa.gov
   - Format: TAB
   - Description: The Planetary Data System provides free-to-use data from space missions. The data I gathered aggregated the Small Main-Belt Asteroid Spectroscopic Survey (SMASS) and Eight-Color Asteroid Survey (ECAS).

## Data Exploration and Preprocessing
After joining and preliminarily cleaning the two datasets, I explored the data in a Jupyter Notebook. I looked at individual variables, made histograms and bar plots for numerical and categorical data respectively, and analyzed the relationship between the target variable and independent variables. I conducted mutual information tests and aggregated them into a heatmap. My specific thoughts and findings are logged in the notebook.

I used scikit-learn pipelines to organize preprocessing in this project. Experimentation was going to play a large role in my learning and this project, so I needed code that could be easily reproduced and edited for fast and efficient work. My testing and earlier analysis led me to simplify my pipeline since the categorical variables I started with did not have nearly enough variance to be useful. Key preprocessing strategies were cross-validated across models. Although there was a high number of NA values for the diameter and albedo features, imputing them using scikit-learn's IterativeImputer proved to be the most successful, and gave credence to the importance of those two variables.

## Machine Learning Model Development
I first created several base models to see which ones were the most successful and which weren't. These included:
1. CatBoost
2. Decision Tree
3. XGBoost
4. AdaBoost
5. Random Forest
6. K-Nearest Neighbours
7. Artificial Neural Network

Out of these, the ones with the best ROC-AUC (One versus rest) scores were CatBoost, XGBoost, and Random Forest. I spent time structuring my ANN, hyperparameter tuning its structure. This yielded modest results but it wasn't competing with the boosting and tree structure algorithms. Further work may produce better results, and I am considering implementing a more robust network.

After establishing base models, I experimented with adding several new engineered features. I established my own custom tooling for cross-validation, and cross-validated batches of new, similarly engineered features, along with a 'Noise' variable that was simply a range of random numbers. This noise variable was used as a baseline to indicate which features were actually providing information to the models, and which were just as good as or worse than extra noise. 

## Model Evaluation
Due to the imbalanced nature of the dataset and the high count of target classes, I used a stratified KFold cross-validation strategy and evaluated my models using the ROC AUC (One vs rest) metric.

## Conclusion
This project was a fun and greatly educational project for me. I hope to improve upon it going forward and dive deeper into the lingering questions I still have for the data as well as my models.

Overall, handling class imbalance and imputation are likely the most important things to handle with this data. Feature engineering and adding more features likely can boost accuracy, and some of the features I added were greatly utilized in my models. From my testing, the most successful features seemed to be those related to diameter and albedo, as well as orbital information. Diameter and albedo by themselves are greatly useful in this classification, as they are very closely linked to the spectral data we collect from asteroids. As for orbital information, I hypothesize that similar asteroids are in similar areas of the solar system oftentimes and that this is in part due to their history of forming, as well as their distance from the Sun. I read a couple of research papers on imputing diameter and albedo values on this dataset, and in the future, I would be interested in applying their methods to impute these features more precisely, before running models again. Class imbalance was an issue I extensively worked on, as models seemed to over-classify asteroids as classes S or L. I tried SMOTE oversampling, undersampling, as well as tuning class weights, and combinations of these three. Different combinations and tunings with different models seemed to produce modest results. I'd like to try out more robust ensembling techniques next time and see if perhaps some Meta-Learning models could help sort this out. Perhaps even a strategy like first using binary classification on class S separately could produce good results.

A few things still have me confused. For one, CatBoost seems to be outperforming all models, even XGBoost. The odd part is that after my preprocessing, I am not using any categorical features, which is what CatBoost is useful for dealing with. I think it may have to do with the way CatBoost handles regularization. CatBoost implements a built-in method to handle overfitting during training by using a technique called "Ordered Regularization." XGBoost on the other hand uses traditional regularization techniques like L1 (Lasso) and L2 (Ridge) regularization. I need to further research and experiment to get more information on this. As stated previously, I'd also like to try out more complex structures for my Artificial Neural Networks. Perhaps tuning the existence of batch normalization layers as well as the strength of dropout layers would help the model catch up or exceed the other algorithms.

## Acknowledgments
NASA JPL and the Planetary Data System were instrumental in being able to do this project, and I want to thank them for providing easily accessible free-use data online.

## References
https://pds.nasa.gov

https://ssd.jpl.nasa.gov

C. R. Chapman, D. Morrison, and B. Zellner Surface properties of asteroids: A synthesis of polarimetry, radiometry, and spectrophotometry, Icarus, Vol. 25, pp. 104 (1975).

D. J. Tholen Asteroid taxonomic classifications in Asteroids II, pp. 1139-1150, University of Arizona Press (1989).

S. J. Bus, F. Vilas, and M. A. Barucci Visible-wavelength spectroscopy of asteroids in Asteroids III, pp. 169, University of Arizona Press (2002).

S. J. Bus and R. P. Binzel Phase II of the Small Main-belt Asteroid Spectroscopy Survey: A feature-based taxonomy, Icarus, Vol. 158, pp. 146 (2002).

Hossain, M.S., Zabed, M.A. (2023). Machine Learning Approaches for Classification and Diameter Prediction of Asteroids. In: Ahmad, M., Uddin, M.S., Jang, Y.M. (eds) Proceedings of International Conference on Information and Communication Technology for Development. Studies in Autonomic, Data-driven and Industrial Computing. Springer, Singapore. (This provided me the inspiration to iteratively impute diameter and albedo values)

---
**Note to Potential Employers:** This README provides an overview of the end-to-end data science project, showcasing my skills in data exploration, analysis, and machine learning model development. The project demonstrates my ability to handle real-world data and deliver actionable insights. If you have any questions or need further details, feel free to reach out to me at akashviyer@gmail.com. Thank you for considering my work!
