import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 

#..................................................................................................................
df = pd.read_csv("Data/processed/featured_data.csv")

#print(df.head())

# creating X And Y 

Y = df["Machine_Failure"]

X = df[["Air_Temp_K",
      "Process_Temp_K",
      " Rotational_Speed_RPM",
      "Torque_Nm",
      "Tool_Wear_Min",
      "temperature_diff",
      "Tool_Wear_type",
      "RPM_type",
      "Torque_Category",
      "Total_failure_count",
      "High_Risk_flag"]]

#print(x.shape)

#code which lets us in this situation how many observations belongs to which class
# print(Y.value_counts())

# # to calculate its persentage 

# print(Y.value_counts(normalize = True)* 100)


#.............................................................................................................
#TRANING THE MODEL ON 80% TRAIN 20% TEST

X_train, X_test ,Y_train , Y_test = train_test_split(
    X,Y,
    test_size = 0.20,
    random_state = 42,
    stratify = Y
)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


#..............................................................................................................
#creating the numerical and categorical features from our data means seperationg them

Numerical_features = ["Air_Temp_K",
      "Process_Temp_K",
      " Rotational_Speed_RPM",
      "Torque_Nm",
      "Tool_Wear_Min",
     "Total_failure_count",
      "temperature_diff"      
     ]

category_feature = [
      "Tool_Wear_type",
      "RPM_type",
      "Torque_Category",
      "High_Risk_flag"]


#.............................................................................................................
# working on column transfer
# becayse we want the test and train columns values should be in numerical form not in string..................


preprocessor = ColumnTransformer(transformers = [("num",StandardScaler(),
                                                Numerical_features),("cats",
                                                OneHotEncoder(handle_unknown="ignore"),category_feature)])

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_train.shape)

#.......................................................................................................


#CREATING THE LOGISTIC REGRESSIN MODEL .................................................................

from sklearn.linear_model import LogisticRegression


model =  LogisticRegression(
    class_weight = "balanced",
    max_iter = 100,
    random_state = 42
)

model.fit(X_train_processed,Y_train)

#print("s")
#................................................................................................................

Y_pred = model.predict(X_test_processed)
print(Y_pred[:20])


#..............................................................................................................

from sklearn.metrics import f1_score

f1 = f1_score(Y_test,Y_pred)
print(f1)  # till this point F1 score comes 0.95 good but one thing
#about our data set that it is highly imbalance as 95 good machines 0.5 failure machine out of 100000

#...............................................................................................................
# working on the confusin matrics on our imbalance dataset with 0.95 f1_score

from sklearn.metrics import confusion_matrix

#[  TN   FP
#   FN    TP  ]

cm = confusion_matrix(Y_test,Y_pred)
print(cm)

#.............................................................................................................
# precision_score recall_score

from sklearn.metrics import precision_score, recall_score

precision = precision_score(Y_test,Y_pred)
recall = recall_score(Y_test,Y_pred)

print("precision:",precision)

print("recall:",recall)

print("F1 score",f1)

#....................................................................................................................
#Accuracy_score(Accuracy of the model)

from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy:",Accuracy)

#.................................................................................................................
#classification report 
# understand it very important for the model evaluation which we are creating 


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

#...................................................................................................................

# working with the roc and auc

Y_pred_proba = model.predict_proba(X_test_processed)[:,1]

#predict_proba gives us two probabily of every machine failure of goood

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(Y_test,Y_pred_proba)
print("AUC-ROC:",roc_auc)






#..........................................................................................................................

#comparing different models

from sklearn.tree import DecisionTreeClassifier


decision_tree = DecisionTreeClassifier (
    class_weight = "balanced",
    
    random_state = 42
)

decision_tree.fit(X_train_processed,Y_train)

#print("s")

#................................................................................................................
print("DECISION TREE")
Y_pred_tree = decision_tree.predict(X_test_processed)
print(Y_pred[:20])

Y_pred_tree_proba = decision_tree.predict_proba(X_test_processed)[:,1]


Accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy:",Accuracy)

precision = precision_score(Y_test,Y_pred)
print("Precision:",precision)


roc_auc = roc_auc_score(Y_test,Y_pred_tree_proba)
print("AUC-ROC:",roc_auc)

recall = recall_score(Y_test,Y_pred_tree)
print("recall:",recall)

f1 = f1_score(Y_test,Y_pred_tree)
print(f1)


#...................................................................................................
# K_Fold Cross Validation

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)



cv_score = cross_val_score(
   model,
   X_train_processed,
   Y_train,
   cv = cv,scoring="f1"
)


print("Cross-validation F1 score;",cv_score)
print("Mean cv F1:",cv_score.mean())
print("Std CV F1:",cv_score.std())

#...................................................................................................
#feature interpretability

print(model.coef_.shape)

feature_names = preprocessor.get_feature_names_out()

print(len(feature_names))
print(len(model.coef_[0]))

coef_df = pd.DataFrame({
"Features":feature_names,
"Coefficient":model.coef_[0]
})

print(coef_df)

coef_df = coef_df.sort_values(
    by = "Coefficient",
    ascending = False
)

print(coef_df)

#.............................................................................................
#Final model pipeline

final_model = Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",
     LogisticRegression(
         class_weight = "balanced",
         random_state = 42
     ))
])

final_model.fit(X_train,Y_train)

final_pred = final_model.predict(X_test)
print("final")
print(f1_score(Y_test,final_pred))

#................................................................................................
# Saving the model

joblib.dump(final_model, "models/final_model.joblib")

#..........................................................................................
#loading our saved model and checking for its performance

loaded_model = joblib.load("models/final_model.joblib")

loaded_predictions = loaded_model.predict(X_test)
loaded_f1 = f1_score(Y_test,loaded_predictions)
print(loaded_f1)

#........................
#for dashboard app healp

print(X.columns.tolist())