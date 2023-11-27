# AgroCode Data Science Cup 2023 First place solution
# Solution architecture
Final solution contains 8 models (one for each $milk yield_{3..8}$). Each model consists from ensemble of two targets prediction boosting cascade. Each cascade is just three boostings: lgbm, xgboost and catboost. Final predictions of cascade reweight with equal weights.  
![alt text](https://github.com/marky24/AgroCode-Data-Sience-Cup-2023-solution/blob/main/src/architecture.png?raw=true)  
## Feature engeneering  
The best features were milk_yield_2 and farm, except them there were a few hancrafted ones. You can find the generation code in generate_features function.   
## Target engeneering  
I think the crucial part of this challenge. Yon can notice that data has strong time shifting target. So prediction this type of target with gradient boosting is not the best idea. So before training we need to normalize it. Here i use two types of normalization: diff with milk_yield_2, linear regression detrend  
![alt text](https://github.com/marky24/AgroCode-Data-Sience-Cup-2023-solution/blob/main/src/linreg_detrend.png?raw=true)  
![alt text](https://github.com/marky24/AgroCode-Data-Sience-Cup-2023-solution/blob/main/src/diff_detrend.png?raw=true)  
More specific parts of solution you can find in file two_targets.py, also there are some inline comments.
# Running code  
Before running you need to install requirements with `pip install -r src/requirements.txt`  
To run code use `python3 two_targets.py -s data/submission.csv -tr data/train.csv -te data/X_test_public.csv`
