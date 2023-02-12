%{
Éamonn Ó Cearnaigh (Kearney)
Machine Learning Coursework
-Final Model Training
-Final Model Saving (For testing zip)
-Final Test Set Saving (For testing zip)
2022
%}

clear all; clc; close all;

% Data input

T = readtable('breast-cancer-wisconsin.csv');
T = table2array(T);

% Missing values - Whole dataset

T_missing = ismissing(T);

% Report number of missing values per column

[row,col] = size(T_missing);
for N = 1:col
    %fprintf("Missing values - column %d:", N)
    column_sum = sum(T_missing(:,N));
end

 % Remove rows containing missing values using dataset IDs

 [row,col] = size(T_missing);

 remove_row_count = 0;
 remove_row_array = [];

 for N = 1:row
    row_sum = sum(T_missing(N,:));
    if row_sum > 0
        %fprintf("\nMissing value detected - row %d:\n", N);
        remove_row_count = remove_row_count + 1;
        remove_row_array = [remove_row_array T(N, 1)];
    end
 end

for N = 1:length(remove_row_array)
    T(T(:, 1) == remove_row_array(N),:) = [];
end

% Removing IDs

T = T(:, 2:11);

%{
Changing label IDs
Classes:
2 - Benign - Changed to 0
4 - Malignant - Changed to 1
%}

[row,col] = size(T);
 for N = 1:row

     if T(N, 10) == 2
         T(N, 10) = 0;

     elseif T(N, 10) == 4
            T(N, 10) = 1;

     end
     
 end

% Splitting data into training (70%) and testing (30%) sets
cv = cvpartition(size(T,1),'HoldOut',0.3);
idx = cv.test;
T_train = T(~idx,:);
T_test  = T(idx,:);

% Splitting features X and labels y for test set
% (Training set split during cross-validation)

X_test = T_test(:, 1:9);
y_test = T_test(:, 10);

% Final model training: Naive Bayes

% 10-fold cross-validation
% Splitting training set into training and validation sets
k = 10;
cv = cvpartition(size(T_train, 1),'KFold', k);

% Initialising evaluation metric sums for averages

sum_accuracy = 0;
sum_specificity = 0;
sum_precision = 0;
sum_recall = 0;
sum_F1 = 0;
sum_roc_auc = 0;

% For each fold
for i = 1:k

    % Train/validate split
    validation_indices = test(cv, i);

    fold_X_train = T_train(~validation_indices, 1:9);
    fold_X_validation  = T_train(validation_indices, 1:9);

    fold_y_train = T_train(~validation_indices, 10);
    fold_y_validation  = T_train(validation_indices, 10);

    % Training

    model_nb = fitcnb(fold_X_train, fold_y_train, 'ClassNames', [0,1], 'DistributionNames', 'kernel', 'Kernel', 'normal');

    % Validation
    
    [labels, PostProbs, MisClassCost] = predict(model_nb, fold_X_validation);
    
    result = confusionmat(fold_y_validation, labels);
    tn = result(1,1); % True Negative
    fp = result(1,2); % False Positive
    fn = result(2,1); % False Negative
    tp = result(2,2); % True Positive

    % Evaluation metrics - Training/Validation (Fold only)

    accuracy = (tn + tp) ./ (tn + tp + fn + fp);
    specificity = tn ./ (tn + fp);
    precision = tp ./ (tp + fp);
    recall = tp ./ (tp + fn);
    F1 = (2 * precision * recall) ./ (precision + recall);

    sum_accuracy = sum_accuracy + accuracy;
    sum_specificity = sum_specificity + specificity;
    sum_precision = sum_precision + precision;
    sum_recall = sum_recall + recall;
    sum_F1 = sum_F1 + F1;

    % ROC & AUC - Training/Validation (Fold only)
    [ROC_X, ROC_Y, ROC_T, ROC_AUC] = perfcurve(fold_y_validation, labels, 1);
    sum_roc_auc = sum_roc_auc + ROC_AUC;

end

% Evaluation metrics - Training/Validation (Average of k folds)

avg_accuracy = sum_accuracy / k;
avg_specificity = sum_specificity / k;
avg_precision = sum_precision / k;
avg_recall = sum_recall / k;
avg_F1 = sum_F1 / k;
avg_roc_auc = sum_roc_auc / k;

disp("Naive Bayes");
fprintf("Average Training Accuracy: %d\n", avg_accuracy);
fprintf("Average Training Specificity: %d\n", avg_specificity);
fprintf("Average Training Precision: %d\n", avg_precision);
fprintf("Average Training Recall: %d\n", avg_recall);
fprintf("Average Training F1: %d\n", avg_F1);
fprintf("Average Training AUC: %d\n\n", avg_roc_auc);

% Final model training: Logistic Regression

% 10-fold cross-validation
% Splitting training set into training and validation sets
k = 10;
cv = cvpartition(size(T_train, 1), 'KFold', k);

% Initialising evaluation metric sums for averages

sum_accuracy = 0;
sum_specificity = 0;
sum_precision = 0;
sum_recall = 0;
sum_F1 = 0;
sum_roc_auc = 0;

% For each fold
for i = 1:k

    % Train/validate split
    validation_indices = test(cv, i);

    fold_X_train = T_train(~validation_indices, 1:9);
    fold_X_validation  = T_train(validation_indices, 1:9);

    fold_y_train = T_train(~validation_indices, 10);
    fold_y_validation  = T_train(validation_indices, 10);

    % Training

    B = glmfit(fold_X_train, categorical(fold_y_train),'binomial','link','logit');

    % Validation

    y_predict = mnrval(B, fold_X_validation);
    y_p = y_predict(:, 1);
    random_variable = [];
    for i=1:length(y_p)
        if y_p(i) > 0.5
            random_variable(i) = 1;
        else
            random_variable(i) = 0;
        end
    end
    
    result = confusionmat(fold_y_validation, random_variable);
    tn = result(1,1); % True Negative
    fp = result(1,2); % False Positive
    fn = result(2,1); % False Negative
    tp = result(2,2); % True Positive

    % Evaluation metrics - Training/Validation (Fold only)

    accuracy = (tn + tp) / (tn + tp + fn + fp);
    specificity = tn / (tn + fp);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = (2 * precision * recall) / (precision + recall);

    sum_accuracy = sum_accuracy + accuracy;
    sum_specificity = sum_specificity + specificity;
    sum_precision = sum_precision + precision;
    sum_recall = sum_recall + recall;
    sum_F1 = sum_F1 + F1;

    % ROC & AUC - Training/Validation (Fold only)
    [ROC_X, ROC_Y, ROC_T, ROC_AUC] = perfcurve(fold_y_validation, y_p, 1);
    sum_roc_auc = sum_roc_auc + ROC_AUC;

end

% Evaluation metrics - Training/Validation (Average of k folds)

avg_accuracy = sum_accuracy / k;
avg_specificity = sum_specificity / k;
avg_precision = sum_precision / k;
avg_recall = sum_recall / k;
avg_F1 = sum_F1 / k;
avg_roc_auc = sum_roc_auc / k;

disp("Logistic Regression");
fprintf("Average Training Accuracy: %d\n", avg_accuracy);
fprintf("Average Training Specificity: %d\n", avg_specificity);
fprintf("Average Training Precision: %d\n", avg_precision);
fprintf("Average Training Recall: %d\n", avg_recall);
fprintf("Average Training F1: %d\n", avg_F1);
fprintf("Average Training AUC: %d\n\n", avg_roc_auc);

% Save test set for use in testing zip file

save('X_test.mat', 'X_test');
save('y_test.mat', 'y_test');

% Save final models for use in testing zip file

save('model_nb.mat', 'model_nb');
save('model_lr.mat', 'B');
