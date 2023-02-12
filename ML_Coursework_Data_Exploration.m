%{
Éamonn Ó Cearnaigh (Kearney)
Machine Learning Coursework
Data Exploration
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

% Splitting data into X and y (for analysis)

T_X = T(:,1:9);
T_y = T(:,10);

% Splitting data into training (70%) and testing (30%) sets
cv = cvpartition(size(T, 1), 'HoldOut', 0.3);
idx = cv.test;
T_train = T(~idx,:);
T_test  = T(idx,:);

% Splitting features X and labels y

X_train = T_train(:, 1:9);
y_train = T_train(:, 10);

X_test = T_test(:, 1:9);
y_test = T_test(:, 10);

% Percentage: Benign (0) vs Malignant (1) - T

benign_T = sum(1 - T_y)/size(T_y,1);
malignant_T = sum(T_y)/size(T_y,1);

% Percentage: Benign (0) vs Malignant (1) - y_train

benign_train = sum(1 - y_train)/size(y_train,1);
malignant_train = sum(y_train)/size(y_train,1);

% Percentage: Benign (0) vs Malignant (1) - y_test

benign_test = sum(1 - y_test)/size(y_test,1);
malignant_test = sum(y_test)/size(y_test,1);

% Min - total set

T_min = min(T);

% Min - split sets

X_train_min = min(X_train);
X_test_min = min(X_test);

y_train_min = min(y_train);
y_test_min = min(y_test);

% Max - total set

T_max = max(T);

% Max - split sets

X_train_max = max(X_train);
X_test_max = max(X_test);

y_train_max = max(y_train);
y_test_max = max(y_test);

% Mean - total set

T_mean = mean(T);

% Mean - split sets

X_train_mean = mean(X_train);
X_test_mean = mean(X_test);

y_train_mean = mean(y_train);
y_test_mean = mean(y_test);

% Standard Deviation - total set

T_std = std(T);

% Standard Deviation - split sets

X_train_std = std(X_train);
X_test_std = std(X_test);

y_train_std = std(y_train);
y_test_std = std(y_test);

% Skewness - total set

T_skew = skewness(T);

% Skewness - split sets

X_train_skew = skewness(X_train);
X_test_skew = skewness(X_test);

y_train_skew = skewness(y_train);
y_test_skew = skewness(y_test);

% Histograms

figure();
X_train_hist = histogram(X_train, 10);
title('Training Features');
figure();
X_test_hist = histogram(X_test, 10);
title('Testing Features');
figure();
y_train_hist = histogram(y_train, 10);
title('Training Labels');
figure();
y_test_hist = histogram(y_test, 10);
title('Testing Labels');

% Correlation Matrix

figure();
T_correlation = corrcoef(T);
colormap(gca,'parula');
heat = heatmap(T_correlation);
heat.Colormap = parula;
title('Correlation Matrix');
