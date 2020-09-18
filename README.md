# SpecAnalysis

-----

SpecAnalysis was made for the analysis of 2D arrays of multivariate data. Pre-processing methods, various regression methods, and feature selection methods are included. Data plots can be saved in several formats, and metadata is always logged when you save data.

SpecAnalysis is intended as a modular software with great flexibility. This readme provides detailed information on the methods used, and the program should be useable even if you are unfamiliar with programming. The code is also commented, and is hopefully understandable. Due to the modular nature of the code, it is relatively easy to add your own methods or ajdust already existing methods.

Feel free to use SpecAnalysis for your research. You can cite this software with: I.L. Jernelv, D.R. Hjelme, Y. Matsuura, and A. Aksnes. Convolutional neural networks for classification and regression analysis of one-dimensional spectral data. arXiv preprint arXiv:2005.07530 (2020).

NB! The program will not teach you anything about data analysis, machine learning, or deep learning, but it can hopefully be a useful resource for looking at your data.

The code for this program is made in Python3.6, and uses functions/methods from scipy and scikit-learn, as well as methods that have been made in-house.

The program has mainly been tested on mid-infrared spectroscopic data.

## Installing

### Linux

The easiest way to run the program in Linux is to download the GitHub repository, and run the specanalysis.py file from terminal. If you are missing any of the required packages you will get a message in the terminal. Simply install any missing packages, and run the program when you have everything installed.

### Windows

In Windows you can install a Python IDE, for example PyCharm or Spyder, and run the file called specanalysis.py from the IDE. The IDE will give an error message if you are missing any required packages to run SpecAnalysis. In that case, just install the required package, and try to run it again.



## Data Format

Spectral files should be in a two-column format where the first column contains the independent variables, and the second column contains the dependent variables. The columns can be either tab- or comma-separated. The file extension for these files should be .csv, .dpt, or .txt. Header lines in these files containing any extra information must start with a '#'.

The spectral files and their metadata can be loaded into the program through special '.list'-files. The first column in this file should contain the filepaths to all individual spectral files in a dataset. Additional columns contain known metadata for each file, such as concentrations for analytes or categorical variables. The first line in each list file is used to label the different columns. The first column is always labelled "Filepath", while the subsequent columns should be given descriptive names. These .list-files can be made in e.g. Notepad++ on Windows or OSX, or in your editor of choice in Linux.

The "data/FTIR_data" folder contains a folder with a few mid-infrared spectroscopy measurements of glucose solutions, and a corresponding "preprint_data.list" file. This example data can be used to try out the program and as a template for your own .list-files.


## Quick Start Guide

1. Start the program. Navigate to the .list-files with the data you want to analyse in the left-hand column. Assign some data to a training set, and a validation set if applicable.
2. Under "Pre-Processing", set any pre-processing methods that you want to apply to the datasets.
3. Under "Regression", set the regression method that you want to use, and any related parameters.
4. Under "Import Options", choose if you want to do training, training and validation, or cross-validation analysis. For "Column of data to use", select the analyte/component of the sample you want to investigate (this defaults to the first analyte in the .list-file).
5. Press the "Run" button in order to run the program with the chosen parameters. Check the box next to "Save" before running if you want the results to be saved.

NB! There is a known bug where the program freezes as it attempts to load in buttons. Press the Alt key in order to bypass this.


## Spectral Pre-Processing
Several spectral pre-processing methods are available, and can be used to e.g. correct for additive and multiplicative errors in the spectra. All pre-processing methods can be found under the "Pre-processing" tab.

In principle, there are no practical restrictions on applying several pre-processing methods on the same spectra. However, multiple pre-processing steps are always done in a set order for practical purposes. If for example both scatter corrections and differentiation are chosen, the scatter correction method will always be done first.

This is the order of pre-processing steps:
1. Data binning
2. Scatter correction
3. Smoothing/filtering
4. Baseline correction
5. Scaling

Some methods are mutually exclusive due to their similarities, e.g. only one type of baseline correction can be performed during one run. One the other hand, the program allows you to perform pre-processing methods simultaneously which may worsen the prediction. For example, performing both scatter correction and differentiation will often not improve the data prediction as compared to only scatter correction or differentiation. The use of the various methods is up to your discretion.

### Data Binning
Data binning is a method that reduces the number of data points in the spectra by finding the average of adjacent data points. With a bin factor of 4, every 4 data points are averaged and substituted by their mean absorbance intensity. The default binning factor in the program is set to "1", which means that no binning is performed.

### Scatter Correction
Scatter correction methods attempt to correct for variability in samples caused by light scattering, which can give both additive and multiplicative effects. A plethora of methods exist, and this software offers three main alternatives: normalisation, standard normal variate (SNV), and multiplicative scatter correction MSC.

#### Normalisation
Spectra can be normalised by checking the box next to "normalisation".

#### Standard Normal Variate
Standard normal variate (SNV) is a pre-processing method that attempts to correct scatter effects in individual spectra. This is done by taking the mean centre of each spectrum, and then dividing each mean-centred spectrum by its own standard deviation.

SNV is done by checking the box next to "SNV".

#### (Extended) Multiplicative Scatter Correction
MSC and EMSC typically correct against a reference spectrum. You can add your reference spectrum by clicking on "reference spectrum" and browsing through your file system. However, the algorithms can also make an internal reference spectrum based on the average of the training data, if no reference spectrum is available. Note that the reference spectrum must have the same dimensions as the imported data.

### Filtering

#### Savitzky-Golay Smoothing and Differentiation
The imported data files can be smoothed using Savitzky-Golay (SG) filtering. SG filtering attempts to smooth noisy data by fitting a polynomial through linear least-squares in successive sub-sets of data points with a certain window length.

SG filtering can be switched on by pressing the radio button  "Use SG". The minimum and maximum window size, as well as the minimum and maximum polynomial order, can then be set in the boxes on row above.

SG filtering also works together with differentiation. When SG filtering and differentiation are selected at the same time, the differentiation is performed stepwise in the SG window after smoothing. The highest derivative that can be determined depends on the polynomial order used for smoothing, i.e. you need to use at least a second-order polynomial in order to calculate up to the second derivative.

#### Fourier Filter
Fourier filtering is enabled by pressing the "Fourier" button on the filter row. You can choose between different window types on the row below, including Blackman, Blackman-Harris, Hamming, Hann, or none. The spectral data will be padded with the selected window size multiplier, which is set to 1.1 as default. The effect of the Fourier filter can be adjusted by setting the filter cut-off on the same row. The check boxes can then be used to display the Fourier-transformed spectra (in linear og log scale), and to display the spectra after the back transformation.

#### Other Filters
Other filters that are included are moving average, Butterworth, and Hamming.

### Baseline Correction
Variations in baseline often occur in spectral data, and can be seen as an addition to the spectra. Baseline correction can be done in a number of ways, and a few simple methods have been implemented here.  

#### Background correction
With "Subtract first variable as baseline" the first bin of each spectrum is set to zero, and the same constant correction is applied to the entire spectrum. Using "Subtract linear background" a linearly increasing baseline between the first and last bins is subtracted from the spectra. With the "Subtract reference spectrum"  option selected a custom background spectrum can be loaded in. This reference spectrum must have the same dimensions as the chosen dataset. If none of these operations is selected the default operation of no baseline correction is applied.

#### Differentiation/Spectral Derivatives
Differentiation of spectral data affects various spectral effects that are not related to the sample analytes, and is a simple tool for correcting spectral data. The first derivative of a spectrum removes additive spectral effects, while the second derivative also removes multiplicative effects. By default, the program uses finite differences as a numerical method to do differentiation. Note that differentiation by finite differences has a tendency to inflate noise, and can worsen prediction, especially if the SNR is low.

As a default, the spectrum is used in its raw form ("Not der"). The first or second derivatives can also be selected, and these a calculated from finite differences. All of these options are calculated sequentially and displayed if the "all" option is selected.

### Scaling/Standardisation
Scaling standardises features by scaling the data to unit variance and is often combined with mean centering. The standard deviation used is calculated from the training data. Mean centering is chosen by default when the program starts, but can be switched off, while scaling is off by default.

### Plots
The bottom row contains checkmark boxes with options for plotting the spectral data before and after pre-processing. By default the data is plotted before pre-processing, and to see the effect of pre-processing it can be useful to plot them afterwards as well. These plots are saved if the "Save" option in the top right corner is chosen.

## Regression Methods
Several regression methods are available in the program, and all program options can be found under the "Regression Methods" tab. Some of these are standard chemometric methods (PCR, PLSR, MLR), and there are also other machine learning (Random Forest) and deep learning methods. Note that regression with deep learning can be very computationally intensive. No regression modelling will be performed if the "None" option is selected, and can be useful for example when investigating how different pre-processing options change the spectra.

### Multiple Linear Regression (MLR)
MLR is a simple type of linear regression which works as an extension of ordinary least-squares regression. MLR is done if the "MLR" button is selected.

### Principal Component Regression (PCR)
A regression method based on principal component analysis (PCA). PCA is an unsupervised data reduction method, where the original data is projected onto a smaller variable subspace with linearly uncorrelated variables. These variables are called principal components. This program uses the PCA implementation from scikit-learn, which uses the singular value decomposition (SVD). This PCA implementation is then combined with linear regression in order to perform PCR.

Select the "PCR" button in order to do PCR. You can set one target principal component, or a range of principal components separated by a colon. After running the regression model, you can press the "Plot components" button in order to plot the principal components. An overview of the PCA weights can also be visualised by selecting the "Plot scatter plot of weights", and you can enter the wanted principal components as the x- and y-axes in this scatter plot.

### Partial Least-Squares Regression (PLSR)
PLSR is one of the most commonly used multivariate analysis methods in spectroscopy. PLSR finds the regression model by projecting both the dependent and independent variables to a new space, whereas PCR only regresses only the dependent variables. This program uses the PLSR implementation from scikit-learn.

Select the "PLSR" button in order to perform PLSR. You can set one target latent variable, or a range of latent variables separated by a colon. After having run the program, you can plot the resulting latent variables by pressing the "Plot latent variables" button.

### Random Forest Regressor
Random forest is an ensemble learning method that can be used for several purposes, and is used specifically for regression in this program. Random forest regression works by creating multiple decision trees, and combining these for regression analysis. This program uses the RandomForestRegressor as implemented in scikit-learn. Note that random forest regression typically has a longer runtime than PLSR and PCR.

Select the "Tree" button in order to perform random forest regression.  The number of trees created is set with the "n_estimators" variable, and the tree branching depth is set with "Tree depth". After the regressor has been run, the feature importance can be viewed by pressing the "Plot feature importance" button. This will plot normalised values for the importance of each predictor.

### Support Vector Regressor
Support vector machines are a set of learning methods used for classification, outlier detection, and regression. As with the random forest, support vector machines are mainly used for regression in this program. This program uses the SupportVectorRegressor (SVR) from scikit-learn. Over-fitting can be an issue in SVR if there are few samples and the number of independent variables is large. If the prediction errors are high due to over-fitting, try to change the kernel function or adjust the regularisation term.

Select the "SVR" button in order to model data with support vector regression. Four kernel function are available for SVR; linear, polynomial, rbf, and sigmoid. The default setting is linear kernel function.  The box next to "degree" is used to set the degree used for a polynomial kernel function, and is ignored by other kernels. "coef0" is a coefficient used for the polynomial and sigmoid kernel functions. "gamma" is used for all kernels except the linear kernel function, and can be set to either "auto" or "scale".


### Other Embedded Methods
Elastic net regularisation is a method for linear regression that combines the penalty variables from lasso and ridge regression (L_1 and L_2, respectively).

Select the "ElasticNet" button in order to model data with elastic net regularisation. The field for "l1_ratio" sets the ratio between the L1 and L2 penalties, and must be between 0 and 1. The default value is 0.5. The elastic net reduces to lasso regularisation if l1_ratio=1, and reduces to ridge regression if l1_ratio = 0.

Ridge regression, lasso, and elastic net are all examples of learning algorithms where the feature selection is built-in, usually called "embedded methods". Due to this, feature selection and learning are performed at the same time, and these methods can be less computationally expensive than combining feature selection with e.g. PLSR. One disadvantage of embedded methods is that the performance and efficiency of the feature selection cannot be readily separated from that of the machine learning.

Feature selection can technically be used together with embedded methods in this software. In some cases it can be useful, for example manual wavelength selection can be used to exclude wavelength regions with little or no useful information. However, for most other cases it is recommended that embedded methods are not used in conjunction with feature selection.

### Neural Network
Neural networks have been implemented for both regression and classification, and these work in more or less the same way. The neural network variants are therefore explained in more detail in a separate section below.   

## Classification methods
Classification methods are used to divide data into pre-defined classes. As with regression analysis, you first need to train a model on training data, and then either do cross-validation or use a separate test/validation set to verify the model.

**Note: In order to chose classification methods, first go under the "Regression Methods" tab, and press the button "Classifier".** Then move to the tab named "Classifier Methods".

### Support Vector Classification
Data can be modelled with support vector classification (SVC) by selecting the "SVC" button. Four kernels are available for SVC, namely linear, polynomial, rbf, and sigmoid. The default setting is the linear kernel function. The box next to "degree" is used to set the degree used for a polynomial kernel function, and is ignored by other kernels. "coef0" is a coefficient used for the polynomial and sigmoid kernel functions. "gamma" is used for all kernels except the linear kernel function, and can be set to either "auto" or "scale".

### PLS-DA
PLS-DA is combination of PLS and discriminant analysis, and uses one hot encoding to enable PLS analysis on classification data. Do PLS-DA by pressing the "PLS-DA" button. The number of PLS latent variables should be input in the box next to this button.

### k-Nearest Neighbours
Do k-nearest neighbours (kNN) classification by pressing the "kNN" button. kNN is based on an integer number of neighbours. You can also test the model with several neighbours by setting a range of integer numbers separated by a colon.

### Logistic Regression
Logistic regression is done by selecting the "LogReg" button. The radio buttons on the same row determine the penalty term that is used. i.e. either l2-penalty, l1-penalty, elastic net, or none.

## Deep Learning
Several methods using artificial neural networks (ANNs) have been added in SpecAnalysis. ANNs can be used on both regression and classification data, so the following description is valid for both regression and classification of spectroscopy data.

### Dense (Fully-connected) Neural Network

### Recurrent Neural Network

### Convolutional Neural Network

## Wavelength Selection Methods

All program options related to feature selection methods can be found under the "Wavelength Selection" tab. Wavelength selection, or feature selection, is used to remove uncorrelated or noisy predictor variables as a means to reduce data complexity or gain more insight into the data.

### Set Data Range Manually
The wavelength region used for the regression methods can be set manually in the field next to "Data range". The range is indicated by two numbers separated by a ":". More than one range can be set, and in this case the ranges should be separated by a ",". By default ":," is entered in the field, and in this case the entire available data range is used.

Another option is load in a .txt-file to designate which independent variables should be used in the analysis. This .txt-file must be in a two-column format, where one column lists the independent variables and the other column uses 1's and 0's to indicate which variables should be used in the regression. In order to load a feature selection file, press the "Select active wavenumbers" button, and navigate to the file you want to use. This file must have the same feature dimensions as the analysed data.

Setting the data range can be used together with other wavelength selection methods. For example if you set a data range and run moving window selection, the moving windows will be built only inside the chosen data range(s).

### Moving Window (MW)
An explicit wavelength selection method that looks for the most informative wavelength intervals. This is done by creating a window that moves over the whole spectral region, and building regression models at each window position. The optimal window can then be chosen based on prediction error and model complexity. MW wavelength selection has been described several times in scientific literature. The MW implementation used here has been made in-house.

MW wavelength selection is enabled by selecting the "Moving Window" button. The window size can be set with a start and stop range. Note that the process will be more time-consuming if a larger window size range is chosen. A heat map of the prediction error is created from the results, with window position on the x-axis and window size on the y-axis. The regression plot with the lowest prediction error will be displayed, and the window position will be indicated in the plot.

### Genetic Algorithm
Wavelength selection by genetic algorithm is enabled by selecting the "Genetic algorithm" button. Wavelength selection by genetic algorithm has more parameters than the other methods. "Num. individuals" determines the number of different individuals in the genetic algorithm, "crossover rate" determines the rate of individuals that have their "genes" mixed to create the next generation, "mutation" rate determines the rate of random change in individuals from one generation to the next, and "GA generations" set the numbers of generations used. The genetic algorithm used here has been implemented in-house.

Running the genetic algorithm will create a plot that shows the best individual from each generation. A regression plot from the best individual in the last generation will also be displayed at the end. The best individual will also be saved in a .txt-file in a two-column format, with 1's and 0's representing which independent variables are used for the regression.

### Sequential Feature Selection
Forward selection is a simple method for feature selection. The model starts by having no features, and then the features that most improves the model are added iteratively. Forward selection stops once no improvement is found for further addition of features. This method starts with all the features in the model, and iteratively removes the least significant feature. Backward elimination stops when there is no added improvement upon removal of more features.

Sequential feature selection (SFS) uses a combination of forward and backward selection. It starts by adding features as in forward selection, but features can be removed if any subsequently added features make them less important. Sequential selection thereby tries to avoid one of the typical issues of forward selection, which is a tendency to add redundant features. However, it is more computationally intensive.

SFS can be chosen with the "Sequential feature selection" button.

<!--  ### Threshold Selection
Each wavenumber is scored individually according to the correlation with the response variable (e.g. glucose concentration). The best wavenumbers, according to a cut-off defined by the user, can then be used in the regression analysis. Threshold selection is easy to implement and usually very efficient. However, the chosen variables can be redundant or may have unexpected interactions. The cut-off value is also rather arbitrary, as it is user-defined.-->

### Feature Selection Validation

## Import Options

### Loading in Data
All files and folders on the same folder level as the program will be shown in a column on the left side of the program. You can navigate through the different folders by clicking on them.

Clicking on a .list-file will select it, and the selection is indicated with a grey background. Two buttons under the folder tree can be used to set the .list-files as either "Training" or "Validation". Several .list-files can be loaded into the program at once. For example, you can mark several .list-files by holding the CTRL button as you are selecting them, and then pressing "Training" or "Validation".

### Data Import Options
Under the "Data Import" tab, you can choose to analyse the data as "Training", "Training and validation", or "X-val on training". Note that a validation dataset must be selected if you wish to run the analysis as "Training and validation", and training data must be selected for all types of analysis.

### Cross-Validation
Cross-validation in the program is done via the ShuffleSplit iterator from scikit-learn. With this iterator, the user chooses the number of data points (N) to be used as "validation" for the cross-validation and the number of iterations.

For the special case of N=1, if the number of iterations is set to "-1", the program will do leave-one-out cross-validation (LOOCV).  

### Column of Data
The selected dataset may have several columns of data variables, such as concentrations of different analytes. Here you can set the column of data you want to use. By default the first data column will be chosen. After running the program once with a dataset selected, the headers of these columns will be displayed on this row.

### Maximum Concentration
Here you can set the highest concentration/amount of the target analyte you want to include in the data, for training and/or validation.

### Unit
Here you can set the unit of the analyte for regression analysis, and the default unit is [mg/dL].

### Assessing Regression Methods
The performance of a regression or classification model can be measured in several ways, this section explains the different methods used in this program.

For data classification the prediction accuracy is by default measured as the percentage of correctly classified measurements.

#### Root-Mean-Square Errors
One way of evaluating the prediction accuracy of a model is to calculate the root-mean-square error (RMSE), which is a scale-dependent measure of error. Root-mean-square error of calibration (RMSEC) should be selected when investigating only training sets. Root-mean-square error of prediction (RMSEP) should be selected when assessing validation data, and RMSEP is then only calculated on the validation data. If the option "default" is chosen, the program will automatically calculate RMSEP/RMSECV for when validation or cross-validation is chosen, and RMSEC if only training data is chosen.

#### Coefficient of Determination
The coefficient of determination, written as $R^2$, is a metric that indicates how much of the variance in the dependent variable that is predicted from the independent variables. An R^2 of 1 indicates that the variance is perfectly predicted. The correlation coefficient (R) is sometimes used, and is therefore also included as an alternative.

#### Standard Error of Prediction
Standard error of prediction (SEP) is quite similar to RMSE, but does not include the bias term. SEP is therefore smaller or equal to the RMSE.

#### Mean Absolute (Percentage) Error



### Saving Your Data
Generated plots will only be saved if you check the box next to "Save". All plots will then be saved, together with a screen shot of the program. A log file will also be created, with all necessary details of the methods used. This log file also contains the names and file paths of all files that were used in the analysis.

The name of the current save folder is shown next to the "Save" options, and can be changed manually. The folder will be created if it does not exist.


### Other Options
Plots can be saved in either .png, .pdf, or .svg format by choosing the appropriately named button.

Plot metrics such as font size, dpi, width and height can be set manually, and the standard values are filled in by default. Grid lines will be drawn over the plot if the box next to "Grid" is checked.

Some information about the methods used will be displayed as text in the plot, such as regression method, number of components/latent variables/etc. used, pre-processing methods, and so on.

### Number of plots
Any generated plots will be shown in full in the program by default, and if you are testing many model parameters the screen may get very crowded. It is therefore recommended that you only test a few variables at a time.

For trying out many parameters and saving the subsequent data, there is an option for setting the maximum number of plots displayed. This value defaults to "-1", where all plots generated will be shown at the same time. If another integer value is set, the program will only display up to the number of maximum plots. Subsequent plots that are generated during the same run will be displayed over older plots as they are generated, and all plots will be saved if the "Save" option is set.


## Other Information

### Going through many model parameters
This program facilitates testing of many parameters in sequence. This functionality can be useful for example if you need to find the best pre-processing steps for a new dataset. As mentioned at several points, most textboxes with variable parameters allow you to input several variables separated by commas, or a range of variables separated by a colon. For pre-processing, most methods also have a "try all" option. For example for spectral derivatives, you can choose "No der.", "1st der.", "2nd der.", or "try all". With the "try all" option selected, the program will go through all the possible options for that method. With several methods selected at the same time, such as filters, derivatives, and scatter correction, the program will go through all possible combinations of the separate methods.
