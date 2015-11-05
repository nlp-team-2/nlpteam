Steps to excecute
------------------------------
Steps to Train the classifier
----
1) make sure that devdata folder is on same directory and all the python code on same directory
2) make sure scikitlearn, numpy, nltk, textblob, itertools, collections, pdb(optional<for debugging>), pickle, argparse packages are installed
3) run command "python train.py"
4) all pickle classifiers are created in classifier directory


Steps to Test the classifier
-----------------
1) make sure all the python code on same directory
2) make sure scikitlearn, numpy, nltk, textblob, itertools, collections, pdb(optional<for debugging>), pickle, argparse packages are installed
3) Run command "python test,py -p <folder path containing classifier> -t <folder path containing test data>"
	example : python test.py -p classifiers/ -t devdata/
4) It will print accuracy and confusion matrix for classifiers and print accuracy, R2 and MSE for birth year regressor
   It will also create age.txt,gender.txt and education.txt in textfiles folder(The Output files)
