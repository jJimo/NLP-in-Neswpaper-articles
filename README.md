<b>NLP in Neswpaper articles </b>

Pregraduate university team project for the class of Data Mining.
The project had three goals:
1. Given a train_set.csv of newspaper's articles, we had to plot a wordcloud for each article category.
2. Implementation of K-means clustering algorithm for the given dataset.
3. Evaluation of scikit-learn's classification algorithms (Multinomial Naive Bayes, Bernouli, KNN, SVM, Random Forest)
   via 10-fold-cross-validation and accuracy and roc-plot metrics. We chose the best versions of the above algorithms     
   and combined them in a new, voting classifier to assign labels to the articles of the test_set.csv .
   
 We converted articles to vectors using a pipeline of count_vectorizer, tfidf transformer and svdl. Impementations of thoses algorithms were provided by scikit-learn.

