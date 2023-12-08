# Cnn_Celebrity_detection

Introduction:
The task involves utilizing Convolutional Neural Networks (CNN) with TensorFlow to classify images of celebrities, including Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The objective is to build a robust machine learning pipeline, encompassing key stages such as dataset loading, preprocessing, model construction, training, evaluation, and prediction.
Dataset Description:
The dataset consists of cropped images of the mentioned celebrities, categorized into different folders. The code uses the OpenCV and Pillow libraries to load and preprocess the images. The dataset is balanced, containing images of each celebrity, and is split into training and testing sets for model development and evaluation.

Preprocessing:
The script preprocesses the images by resizing them to a uniform size (128x128 pixels) and normalizing pixel values. It uses the train_test_split function from scikit-learn to split the dataset into training and testing sets. Additionally, the code employs TensorFlow's normalization function to normalize pixel values in the range [0, 1].

Modeling:
The model architecture is defined using TensorFlow and Keras. It comprises convolutional layers followed by max-pooling layers for feature extraction and down-sampling. The flattened output is then passed through densely connected layers, ending with a softmax layer for multi-class classification. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.

Training:
The training phase involves fitting the model to the training data using the fit function. The training set is further split into training and validation subsets, and an early stopping callback is employed to prevent overfitting. The training process is monitored, and the model is trained for 30 epochs with a batch size of 128.

Prediction:
The code includes a function for making predictions on new images using the trained model. It loads an image, preprocesses it, and obtains a prediction for the celebrity class. The predicted class is then mapped to a celebrity name.

Conclusion:
The script successfully demonstrates the creation and training of a CNN model for celebrity image classification. The evaluation phase reports the accuracy of the model on the test set. The prediction function showcases the model's ability to make predictions on new images. Further improvements could include hyperparameter tuning, data augmentation, and exploring more complex architectures for enhanced performance.
