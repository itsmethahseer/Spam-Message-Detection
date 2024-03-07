## Spam Message Classification Project

This project aims to classify SMS messages as either spam or not spam using machine learning techniques. The project utilizes various classification algorithms such as Logistic Regression, Random Forest, Decision Tree, Multinomial Naive Bayes, and a Voting Classifier ensemble method. 

### Dataset
The dataset used for training and evaluation is named "Spam SMS Collection.txt". It consists of a collection of SMS messages labeled as spam or non-spam.

### Project Components
1. **Main Python Script (main.py):**
    - This script contains the FastAPI application setup for serving the trained model.
    - Pre-trained models (Random Forest) and the TF-IDF tokenizer are loaded from the disk.
    - FastAPI endpoints are defined for message classification.
    - Preprocessing steps are implemented to clean and transform input text for prediction.

2. **Jupyter Notebook (Spam Message Classification.ipynb):**
    - This notebook contains the code for training the classification models.
    - Various algorithms like Logistic Regression, Random Forest, Decision Tree, Multinomial Naive Bayes, and a Voting Classifier are trained and evaluated.
    - Data preprocessing steps, including text cleaning and TF-IDF vectorization, are performed.
    - Model performance metrics and evaluations are provided.

### Instructions for Usage:
1. **Installation:**
    - Ensure Python is installed on your system.
    - Install the required libraries using `pip install -r requirements.txt`.

2. **Training (Optional):**
    - If you wish to retrain the models, refer to the Jupyter notebook "Spam Message Classification.ipynb".
    - Execute the notebook to train and evaluate various models on the provided dataset.

3. **Model Deployment:**
    - Run the FastAPI application using `uvicorn main:app --reload`.
    - The API will be hosted locally and accessible through `http://127.0.0.1:8000`.
    - Use tools like cURL, Postman, or integrate the API into your applications to classify SMS messages.

### File Structure:
- **main.py**: Main script containing the FastAPI application and model serving code.
- **Spam SMS Collection.txt**: Dataset containing labeled SMS messages for training and evaluation.
- **Spam Message Classification.ipynb**: Jupyter notebook containing model training code and evaluation.
- **rf_model.pkl**: Pre-trained Random Forest model saved using pickle.
- **tokenizer.pkl**: Pre-trained TF-IDF vectorizer saved using pickle.
- **requirements.txt**: File listing required Python libraries and their versions.

### Acknowledgments:
- The project utilizes FastAPI for creating a REST API.
- Machine learning models are trained using scikit-learn.
- Natural language processing tasks are performed using NLTK and scikit-learn.

### Contact Information:
For any inquiries or assistance, please contact [zacthahseer123@gmail.com].
