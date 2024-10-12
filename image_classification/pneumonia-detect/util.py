import numpy as np

def classify(image, model):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (ultralytics.YOLO): A trained YOLO model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert image to RGB (in case it's not already)
    image = image.convert('RGB')

    # Resize image to the model's required size (use the size from your training)
    image = image.resize((64, 64))  # Change this to match your training image size
    # Make prediction
    results = model.predict(image)
    # Extract prediction and confidence
    probs = results[0].probs.numpy().data
    names_dict = results[0].names
    predict = names_dict[np.argmax(probs)]
    score = float(probs[np.argmax(probs)])

    return predict, score
