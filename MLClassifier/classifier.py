import turicreate as turi
url_train = "dataset/train/"
url_test = "dataset/test/"

#Loading images
training_data = turi.image_analysis.load_images(url_train)
testing_data = turi.image_analysis.load_images(url_test)

#Link folders names to labels
training_data["foodType"] = training_data["path"].apply(lambda path: "Rice" if "rice" in path else "Soup")

#Save the dictionary of all data
training_data.save("rice_or_soup.sframe")
training_data.explore()

#Load the disctionary
dataBuffer = turi.SFrame("rice_or_soup.sframe")

# Train the model
model = turi.image_classifier.create(training_data, target="foodType", model="resnet-50")
training = model.evaluate(training_data)

print training["accuracy"]

predictions = model.predict(testing_data)

print predictions["accuracy"]

model.save("rice_or_soup.model")
model.export_coreml("RiceSoupClassifier.mlmodel")
