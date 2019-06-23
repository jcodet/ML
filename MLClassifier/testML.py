import turicreate as turi
url = "dataset/test/"

#Loading images
data = turi.image_analysis.load_images(url)

#Link folders names to labels
#data["foodType"] = data["path"].apply(lambda path: "Rice" if "rice" in path else "Soup")

#Save the dictionary of all data
data.save("test.sframe")
data.explore()

#Load the disctionary
dataBuffer = turi.SFrame("test.sframe")

# Train the model
model = turi.load_model("rice_or_soup.model")
evaluations = model.evaluate(dataBuffer)

print "Evaluation"
print evaluations["accuracy"]
