import numpy as np
import random

# read data files
train_data = open("train.data", "r").read()
test_data = open("test.data", "r").read()

train_data = train_data.splitlines()
test_data = test_data.splitlines()

# create numpy arrays for test/train data and weights
train1 = np.zeros((40,4))
train2 = np.zeros((40,4))
train3 = np.zeros((40,4))

test1 = np.zeros((10,4))
test2 = np.zeros((10,4))
test3 = np.zeros((10,4))

# format input data as 4 floats
def formatData(item):
    item = item.split(",")
    formatted = []
    for i in range(4):
        formatted.append(float(item[i]))
    return formatted

# add formatted instances to numpy arrays
for instance in range(0,40):
        train1[instance] = (formatData(train_data[instance]))

for instance in range(0,40):
        train2[instance] = (formatData(train_data[instance + 40]))

for instance in range(0,40):
        train3[instance] = (formatData(train_data[instance + 80]))

for instance in range(0,10):
        test1[instance] = (formatData(test_data[instance]))

for instance in range(0,10):
        test2[instance] = (formatData(test_data[instance + 10]))

for instance in range(0,10):
        test3[instance] = (formatData(test_data[instance + 20]))


# perceptron algorithm
def perceptron(weights, bias, instance):
    # sum output - weights * data instances
    activation = np.dot(weights, instance) + bias
    return activation

def perceptronTrain(postiveClass, negativeClass, trainingIter, regularisation=False, regCoef=0.1):
    trainingResults = []
    weights = np.zeros(4)
    bias = 0
    for test in range(trainingIter):
        classChoice = 0
        # positive class on even numbered tests, negative class on odd
        if test%2 == 0:
            classChoice = 1
            instance = postiveClass[random.randrange(postiveClass.shape[0])]
        else:
            classChoice = -1
            instance = negativeClass[random.randrange(negativeClass.shape[0])]
        activation = perceptron(weights, bias, instance)
        if classChoice * activation <= 0:
            if regularisation:
                weights = weights + (classChoice * instance) - ((2 * regCoef) * weights)
            else:
                weights = weights + (classChoice * instance)
            bias = bias + classChoice
            if classChoice == 1:
                trainingResults.append("FP")
            else:
                trainingResults.append("FN")
        else:
            if classChoice == 1:
                trainingResults.append("TP")
            else:
                trainingResults.append("TN")
    print(trainingResults)
    return weights, bias

# this will take an array of test instances and tell you how many are positively classified
def perceptronTest(testInstances, weights, bias):
    numPositives = 0
    for i in range(testInstances.shape[0]):
        instance = testInstances[i]
        if perceptron(weights, bias, instance) > 0:
            numPositives += 1
    return numPositives

def onevsrestTraining(positiveClass, negativeClass1, negativeClass2, trainingIter, regularisation=False, regCoef=0.1):
    combinedNegative = np.concatenate((negativeClass1, negativeClass2), axis=0)
    weights, bias = perceptronTrain(positiveClass, combinedNegative, trainingIter, regularisation, regCoef)
    return weights, bias

def onevsrestTesting(weights, bias, testInstances):
    results = []
    for i in range(testInstances.shape[0]):
        score = np.zeros(3)
        for j in range(3):
            score[j] = perceptron(weights[j], bias[j], testInstances[i])
        results.append(np.argmax(score) + 1)
    return results


#--------------- Task 3 binary training & testing -------------------
# train 1v2
print("Training 1v2")
weights_1v2, bias_1v2 = perceptronTrain(train1, train2, 20)
print("W:", weights_1v2, "b:", bias_1v2)
print("")

# train 2v3
print("Training 2v3")
weights_2v3, bias_2v3 = perceptronTrain(train2, train3, 20)
print("W:", weights_2v3, "b:", bias_2v3)
print("")

# train 1v3
print("Training 1v3")
weights_1v3, bias_1v3 = perceptronTrain(train1, train3, 20)
print("W:", weights_1v3, "b:", bias_1v3)
print("")

# test 1v2
print("Testing binary classifer with 10 cases each")

numPositives_1v2_class1 = perceptronTest(test1, weights_1v2, bias_1v2)
print("Classifier 1v2. Testing with class-1. Number classified positive: ", numPositives_1v2_class1)

numPositives_1v2_class2 = perceptronTest(test2, weights_1v2, bias_1v2)
print("Classifier 1v2. Testing with class-2. Number classified positive: ", numPositives_1v2_class2)

# test 2v3
numPositives_2v3_class2 = perceptronTest(test2, weights_2v3, bias_2v3)
print("Classifier 2v3. Testing with class-2. Number classified positive: ", numPositives_2v3_class2)

numPositives_2v3_class3 = perceptronTest(test3, weights_2v3, bias_2v3)
print("Classifier 2v3. Testing with class-3. Number classified positive: ", numPositives_2v3_class3)

# test 1v2
numPositives_1v3_class1 = perceptronTest(test1, weights_1v3, bias_1v3)
print("Classifier 1v3. Testing with class-1. Number classified positive: ", numPositives_1v3_class1)

numPositives_1v3_class3 = perceptronTest(test3, weights_1v3, bias_1v3)
print("Classifier 1v3. Testing with class-3. Number classified positive: ", numPositives_1v3_class3)
print("")
print("")


print ("----------------- multi-class classifer training and testing -------------------------")
print("")
print("Training 1 vs Rest Classifier")
weights_1vsRest = np.zeros((3, 4))
bias_1vsRest = np.zeros((3,1))
print("")
print("Training class 1 discriminator")
weights_1vsRest[0], bias_1vsRest[0] = onevsrestTraining(train1, train2, train3, 20)
print("Training class 2 discriminator")
weights_1vsRest[1], bias_1vsRest[1] = onevsrestTraining(train2, train3, train1, 20)
print("Training class 3 discriminator")
weights_1vsRest[2], bias_1vsRest[2] = onevsrestTraining(train3, train1, train2, 20)
print("")

# testing 1 vs rest
print("One vs Rest Testing")
prediction1 = onevsrestTesting(weights_1vsRest, bias_1vsRest, test1)
print("Prediction against class 1 test data", prediction1)
prediction1 = onevsrestTesting(weights_1vsRest, bias_1vsRest, test2)
print("Prediction against class 2 test data", prediction1)
prediction1 = onevsrestTesting(weights_1vsRest, bias_1vsRest, test3)
print("Prediction against class 3 test data", prediction1)





print("")
print("")
print("--------------------Regularisation - 1 vs Rest ------------------------------")

coefficients = [0.01, 0.1, 1.0, 10.0, 100.0]

for regCoef in coefficients:
    print("")
    print("********* Using", regCoef, "for lambda")
    weights_regularisation = np.zeros((3, 4))
    bias_reg = np.zeros((3,1))
    print("")
    print("Training class 1 discriminator")
    weights_regularisation[0], bias_reg[0] = onevsrestTraining(train1, train2, train3, 20, True, regCoef)
    print("Training class 2 discriminator")
    weights_regularisation[1], bias_reg[1] = onevsrestTraining(train2, train3, train1, 20, True, regCoef)
    print("Training class 3 discriminator")
    weights_regularisation[2], bias_reg[2] = onevsrestTraining(train3, train1, train2, 20, True, regCoef)
    print("")
    print("Weights:", np.array2string(weights_regularisation).replace("\n", ","))
    print("Bias:", np.array2string(bias_reg).replace("\n", ","))
    print("")

    # testing with regularisation
    prediction1 = onevsrestTesting(weights_regularisation, bias_reg, test1)
    print("Prediction against class 1 test data", prediction1)
    prediction1 = onevsrestTesting(weights_regularisation, bias_reg, test2)
    print("Prediction against class 2 test data", prediction1)
    prediction1 = onevsrestTesting(weights_regularisation, bias_reg, test3)
    print("Prediction against class 3 test data", prediction1)
