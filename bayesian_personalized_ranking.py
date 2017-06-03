import sys;
import numpy;
from pandas import read_csv;
from keras.layers import Input, Embedding, Flatten, Lambda;
from keras.models import Model;
from keras.initializers import RandomNormal;
from keras.regularizers import l2;
from keras.callbacks import EarlyStopping;
from keras import backend as K;

# Please refer to matrix_factorization.py for more comments on similar codes
class BayesianPersonalizedRanking:
    # positiveRatingThreshold: lowest value of ratings viewed as "positive" items; "None" views all rating records as positive items
    # negativeSampleCountPerPositive: number of negative items sampled for each (user, positive item) in training data
    def __init__(self, userCount, itemCount, embeddingDimension = 10, regularizationScale = 0.0, epochCount = 20, validationRatio = 0.1, positiveRatingThreshold = None, negativeSampleCountPerPositive = 3, batchSize = 256):
        self.userCount = userCount;
        self.itemCount = itemCount;
        self.embeddingDimension = embeddingDimension;
        self.regularizationScale = regularizationScale;
        self.epochCount = epochCount;
        self.validationRatio = validationRatio;
        self.positiveRatingThreshold = positiveRatingThreshold;
        self.negativeSampleCountPerPositive = negativeSampleCountPerPositive;
        self.batchSize = batchSize;

        self.build();

    def build(self):
        self.userInputLayer = Input(shape = (1, ), dtype = "int32");

        self.itemPositiveInputLayer = Input(shape = (1, ), dtype = "int32");
        self.itemNegativeInputLayer = Input(shape = (1, ), dtype = "int32");
        
        self.userEmbeddingLayer = Embedding(input_dim = self.userCount, output_dim = self.embeddingDimension, input_length = 1, 
                embeddings_regularizer = l2(self.regularizationScale), embeddings_initializer = RandomNormal())(self.userInputLayer);
        self.userEmbeddingLayer = Flatten()(self.userEmbeddingLayer);
        
        # Both positive and negative items share the same embedding space
        itemEmbeddingLayer = Embedding(input_dim = self.itemCount, output_dim = self.embeddingDimension, input_length = 1, 
                embeddings_regularizer = l2(self.regularizationScale), embeddings_initializer = RandomNormal());

        self.itemPositiveEmbeddingLayer = itemEmbeddingLayer(self.itemPositiveInputLayer);
        self.itemPositiveEmbeddingLayer = Flatten()(self.itemPositiveEmbeddingLayer);
        
        self.itemNegativeEmbeddingLayer = itemEmbeddingLayer(self.itemNegativeInputLayer);
        self.itemNegativeEmbeddingLayer = Flatten()(self.itemNegativeEmbeddingLayer);

        dotDifferenceLayer = Lambda(self.getDotDifference, output_shape = self.getDotDifferenceShape) \
            ([self.userEmbeddingLayer, self.itemPositiveEmbeddingLayer, self.itemNegativeEmbeddingLayer]);
        
        self.model = Model(inputs = [self.userInputLayer, self.itemPositiveInputLayer, self.itemNegativeInputLayer], outputs = dotDifferenceLayer);
        self.model.compile(optimizer = "adam", loss = self.getSoftplusLoss, metrics = [self.getAUC]);

    def fit(self, ratingMatrix):
        numpy.random.shuffle(ratingMatrix);
        
        userMatrix, itemPositiveMatrix, itemNegativeMatrix = self.generateInstances(ratingMatrix);

        # Label set does not exist in BPR, so we give Keras with a dummy label set
        labelMatrix = numpy.ones((userMatrix.shape[0], 1), dtype = int);

        self.model.fit([userMatrix, itemPositiveMatrix, itemNegativeMatrix], labelMatrix, epochs = self.epochCount, batch_size = self.batchSize,
                validation_split = self.validationRatio, callbacks = [EarlyStopping(mode = "min")]);

        userIndexMatrix = numpy.arange(self.userCount, dtype = int).reshape(-1, 1);
        itemIndexMatrix = numpy.arange(self.itemCount, dtype = int).reshape(-1, 1);
        
        userEmbeddingOutputModel = Model(inputs = self.userInputLayer, outputs = self.userEmbeddingLayer);
        userEmbeddingMatrix = userEmbeddingOutputModel.predict(userIndexMatrix);
        
        itemEmbeddingOutputModel = Model(inputs = self.itemPositiveInputLayer, outputs = self.itemPositiveEmbeddingLayer);
        itemEmbeddingMatrix = itemEmbeddingOutputModel.predict(itemIndexMatrix);

        return userEmbeddingMatrix, itemEmbeddingMatrix; 

    def getSoftplusLoss(self, labelMatrix, predictionMatrix):
        return K.mean(K.softplus(- predictionMatrix));

    # Count the ratio of prediction value > 0 (i.e., predicting positive item score > negative item score for a user)
    def getAUC(self, labelMatrix, predictionMatrix):
        return K.mean(K.switch(predictionMatrix > 0, 1, 0));

    def getDotDifference(self, parameterMatrixList):
        userEmbeddingMatrix, itemPositiveEmbeddingMatrix, itemNegativeEmbeddingMatrix = parameterMatrixList;
        return K.batch_dot(userEmbeddingMatrix, itemPositiveEmbeddingMatrix, axes = 1) - K.batch_dot(userEmbeddingMatrix, itemNegativeEmbeddingMatrix, axes = 1)

    def getDotDifferenceShape(self, shapeVectorList):
        userEmbeddingShapeVector, itemPositiveEmbeddingShapeVector, itemNegativeEmbeddingShapeVector = shapeVectorList;
        return userEmbeddingShapeVector[0], 1;

    def generateInstances(self, ratingMatrix):
        positiveSetList = [set() for user in range(self.userCount)];
        positiveCount = 0;

        for (user, item, rating) in ratingMatrix: 
            if self.positiveRatingThreshold is not None and rating < self.positiveRatingThreshold:
                continue;
            
            user = int(user);
            item = int(item);
            positiveSetList[user].add(item);
            positiveCount += 1;

        instanceMatrix = numpy.zeros((self.negativeSampleCountPerPositive * positiveCount, 3), dtype = int);
        row = 0;

        for (user, item, rating) in ratingMatrix:
            if self.positiveRatingThreshold is not None and rating < self.positiveRatingThreshold:
                continue;

            user = int(user);
            item = int(item);

            for s in range(self.negativeSampleCountPerPositive):
                other = numpy.random.randint(self.itemCount);
                while other in positiveSetList[user]:
                    other = numpy.random.randint(self.itemCount);

                instanceMatrix[row] = [user, item, other];
                row += 1;

        numpy.random.shuffle(instanceMatrix);

        return instanceMatrix[:, 0].reshape(-1, 1), instanceMatrix[:, 1].reshape(-1, 1), instanceMatrix[:, 2].reshape(-1, 1);

class DataProcessor:
    def __init__(self):
        self.userNameToIndexDict = dict();
        self.userIndexToNameDict = dict();
        self.userIndex = 0;
        
        self.itemNameToIndexDict = dict();
        self.itemIndexToNameDict = dict();
        self.itemIndex = 0;

    def readRatings(self, filePath):
        ratingMatrix = read_csv(filePath, header = None, delim_whitespace = True).as_matrix().astype(float);

        for (r, (user, item, weight)) in enumerate(ratingMatrix):
            user = int(user);
            item = int(item);

            if user not in self.userNameToIndexDict:
                self.userNameToIndexDict[user] = self.userIndex;
                self.userIndexToNameDict[self.userIndex] = user;
                self.userIndex += 1;

            if item not in self.itemNameToIndexDict:
                self.itemNameToIndexDict[item] = self.itemIndex;
                self.itemIndexToNameDict[self.itemIndex] = item;
                self.itemIndex += 1;

            user = self.userNameToIndexDict[user];
            item = self.itemNameToIndexDict[item];

            ratingMatrix[r] = [user, item, weight];

        return ratingMatrix;

    def writeEmbeddings(self, userFilePath, itemFilePath, userEmbeddingMatrix, itemEmbeddingMatrix):
        userNameVector = numpy.array([self.userIndexToNameDict[index] for index in range(userEmbeddingMatrix.shape[0])]);
        itemNameVector = numpy.array([self.itemIndexToNameDict[index] for index in range(itemEmbeddingMatrix.shape[0])]);

        numpy.savetxt(userFilePath, numpy.hstack((userNameVector[numpy.newaxis].T, userEmbeddingMatrix)), fmt = "%g", header = "", delimiter = " ");
        numpy.savetxt(itemFilePath, numpy.hstack((itemNameVector[numpy.newaxis].T, itemEmbeddingMatrix)), fmt = "%g", header = "", delimiter = " ");

def main():
    ratingFilePath = sys.argv[1];
    userFilePath = ratingFilePath + ".user_embedding";
    itemFilePath = ratingFilePath + ".item_embedding";

    print("Read files");
    # ratingMatrix is an adjacency list with each row (user, item, rating)
    dataProcessor = DataProcessor();
    ratingMatrix = dataProcessor.readRatings(ratingFilePath);
    
    userCount = int(ratingMatrix[: , 0].max()) + 1;
    itemCount = int(ratingMatrix[: , 1].max()) + 1;

    print(userCount, "users");
    print(itemCount, "items");
    print(ratingMatrix.shape[0], "ratings");

    print("Run models");
    model = BayesianPersonalizedRanking(userCount, itemCount);
    userEmbeddingMatrix, itemEmbeddingMatrix = model.fit(ratingMatrix);
   
    print("Write embedding results");
    dataProcessor.writeEmbeddings(userFilePath, itemFilePath, userEmbeddingMatrix, itemEmbeddingMatrix);

    print("OK");

if __name__ == "__main__": 
    main();
