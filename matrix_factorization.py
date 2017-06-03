import sys;
import numpy;
from pandas import read_csv;
from keras.layers import Input, Embedding, Dot, Add, Flatten, Lambda, Dense;
from keras.models import Model;
from keras.initializers import RandomNormal;
from keras.regularizers import l2;
from keras.callbacks import EarlyStopping;
from keras import backend as K;

class MatrixFactorization:
    def __init__(self, userCount, itemCount, embeddingDimension = 10, regularizationScale = 0.0, epochCount = 20, validationRatio = 0.1, batchSize = 256, openBias = False, openLambdaLayer = True):
        self.userCount = userCount;
        self.itemCount = itemCount;
        self.embeddingDimension = embeddingDimension;
        self.regularizationScale = regularizationScale;
        self.epochCount = epochCount;
        self.validationRatio = validationRatio;
        self.batchSize = batchSize;
        self.openBias = openBias;
        self.openLambdaLayer = openLambdaLayer;

        self.build();

    def build(self):
        # Define user input -- user index (an integer)
        self.userInputLayer = Input(shape = (1, ), dtype = "int32");

        # Define item input -- item index (an integer)
        self.itemInputLayer = Input(shape = (1, ), dtype = "int32");

        # Define user embedding matrix
        # Embedding oject returns a neural layer as function y = f(x);
        #     then we assign its input x = userInputLayer and output y = userEmbeddingLayer, that is,
        #     f = Embedding( ... );
        #     userEmbeddingLayer = f(userInputLayer)
        # Argument input_length corresponds to the dimension of input layer -- user index, a single integer
        # We adopt additional Flatten() layer to transform the default output tensor shape (batch_size, input_length = 1, input_dim)
        #     to shape (batch_size, input_dim), that is, eliminate the extra dimension of Embedding outputs
        # batch_size is actually the size of a mini-batch
        self.userEmbeddingLayer = Embedding(input_dim = self.userCount, output_dim = self.embeddingDimension, input_length = 1, 
                embeddings_regularizer = l2(self.regularizationScale), embeddings_initializer = RandomNormal())(self.userInputLayer);
        self.userEmbeddingLayer = Flatten()(self.userEmbeddingLayer);

        # Define item embedding matrix
        self.itemEmbeddingLayer = Embedding(input_dim = self.itemCount, output_dim = self.embeddingDimension, input_length = 1, 
                embeddings_regularizer = l2(self.regularizationScale), embeddings_initializer = RandomNormal())(self.itemInputLayer);
        self.itemEmbeddingLayer = Flatten()(self.itemEmbeddingLayer);

        # Define embedding dot product
        # Effect of both Dot and Lambda layers are equivalent, but I leave Lambda here for future references
        if self.openLambdaLayer:
            dotLayer = Lambda(self.getDotProduct, output_shape = self.getDotProductShape)([self.userEmbeddingLayer, self.itemEmbeddingLayer]);
        else:
            dotLayer = Dot(axes = -1)([self.userEmbeddingLayer, self.itemEmbeddingLayer]);

        if self.openBias:
            # Define user bias variables
            self.userBiasLayer = Embedding(input_dim = self.userCount, output_dim = 1, input_length = 1, 
                    embeddings_regularizer = l2(self.regularizationScale), embeddings_initializer = RandomNormal())(self.userInputLayer);
            self.userBiasLayer = Flatten()(self.userBiasLayer);
        
            # Define item bias variables
            self.itemBiasLayer = Embedding(input_dim = self.itemCount, output_dim = 1, input_length = 1, 
                    embeddings_regularizer = l2(self.regularizationScale), embeddings_initializer = RandomNormal())(self.itemInputLayer);
            self.itemBiasLayer = Flatten()(self.itemBiasLayer);

            # Define the bias varaible for all ratings
            # We always give dummy input "1.0" for the layer
            self.oneInputLayer = Input(shape = (1, ));
            self.ratingBiasLayer = Dense(1, activation = "linear", use_bias = False,
                    kernel_regularizer = l2(self.regularizationScale), kernel_initializer = RandomNormal())(self.oneInputLayer);
            
            dotBiasSumLayer = Lambda(self.getDotBiasSum, output_shape = self.getDotBiasSumShape) \
                    ([dotLayer, self.userBiasLayer, self.itemBiasLayer, self.ratingBiasLayer]);

            self.model = Model(inputs = [self.userInputLayer, self.itemInputLayer, self.oneInputLayer], outputs = dotBiasSumLayer);
        else:
            self.model = Model(inputs = [self.userInputLayer, self.itemInputLayer], outputs = dotLayer);
        
        # Define objective function of mean squared error
        # We use Adam algorithm to automatically adjust the learning rate of each embedding parameter
        self.model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = [self.getRMSE]);

    def fit(self, ratingMatrix):
        # We have to shuffle the input data first, because Keras validation_split argument just takes the last part of the input data as validation set
        numpy.random.shuffle(ratingMatrix);
        
        # We need to input matrix shape (batch_size, input_dim = 1)
        # Function reshape(-1, 1) transform 1D tensor shape (batch_size, ) to 2D tensor shaep (batch_size, 1)
        #     "-1" means the last dimension value of a tensor, i.e., "batch_size" here
        #     It is suggested to define standard input shape (batch_size, 1) instead of (batch_size, ) even if the latter is accepted by Keras
        # oneMatrix is just the dummy input for rating bias layer
        userMatrix = ratingMatrix[:, 0].reshape(-1, 1).astype(int);
        itemMatrix = ratingMatrix[:, 1].reshape(-1, 1).astype(int);
        oneMatrix = numpy.ones((ratingMatrix.shape[0], 1));
        labelMatrix = ratingMatrix[:, 2].reshape(-1, 1);
        
        # Run model with early stopping, that is, stopping learning if the validation loss is no longer decreasing
        if self.openBias:
            self.model.fit([userMatrix, itemMatrix, oneMatrix], labelMatrix, epochs = self.epochCount, batch_size = self.batchSize,
                    validation_split = self.validationRatio, callbacks = [EarlyStopping(mode = "min")]);
        else:
            self.model.fit([userMatrix, itemMatrix], labelMatrix, epochs = self.epochCount, batch_size = self.batchSize,
                    validation_split = self.validationRatio, callbacks = [EarlyStopping(mode = "min")]);

        userIndexMatrix = numpy.arange(self.userCount, dtype = int).reshape(-1, 1);
        itemIndexMatrix = numpy.arange(self.userCount, dtype = int).reshape(-1, 1);

        # Extract user embedding matrix, shape (user_count, embedding_dim)
        # We define a small model to capture the results of user embedddings
        userEmbeddingOutputModel = Model(inputs = self.userInputLayer, outputs = self.userEmbeddingLayer);
        userEmbeddingMatrix = userEmbeddingOutputModel.predict(userIndexMatrix);
        
        # Extract item embedding matrix, shape (item_count, embedding_dim)
        itemEmbeddingOutputModel = Model(inputs = self.itemInputLayer, outputs = self.itemEmbeddingLayer);
        itemEmbeddingMatrix = itemEmbeddingOutputModel.predict(itemIndexMatrix);

        if self.openBias:
            # Extract user bias matrix, shape (user_count, 1)
            userBiasOutputModel = Model(inputs = self.userInputLayer, outputs = self.userBiasLayer);
            userBiasMatrix = userBiasOutputModel.predict(userIndexMatrix);
            
            # Extract item bias matrix, shape (item_count, 1)
            itemBiasOutputModel = Model(inputs = self.itemInputLayer, outputs = self.itemBiasLayer);
            itemBiasMatrix = itemBiasOutputModel.predict(itemIndexMatrix);

            # Extract bias for all ratings, shape (1, 1)
            ratingBiasOutputModel = Model(inputs = self.oneInputLayer, outputs = self.ratingBiasLayer);
            ratingBias = ratingBiasOutputModel.predict(numpy.array( [[1.0]] ));

            # Here we do not further handle the bias values, but they can be also stored for future prediction

        return userEmbeddingMatrix, itemEmbeddingMatrix;

    # labelMatrix shape (batch_size, 1)
    # predictionMatrix shape (batch_size, 1)
    def getRMSE(self, labelMatrix, predictionMatrix):
        return K.sqrt(K.mean(K.square(labelMatrix - predictionMatrix)));
    
    # parameterMatrix shape (batch_size, parameter_dim)
    # Each row of a parameter matrix corresponds to the parameters of an training instance
    # batch_row(axes = 1) can compute dot product over two matrices of same size row by row
    def getDotProduct(self, parameterMatrixList):
        userEmbeddingMatrix, itemEmbeddingMatrix = parameterMatrixList;
        return K.batch_dot(userEmbeddingMatrix, itemEmbeddingMatrix, axes = 1);

    # shapeVector == [batch_size, parameter_dim]
    # We return shape (batch_size, 1) which is the size of the dot product result for parameter matrices
    def getDotProductShape(self, shapeVectorList):
        userEmbeddingShapeVector, itemEmbeddingShapeVector = shapeVectorList;
        return userEmbeddingShapeVector[0], 1;

    # parameterMatrix shape (batch_size, parameter_dim = 1)
    def getDotBiasSum(self, parameterMatrixList):
        dotMatrix, userBiasMatrix, itemBiasMatrix, ratingBiasMatrix = parameterMatrixList;
        return dotMatrix + userBiasMatrix + itemBiasMatrix + ratingBiasMatrix;
    
    # shapeVector == [batch_size, parameter_dim = 1]
    def getDotBiasSumShape(self, shapeVectorList):
        dotShapeVector, userBiasShapeVector, itemBiasShapeVector, ratingBiasShapeVector = shapeVectorList;
        return userBiasShapeVector[0], 1;

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

    ratingMatrix[:, 2] -= ratingMatrix[0:, 2].mean();

    print(userCount, "users");
    print(itemCount, "items");
    print(ratingMatrix.shape[0], "ratings");

    print("Run models");
    model = MatrixFactorization(userCount, itemCount);
    userEmbeddingMatrix, itemEmbeddingMatrix = model.fit(ratingMatrix);
   
    print("Write embedding results");
    dataProcessor.writeEmbeddings(userFilePath, itemFilePath, userEmbeddingMatrix, itemEmbeddingMatrix);

    print("OK");

if __name__ == "__main__": 
    main();
