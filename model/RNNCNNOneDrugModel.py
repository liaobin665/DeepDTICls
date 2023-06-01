"""
author: liaobin
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import uuid
# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.backend as K
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D, Flatten
from tensorflow.keras.layers import Concatenate, Bidirectional, LSTM, GRU, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score, cohen_kappa_score, precision_score, recall_score, accuracy_score

class RNNCNNOneDrugModel(object):
    def __init__(self, dropout=0.2, filters=128, decay=0.0, prot_len=2500, activation="relu", drug_len1=2048):
        self.__dropout = dropout
        self.__filters = filters
        self.__prot_len = prot_len
        self.__drug_len1 = drug_len1
        self.__activation = activation
        self.__decay = decay
        self.__model_t = self.model(self.__dropout, self.__filters, prot_len=self.__prot_len,
                                    activation=self.__activation, drug_len1=self.__drug_len1)
        opt = Adam(lr=0.0001, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        # K.get_session().run(tf.global_variables_initializer())
        K.get_session().run(tf.compat.v1.global_variables_initializer())

    def Conv(self, size, filters, activation, initializer, regularizer_param):
        def fun(input):
            convlayer = Convolution1D(filters=filters, kernel_size=size, padding='same', kernel_initializer=initializer,
                                    kernel_regularizer=l2(regularizer_param))(input)
            convlayer = BatchNormalization()(convlayer)
            convlayer = Activation(activation)(convlayer)
            return GlobalMaxPooling1D()(convlayer)
        return fun

    def model(self, dropout, filters, prot_len=2500, activation='relu', initializer="glorot_normal", drug_len1=2048):

        input_drug = Input(shape=(drug_len1,))
        input_protein = Input(shape=(prot_len,))
        params_dic = {"kernel_initializer": initializer,
                      "kernel_regularizer": l2(0.0001),
                      }

        layer_protein = Lambda(lambda x: K.expand_dims(x, axis=1))(input_protein)
        layer_protein = Bidirectional(GRU(256))(layer_protein)

        layer_protein = Lambda(lambda x: K.expand_dims(x, axis=1))(layer_protein)
        layer_protein = SpatialDropout1D(0.2)(layer_protein)
        proteinConv = [self.Conv(stride_size, filters, activation, initializer, 0.0001)(layer_protein) for stride_size in
                    (10, 15)]
        if len(proteinConv) != 1:
            layer_protein = Concatenate(axis=1)(proteinConv)
        else:
            layer_protein = proteinConv[0]

        layer_protein = Dense(128, **params_dic)(layer_protein)
        layer_protein = BatchNormalization()(layer_protein)
        layer_protein = Activation(activation)(layer_protein)
        layer_protein = Dropout(dropout)(layer_protein)

        drug_exp_dims = Lambda(lambda x: K.expand_dims(x, axis=1))(input_drug)
        input_layer_d1 = Bidirectional(GRU(256))(drug_exp_dims)

        layer_drug = Lambda(lambda x: K.expand_dims(x, axis=1))(input_layer_d1)
        layer_drug = SpatialDropout1D(0.2)(layer_drug)
        drug_conv = [self.Conv(stride_size, filters, activation, initializer, 0.0001)(layer_drug) for stride_size in (3, 3)]
        if len(drug_conv) != 1:
            layer_drug = Concatenate(axis=1)(drug_conv)
        else:
            layer_drug = drug_conv[0]

        layer_drug = Dense(128, **params_dic)(layer_drug)
        layer_drug = BatchNormalization()(layer_drug)
        layer_drug = Activation(activation)(layer_drug)

        layer_compact = Concatenate(axis=1)([layer_drug, layer_protein])
        layer_compact = Lambda(lambda x: K.expand_dims(x, axis=1))(layer_compact)
        layer_compact = Bidirectional(GRU(128))(layer_compact)
        layer_compact = BatchNormalization()(layer_compact)
        layer_compact = Activation(activation)(layer_compact)
        # if we use sigmoid, do not need + 0.5
        layer_compact = Dense(1, activation='tanh', activity_regularizer=l2(0.0001), **params_dic)(layer_compact)
        layer_compact = Lambda(lambda x: (x + 1.) / 2.)(layer_compact)

        layer_out = Model(inputs=[input_drug, input_protein], outputs=layer_compact)
        return layer_out

    def fit(self, drug_feature1, protein_feature, label, test_drug1, test_pro, test_label, n_epoch=10, batch_size=32):
        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature1, protein_feature], label,
                                         epochs=_ + 1, batch_size=batch_size,
                                         validation_data=([test_drug1, test_pro], test_label),
                                         shuffle=True, verbose=1, initial_epoch=_)
        return self.__model_t

    def summary(self):
        self.__model_t.summary()

    def validation(self, drug_feature1, protein_feature, label, test_drug1, test_pro, test_label, output_file=False, n_epoch=40, batch_size=128, **kwargs):
        epochs = []
        precisionScores = []
        recallScores = []
        accuracyScores = []
        F1_scores = []
        kappas = []
        AUCs = []
        AUPRs = []
        trainLosses = []
        trainAccuracyScores = []
        valLosses = []
        valAccuracyes = []
        datasetname = ''
        for i_epoch in range(n_epoch):
            epochs.append(i_epoch+1)
            history = self.__model_t.fit([drug_feature1, protein_feature], label,
                                         epochs=i_epoch + 1, batch_size=batch_size,
                                         validation_data=([test_drug1, test_pro], test_label),
                                         shuffle=True, verbose=1, initial_epoch=i_epoch)
            trainLosses.append(history.history["loss"][0].astype('float'))
            trainAccuracyScores.append(history.history["accuracy"][0].astype('float'))
            valLosses.append(history.history["val_loss"][0].astype('float'))
            valAccuracyes.append(history.history["val_accuracy"][0].astype('float'))
            for dataset in kwargs:
                print("\tPredction of " + dataset)
                datasetname=dataset
                test_p = kwargs[dataset]["protein_feature"]
                test_d1 = kwargs[dataset]["drug_feature1"]
                test_label = kwargs[dataset]["label"]
                predictionResult = self.__model_t.predict([test_d1, test_p])
                fpr, tpr, thresholds_AUC = roc_curve(test_label, predictionResult)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label, predictionResult)
                AUPR = auc(recall, precision)
                # the activtion function is tanh, if use sigmoid, do not need + 0.5
                precisionScore = precision_score(test_label,  (predictionResult + 0.5).astype(int))
                recallScore = recall_score(test_label,  (predictionResult + 0.5).astype(int))
                accuracyScore = accuracy_score(test_label,  (predictionResult + 0.5).astype(int))
                F1_score = f1_score(test_label, (predictionResult + 0.5).astype(int))
                kappa = cohen_kappa_score(test_label.astype(int), (predictionResult + 0.5).astype(int))

                precisionScores.append(precisionScore)
                recallScores.append(recallScore)
                accuracyScores.append(accuracyScore)
                F1_scores.append(F1_score)
                kappas.append(kappa)
                AUCs.append(AUC)
                AUPRs.append(AUPR)

                # print the result
                print("\tprecisionScore: %0.4f" % precisionScore)
                print("\trecallScore: %0.4f" % recallScore)
                print("\taccuracyScore: %0.4f" % accuracyScore)
                print("\tKappa: %0.4f" % kappa)
                print("\tArea Under ROC Curve(AUC): %0.4f" % AUC)
                print("\tArea Under PR Curve(AUPR): %0.4f" % AUPR)
                print("\tF1_score: %0.4f" % F1_score)
        validationResultDF = pd.DataFrame({'epoch_index':epochs,'train_loss':trainLosses,'val_loss':valLosses,
                                           'train_Accuracy':trainAccuracyScores,'val_Accuracyes':valAccuracyes,
                                           'test_precisionScore':precisionScores,
                                           'test_recallScore':recallScores,'test_accuracyScore':accuracyScores,
                                           'test_F1_score':F1_scores,'test_kappa':kappas,
                                           'test_AUC':AUCs,'test_AUPR':AUPRs})
        filename = 'validationResult_%s.csv' % str(datasetname)
        validationResultDF.to_csv(filename)

    def predict(self, **kwargs):
        results_dic = {}

        for dataset in kwargs:
            result_dic = {}
            test_p = kwargs[dataset]["protein_feature"]
            test_d = kwargs[dataset]["drug_feature"]
            result_dic["label"] = kwargs[dataset]["label"]
            result_dic["predicted"] = self.__model_t.predict([test_d, test_p])
            results_dic[dataset] = result_dic
        return results_dic

    def save(self, output_file):
        self.__model_t.save(output_file)
