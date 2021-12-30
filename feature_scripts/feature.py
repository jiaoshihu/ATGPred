#!/usr/bin/env python
# _*_coding:utf-8_*_
import pandas as pd
import joblib
import os
from feature_scripts.ASDC import get_ASDC
from feature_scripts.APAAC import get_APAAC
from feature_scripts.CTDC import get_CTDC
from feature_scripts.CTDD import get_CTDD
from feature_scripts.CTDT import get_CTDT
from feature_scripts.DPC import get_DPC
from feature_scripts.DDE import get_DDE
from feature_scripts.PAAC import get_PAAC
from feature_scripts.QSOrder import get_QSOrder

def get_proba_feature(fastas):
    ASDC_feature = pd.DataFrame(get_ASDC(fastas))
    APAAC_feature = pd.DataFrame(get_APAAC(fastas))
    CTDC_feature = pd.DataFrame(get_CTDC(fastas))
    CTDD_feature = pd.DataFrame(get_CTDD(fastas))
    CTDT_feature = pd.DataFrame(get_CTDT(fastas))
    DPC_feature = pd.DataFrame(get_DPC(fastas))
    DDE_feature = pd.DataFrame(get_DDE(fastas))
    PAAC_feature = pd.DataFrame(get_PAAC(fastas))
    QSOrder_feature = pd.DataFrame(get_QSOrder(fastas))

    temp_feature = []
    scale1 = joblib.load("./models/CTDD_scaler.pkl")
    model1 = joblib.load("./models/CTDD_ET.pkl")
    x1 = scale1.transform(CTDD_feature)
    y_pred_proba1 = model1.predict_proba(x1)
    temp_feature.append(y_pred_proba1[:,1])

    scale2 = joblib.load("./models/DDE_scaler.pkl")
    model2 = joblib.load("./models/DDE_SVM.pkl")
    x2 = scale2.transform(DDE_feature)
    y_pred_proba2 = model2.predict_proba(x2)
    temp_feature.append(y_pred_proba2[:,1])

    scale3 = joblib.load("./models/QSOrder_scaler.pkl")
    model3 = joblib.load("./models/QSOrder_ET.pkl")
    x3 = scale3.transform(QSOrder_feature)
    y_pred_proba3 = model3.predict_proba(x3)
    temp_feature.append(y_pred_proba3[:,1])

    scale4 = joblib.load("./models/CTDT_scaler.pkl")
    model4 = joblib.load("./models/CTDT_SVM.pkl")
    x4 = scale4.transform(CTDT_feature)
    y_pred_proba4 = model4.predict_proba(x4)
    temp_feature.append(y_pred_proba4[:,1])

    scale5 = joblib.load("./models/DPC_scaler.pkl")
    model5 = joblib.load("./models/DPC_ET.pkl")
    x5 = scale5.transform(DPC_feature)
    y_pred_proba5 = model5.predict_proba(x5)
    temp_feature.append(y_pred_proba5[:,1])

    scale6 = joblib.load("./models/DPC_scaler.pkl")
    model6 = joblib.load("./models/DPC_AB.pkl")
    x6 = scale6.transform(DPC_feature)
    y_pred_proba6 = model6.predict_proba(x6)
    temp_feature.append(y_pred_proba6[:,1])

    scale7 = joblib.load("./models/CTDD_scaler.pkl")
    model7 = joblib.load("./models/CTDD_AB.pkl")
    x7 = scale7.transform(CTDD_feature)
    y_pred_proba7 = model7.predict_proba(x7)
    temp_feature.append(y_pred_proba7[:,1])

    scale8 = joblib.load("./models/CTDD_scaler.pkl")
    model8 = joblib.load("./models/CTDD_SVM.pkl")
    x8 = scale8.transform(CTDD_feature)
    y_pred_proba8 = model8.predict_proba(x8)
    temp_feature.append(y_pred_proba8[:,1])

    scale9 = joblib.load("./models/APAAC_scaler.pkl")
    model9 = joblib.load("./models/APAAC_ET.pkl")
    x9 = scale9.transform(APAAC_feature)
    y_pred_proba9 = model9.predict_proba(x9)
    temp_feature.append(y_pred_proba9[:,1])

    scale10 = joblib.load("./models/ASDC_scaler.pkl")
    model10 = joblib.load("./models/ASDC_ET.pkl")
    x10 = scale10.transform(ASDC_feature)
    y_pred_proba10 = model10.predict_proba(x10)
    temp_feature.append(y_pred_proba10[:,1])

    scale11 = joblib.load("./models/DDE_scaler.pkl")
    model11 = joblib.load("./models/DDE_AB.pkl")
    x11 = scale11.transform(DDE_feature)
    y_pred_proba11 = model11.predict_proba(x11)
    temp_feature.append(y_pred_proba11[:,1])

    scale12 = joblib.load("./models/ASDC_scaler.pkl")
    model12 = joblib.load("./models/ASDC_LR.pkl")
    x12 = scale12.transform(ASDC_feature)
    y_pred_proba12 = model12.predict_proba(x12)
    temp_feature.append(y_pred_proba12[:,1])

    scale13 = joblib.load("./models/PAAC_scaler.pkl")
    model13 = joblib.load("./models/PAAC_ET.pkl")
    x13 = scale13.transform(PAAC_feature)
    y_pred_proba13 = model13.predict_proba(x13)
    temp_feature.append(y_pred_proba13[:,1])

    scale14 = joblib.load("./models/CTDC_scaler.pkl")
    model14 = joblib.load("./models/CTDC_LR.pkl")
    x14 = scale14.transform(CTDC_feature)
    y_pred_proba14 = model14.predict_proba(x14)
    temp_feature.append(y_pred_proba14[:,1])

    temp_feature = pd.DataFrame(temp_feature)
    proba_feature = pd.DataFrame(temp_feature.values.T)
    proba_feature.to_csv("14D_proba_features.csv", index=0)
    return proba_feature


