import pandas as pd
import re
import random
import sys
import statistics
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model.baselines import LabelModel
from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling import LFAnalysis

# Define labels
ABSTAIN = -1
FP = 0
TP = 1

# Define labeling function per quality level (h=high,m=medium,l=low)
@labeling_function()
def tp1h(x):
    is_abstain = []
    after_a = []
    for i in range(0, len(x.trace)):
        if x.trace[i] == "Sign":
            for j in range(i + 1, len(x.trace)):
                after_a.append(x.trace[j])
            if "Receive goods or services" in after_a:
                is_abstain.append(True)
            else:
                is_abstain.append(False)
        after_a = []
        i = i + 1
    if False in is_abstain:
        return TP
    else:
        return ABSTAIN


@labeling_function()
def tp2h(x):
    return TP if "Receive goods or services" in x.tracestring and not "Receive invoice" in x.tracestring \
        else ABSTAIN


@labeling_function()
def tp3h(x):
    is_abstain = []
    after_a = []
    for i in range(0, len(x.trace)):
        if x.trace[i] == "Sign":
            for j in range(i + 1, len(x.trace)):
                after_a.append(x.trace[j])
            if "Receive invoice" in after_a:
                is_abstain.append(True)
            else:
                is_abstain.append(False)
        after_a = []
        i = i + 1
    if False in is_abstain:
        return TP
    else:
        return ABSTAIN


@labeling_function()
def tp4h(x):
    is_abstain = []
    after_a = []
    before_a = []
    for i in range(0, len(x.trace)):
        if x.trace[i] == "Create PO":
            for j in range(i + 1, len(x.trace)):
                after_a.append(x.trace[j])
            for k in range(i-1, -1):
                before_a.append(x.trace[k])
            if "Create PO" in after_a or "Create PO" in before_a:
                is_abstain.append(False)
            else:
                is_abstain.append(True)
        after_a = []
        i = i + 1
    if False in is_abstain: # result = Falseun
        return TP
    else:
        return ABSTAIN


@labeling_function()
def tp5h(x):
    return TP if not re.search("Sign.*Receive invoice", x.tracestring, flags=re.I) \
        else ABSTAIN


@labeling_function()
def tp6h(x):
    return TP if "Sign" in x.tracestring and not re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I) \
        else ABSTAIN #regex response

@labeling_function()
def tp7h(x):
    return TP if "Receive goods or services" in x.tracestring and not "Receive invoice" in x.tracestring \
        else ABSTAIN

@labeling_function()
def tp8h(x):
    return TP if not re.search("Sign.*Receive invoice", x.tracestring, flags=re.I) \
        else ABSTAIN


@labeling_function()
def tp9h(x):
    return TP if "Sign" in x.tracestring and not re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I) \
        else ABSTAIN #regex response

@labeling_function()
def tp10h(x):
    return TP if "Sign" in x.tracestring and not re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I) \
        else ABSTAIN #regex response

@labeling_function()
def tp11h(x):
    return TP if "Receive goods or services" in x.tracestring and not "Receive invoice" in x.tracestring \
        else ABSTAIN

@labeling_function()
def tp12h(x):
    return TP if not re.search("Sign.*Receive invoice", x.tracestring, flags=re.I) \
        else ABSTAIN


@labeling_function()
def tp13h(x):
    return TP if "Sign" in x.tracestring and not re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I) \
        else ABSTAIN #regex response



@labeling_function()
def fp1h(x):
    return FP if tp1h(x) == ABSTAIN and tp2h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp2h(x):
    return FP if tp1h(x) == ABSTAIN and tp3h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp3h(x):
    return FP if tp1h(x) == ABSTAIN and tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp4h(x):
    return FP if tp2h(x) == ABSTAIN and tp3h(x) == ABSTAIN\
        else ABSTAIN


@labeling_function()
def fp5h(x):
    return FP if tp2h(x) == ABSTAIN and tp4h(x) == ABSTAIN and tp3h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp6h(x):
    return FP if tp3h(x) == ABSTAIN and tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp7h(x):
    return FP if tp3h(x) == ABSTAIN \
        else ABSTAIN

@labeling_function()
def fp8h(x):
    return FP if tp1h(x) == ABSTAIN and tp3h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp9h(x):
    return FP if tp1h(x) == ABSTAIN and tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp10h(x):
    return FP if tp2h(x) == ABSTAIN and tp3h(x) == ABSTAIN\
        else ABSTAIN

@labeling_function()
def fp11h(x):
    return FP if tp1h(x) == ABSTAIN and tp3h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp12h(x):
    return FP if tp1h(x) == ABSTAIN and tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp13h(x):
    return FP if tp2h(x) == ABSTAIN and tp3h(x) == ABSTAIN\
        else ABSTAIN


tp_h = [tp1h,tp2h,tp3h,tp4h,tp5h,tp6h,tp7h,tp8h,tp9h,tp10h,tp11h,tp12h,tp13h]
fp_h = [fp1h,fp2h,fp3h,fp4h,fp5h,fp6h,fp7h,fp8h,fp9h,fp10h,fp11h,fp12h,fp13h]


@labeling_function()
def tp1m(x):
    return TP if not "Create PO,Sign" in x.tracestring \
        else ABSTAIN


@labeling_function()
def tp2m(x):
    return TP if "Receive goods or services" in x.tracestring and not "Receive invoice" in x.tracestring \
        else FP


@labeling_function()
def tp3m(x):
    return TP if "Receive invoice" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else FP


@labeling_function()
def fp4m(x):
    return FP if tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp5m(x):
    return FP if tp2h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp6m(x):
    return FP if tp1h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp7m(x):
    return FP if ("Sign" in x.tracestring and re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I)) and ("Receive goods or services" in x.tracestring and "Receive invoice" in x.tracestring) \
        else ABSTAIN

@labeling_function()
def fp8m(x):
    return FP if tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp9m(x):
    return FP if tp2h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp10m(x):
    return FP if tp1h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp11m(x):
    return FP if ("Sign" in x.tracestring and re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I)) and ("Receive goods or services" in x.tracestring and "Receive invoice" in x.tracestring) \
        else ABSTAIN


@labeling_function()
def fp12m(x):
    return FP if tp4h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp13m(x):
    return FP if tp2h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp14m(x):
    return FP if tp1h(x) == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp15m(x):
    return FP if ("Sign" in x.tracestring and re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I)) and ("Receive goods or services" in x.tracestring and "Receive invoice" in x.tracestring) \
        else ABSTAIN


@labeling_function()
def fp16m(x):
    return FP if re.search("^Create PR", x.tracestring, flags=re.I) and re.search("Receive goods or services{1}", x.tracestring, flags=re.I)\
        else ABSTAIN


@labeling_function()
def tp4m(x):
    return TP if not re.search("Sign.*Receive invoice", x.tracestring, flags=re.I) \
        else FP


@labeling_function()
def tp5m(x):
    return TP if "Receive invoice" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else FP


@labeling_function()
def tp6m(x):
    return TP if not re.search("Sign.*Receive invoice", x.tracestring, flags=re.I) \
        else FP


@labeling_function()
def tp7m(x):
    return TP if "Sign" in x.tracestring and not re.search("Sign.*Receive goods or services", x.tracestring, flags=re.I) \
        else FP


@labeling_function()
def tp8m(x):
    return TP if not "Pay" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else ABSTAIN


@labeling_function()
def tp9m(x):
    return TP if not "Pay" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else ABSTAIN


@labeling_function()
def tp10m(x):
    return TP if not "Pay" in x.tracestring and not "Receive invoice" in x.tracestring \
        else ABSTAIN

@labeling_function()
def tp11m(x):
    return TP if not "Create PO,Sign" in x.tracestring \
        else ABSTAIN


@labeling_function()
def tp12m(x):
    return TP if "Receive goods or services" in x.tracestring and not "Receive invoice" in x.tracestring \
        else FP


@labeling_function()
def tp13m(x):
    return TP if "Receive invoice" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else FP


tp_m = [tp1m,tp2m,tp3m,tp4m,tp5m,tp6m,tp7m,tp8m,tp9m,tp10m,tp11m,tp12m,tp13m]
fp_m = [fp4m,fp5m,fp6m,fp7m,fp8m,fp9m,fp10m,fp11m,fp12m,fp13m,fp14m,fp15m,fp16m]


@labeling_function()
def tp1l(x):
    return TP if not "Pay" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else FP


@labeling_function()
def tp2l(x):
    return TP if not "Pay" in x.tracestring and not "Receive invoice" in x.tracestring \
        else FP


@labeling_function()
def tp7l(x):
    return TP if fp4m == ABSTAIN and fp5m == ABSTAIN \
        else FP


@labeling_function()
def fp1l(x):
    return FP if "Create PR,Approve PR" in x.tracestring and "Approve PR,Create PO" in x.tracestring \
            else ABSTAIN


@labeling_function()
def fp2l(x):
    return FP if re.search("^Create PR", x.tracestring, flags=re.I) and re.search("Approve PR{1}", x.tracestring, flags=re.I)\
        else ABSTAIN


@labeling_function()
def fp3l(x):
    return FP if "Create PR,Approve PR" in x.tracestring and "Approve PR,Create PO" in x.tracestring \
            else ABSTAIN


@labeling_function()
def fp4l(x):
    return FP if "Create PR,Approve PR" in x.tracestring and "Approve PR,Create PO" in x.tracestring \
            else ABSTAIN


@labeling_function()
def fp7l(x):
    return FP if re.search("^Create PR", x.tracestring, flags=re.I) and re.search("Create PO{1}", x.tracestring, flags=re.I)\
        else ABSTAIN


@labeling_function()
def fp8l(x):
    return FP if re.search("^Create PR", x.tracestring, flags=re.I) and re.search("Create PO{1}", x.tracestring, flags=re.I)\
        else ABSTAIN


@labeling_function()
def fp9l(x):
    return FP if re.search("^Create PR", x.tracestring, flags=re.I) and re.search("Sign{1}", x.tracestring, flags=re.I)\
        else ABSTAIN


@labeling_function()
def fp10l(x):
    return FP if tp1l == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp11l(x):
    return FP if tp2l == ABSTAIN \
        else ABSTAIN


@labeling_function()
def fp12l(x):
    return FP if re.search("^Create PR", x.tracestring, flags=re.I) \
        else ABSTAIN

@labeling_function()
def fp13l(x):
    return FP if re.search("Sign{1}", x.tracestring, flags=re.I) \
        else ABSTAIN

@labeling_function()
def tp3l(x):
    return TP if fp1l == ABSTAIN \
        else FP


@labeling_function()
def tp4l(x):
    return TP if fp2l == ABSTAIN \
        else FP


@labeling_function()
def tp5l(x):
    return TP if fp3l == ABSTAIN \
        else FP


@labeling_function()
def tp6l(x):
    return TP if fp4l == ABSTAIN \
        else FP

@labeling_function()
def tp14l(x):
    return TP if fp1l == ABSTAIN \
        else FP


@labeling_function()
def tp8l(x):
    return TP if fp2l == ABSTAIN \
        else FP


@labeling_function()
def tp9l(x):
    return TP if not "Pay" in x.tracestring and not "Receive goods or services" in x.tracestring \
        else FP

@labeling_function()
def tp10l(x):
    return TP if fp3l == ABSTAIN \
        else FP


@labeling_function()
def tp11l(x):
    return TP if fp4l == ABSTAIN \
        else FP

@labeling_function()
def tp12l(x):
    return TP if fp3l == ABSTAIN \
        else FP


@labeling_function()
def tp13l(x):
    return TP if fp4l == ABSTAIN \
        else FP




#tp_l = [tp1l,tp2l,tp3l,tp4l,tp5l,tp6l,tp7l,tp8l,tp9l,tp10l,tp11l,tp12l,tp13l,tp14l]
tp_l = [fp1l,fp2l,fp3l,fp4l,fp7l,tp6l,tp7l,tp8l,tp9l,tp10l,tp11l,tp12l,tp13l,tp14l]
fp_l = [fp1l,fp2l,fp3l,fp4l,fp7l,fp8l,fp9l,fp10l,fp11l,fp12l,fp13l]

lst_acc_mv = []
lst_acc_lm = []

# Combine labeling functions in a model, repeat
# for j in range(0,6): # straks veranderen naar range(0,6)
    # for i in range(1,11):
        # seedValue = random.randint(0,1000000)
        random.seed(i*random.randint(0,1000000))
        print(i)
        lfs = random.sample(tp_h, k=0) + random.sample(tp_m, k=0) + random.sample(tp_l, k=0) + \
              random.sample(fp_h, k=0) + random.sample(fp_m, k=0) + random.sample(fp_l, k=11)
        print(lfs)

        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train)  # Label matrix event log
        L_valid = applier.apply(df=df_valid)  # Label matrix validation set²

        # on training set
        label_model = LabelModel(cardinality=2, verbose=True)
        majority_model = MajorityLabelVoter(cardinality=2, verbose=True)
        label_model.fit(L_train, n_epochs=5000, log_freq=50, seed=100)
        label_model.fit(L_valid, n_epochs=5000, log_freq=50, seed=100) #train the model
        pred_train_mv = majority_model.predict(L=L_train)
        pred_train_lm = label_model.predict(L=L_train)

        # Add column with the final labels
        # df_train["lm_label"]=label_model.predict(L=L_train,tie_break_policy="random")
        # df_train["mv_label"]= majority_model.predict(L=L_train,tie_break_policy="random")
        # df_valid["lm_label"]=label_model.predict(L=L_valid,tie_break_policy="random") # FP is never predicted
        # df_valid["mv_label"]= majority_model.predict(L=L_valid,tie_break_policy="random")
        # df_valid["label_true"]=Y_valid

        # preds_t = {'Predictions MV': df_train["mv_label"], 'Predictions LM': df_train["lm_label"]}
        # predictions_train = pd.DataFrame(preds_t)
        # preds_v = {'Predictions MV': df_valid["mv_label"], 'Predictions LM': df_valid["lm_label"]}
        # predictions_valid = pd.DataFrame(preds_v)

        # Calculate accuracy
        majority_acc = majority_model.score(L_valid,Y_valid,tie_break_policy="random")[
            "accuracy"
        ]
        print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
        # majorityvote_summary = majority_model.score(L_valid,Y_valid,tie_break_policy="random", metrics=["accuracy"])

        labelmodel_acc = label_model.score(L_valid,Y_valid,tie_break_policy="random")[
            "accuracy"
        ]
        print(f"{'Label Model Accuracy:':<25} {labelmodel_acc * 100:.1f}%")
        # labelmodel_summary = [label_model.score(L_valid,Y_valid,tie_break_policy="random", metrics=["accuracy"])]

        # Store values in list
        acc_mv = majority_acc
        lst_acc_mv.append(acc_mv)
        acc_lm = labelmodel_acc
        lst_acc_lm.append(acc_lm)

        # Labeling function accuracy
        lf_accuracies = LFAnalysis(L=L_valid, lfs=lfs).lf_empirical_accuracies(Y_valid)
        print(lf_accuracies)
        lf_acc.append(lf_accuracies)

        i=i+1

    mean_acc_mv.append(statistics.mean(lst_acc_mv))
    mean_acc_lm.append(statistics.mean(lst_acc_lm))

# LF and label analysis
# lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
# lf_probs = LFAnalysis(L=L_valid, lfs=lfs).lf_empirical_probs(Y_valid, 2) # Raar, wil een Y_test die evengroot als de training set
# lf_conflicts = LFAnalysis(L=L_train, lfs=lfs).lf_conflicts()
# label_conflict = LFAnalysis(L_train, lfs).label_conflict()
# label_overlap = LFAnalysis(L_train, lfs).label_overlap() #at least two labels
# label_coverage = LFAnalysis(L_train, lfs).label_coverage() #at least one label

# lf_probs = LFAnalysis(L=L_valid, lfs=lfs).lf_empirical_probs(Y=Y_valid,k=3)
# lf_accuracies = LFAnalysis(L=L_valid, lfs=lfs).lf_empirical_accuracies(Y_valid)
# print(lf_accuracies)
# lf_acc.append(lf_accuracies)


