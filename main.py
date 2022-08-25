import pandas as pd

import dowhy as dowhy
import matplotlib.pyplot as plt
from dowhy import CausalModel, causal_graph
from IPython.display import Image, display
import dowhy.datasets
# test dowhy
# data = dowhy.datasets.linear_dataset(
#     beta=10,
#     num_common_causes=5,
#     num_instruments=2,
#     num_samples=10000,
#     treatment_is_binary=True)
# model = CausalModel(
#     data=data["df"],
#     treatment=data["treatment_name"],
#     outcome=data["outcome_name"],
#     common_causes=data["common_causes_names"],
#     instruments=data["instrument_names"])
# model.view_model(layout="dot")
# display(Image(filename="causal_model.png"));


# upload data
df = pd.read_csv('new.csv', encoding='utf-8')
df.info()

# construct a causal model based on domain knowledge
# 实例化七个可能的假设
# model1
model1 = dowhy.CausalModel(
    data=df,
    treatment='listed_in2',
    outcome='score1',
    common_causes=['cast1', 'cast2', 'cast3', 'country1', 'duration1', 'release_year'],
    instruments=['id', 'director1'])
model1.view_model(layout="dot")
plt.title("causal_model1.png")

from IPython.display import Image, display
display(Image(filename="causal_model.png"))
# model2
model2 = dowhy.CausalModel(
    data=df,
    treatment='cast1',
    outcome='score1',
    common_causes=['listed_in2', 'cast2', 'cast3', 'country1', 'release_year'],
    instruments=['director1'])
model2.view_model(layout="dot")
plt.title("causal_model2.png")

# model3
model3 = dowhy.CausalModel(
    data=df,
    treatment='cast2',
    outcome='score1',
    common_causes=['listed_in2', 'cast1', 'cast3', 'country1', 'release_year'],
    instruments=['director1'])
model3.view_model(layout="dot")
plt.title("causal_model3.png")

# model4
model4 = dowhy.CausalModel(
    data=df,
    treatment='cast3',
    outcome='score',
    common_causes=['listed_in2', 'cast1', 'cast2', 'country1', 'release_year'],
    instruments=['director1'])
model4.view_model(layout="dot")
plt.title("causal_model4.png")

# model5
model5 = dowhy.CausalModel(
    data=df,
    treatment='country1',
    outcome='score1',
    common_causes=['listed_in2', 'duration1', 'release_year'],
    instruments=['director1', 'cast1', 'cast2', 'cast3'])
model5.view_model(layout="dot")
plt.title("causal_model5.png")

# model6
model6 = dowhy.CausalModel(
    data=df,
    treatment='duration1',
    outcome='score1',
    common_causes=['listed_in2', 'cast1', 'cast2', 'cast3', 'release_year'],
    instruments=['director1'])
model6.view_model(layout="dot")
plt.title("causal_model6.png")

# model7
model7 = dowhy.CausalModel(
    data=df,
    treatment='release_year',
    outcome='score1',
    common_causes=[ 'country1', 'listed_in2'],
    instruments=['director1', 'cast1', 'cast2', 'cast3'])
model7.view_model(layout="dot")
plt.title("causal_model7.png")

# 对假设进行计算和检验
# # identify the estimand and give the formula based on the graph
identified_estimand = model1.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# # use the formula to calculate the mean value(ate) of our assumption based on dataset
causal_estimate = model1.estimate_effect(identified_estimand,
                                         method_name="backdoor.linear_regression", target_units='ate')
# print(causal_estimate)
print("Causal Estimate is " + str(causal_estimate.value))

# # refute the assumption:Add a random confounding variable
res_random = model1.refute_estimate(identified_estimand, causal_estimate, method_name="random_common_cause")
print(res_random)  # estimate effect和new effect基本保持稳定，p value较小

# #　Replace intervention with random variables
res_placebo = model1.refute_estimate(identified_estimand, causal_estimate,
                                     method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)  # new effect接近于0

# ＃　Remove a random subset of the data
res_subset = model1.refute_estimate(identified_estimand, causal_estimate,
                                    method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)  # estimate effect和new effect基本保持稳定
