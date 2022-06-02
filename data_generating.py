import pandas as pd
import numpy as np

''' 
# of Jobs = 100 
Processing Time ~ U(1, 20)
Weight ~ U(1, 5)
'''


def generating_pt(jt=10, machine=8):
    processing_time = np.random.uniform(1, 19, size=jt*machine)

    data = pd.DataFrame(processing_time, columns=["P_ij"])
    data.to_csv("./environment/data/data.csv")


def generating_weight(jt=10):
    weight = np.random.uniform(0, 5, size=jt)
    data = pd.DataFrame(weight, columns=["W"])
    data.to_csv("./environment/data/weight.csv")


if __name__ == "__main__":
    generating_pt()
    generating_weight()



