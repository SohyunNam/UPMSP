import pandas as pd


def read_weight(jt=10):
    weight_data = pd.read_csv("C:/Users/sohyon/PycharmProjects/UPJSP_SH/environment/data/weight.csv")
    weight_data = list(weight_data["W"])

    return weight_data

def cal_tard(file_path):
    weight = read_weight()
    result = pd.read_csv(file_path)
    result_tard = result[result["Event"] == "Completed"]
    result_tard = result_tard.reset_index(drop=True)

    total_tard = 0
    for i in range(len(result_tard)):
        temp = result_tard.iloc[i]
        jt = int(temp["JobType"])
        total_tard += weight[jt] * float(temp["Memo"])

    mean_wt = total_tard / len(result_tard)

    return mean_wt

if __name__ == "__main__":
    weight = read_weight()
    print(cal_tard("./environment/resultlog_99750.csv", weight))