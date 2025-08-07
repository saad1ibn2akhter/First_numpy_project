# WEATHER DATA ANALYZER
# ------------------------------------------------------------------------------------------
# ===================== Project Tasks Overview =====================
# Simulate or load daily temperature data (e.g., for a month or a year) as a NumPy array.(OK)
# Calculate statistics like daily averages, min/max temperatures, temperature ranges.(OK)
# Identify heatwaves or cold spells based on thresholds.(OK)
# Find trends such as moving averages or weekly averages.
# Detect outliers or sudden spikes in temperature.(OK)
# 1. Generate synthetic temperature data for multiple regions over a month.
# 2. Add random fluctuations and noise to simulate real-world readings.
# 3. Store and structure the data using NumPy arrays (2D).
# 4. Compute statistical summaries:
#    - Average temperature per day and per region.
#    - Maximum and minimum temperatures per region.
#    - Variance and standard deviation of temperatures.
# 5. Identify "anomalous" days based on a threshold deviation.(OK)
# 6. Find the region with the highest average temperature.
# 7. Sort and rank regions based on average temperatures.
# 8. Perform data slicing and reshaping for analysis.
# 9. Use NumPy’s advanced indexing and boolean filtering.
# 10. Display all computed summaries and insights.
# 11. Approximate precipitation(OK)
# ================================================================

# ------------------------------------------------------------------------------------------
import numpy as np
from io import StringIO
from datetime import date
import matplotlib.pyplot as plt

# del range


def line():
    print(
        "-------------------------------------------------------------------"
    )

def heatwaves(monthly_temp, pf):
    overall_avg = np.mean(monthly_temp)
    heat_arr = []
    for start in range(30):
        for end in range(start + 4,30):
            if start == 0:
                sum = pf[end]
            else:
                sum = pf[end] - pf[start - 1]
            sz = end - start + 1
            val = sum / sz
            if val >= overall_avg + 2:
                heat_arr.append((start, end))
    return heat_arr

def Approximate_precipitation(temp, humidity, pressure, wind, cloud):
  print("working on it")
  #PoP=0.3H−0.2T−0.1P+0.25C+0.15W
  pop = np.abs(float((0.3*humidity) -(0.2*temp) - (0.1*pressure) +(0.25*cloud)+(0.15*wind)))
  return max(0,min(pop,100))

def anomaly(monthly_temp):
  ano =0
  mx = -11
  for i in range(1,monthly_temp.size-2):
    if(abs(monthly_temp[i]-monthly_temp[i-1]) +abs(monthly_temp[i]-monthly_temp[i+1])>=mx):
      ano = monthly_temp[i]

  return ano


monthly_temp = np.random.randint(20, 30, size=(30))
print("Monthly temp. data ---> \n", monthly_temp)

today = date.today()
today_str = today.strftime("%d-%m-%Y")
today_arr = np.genfromtxt(StringIO(today_str), delimiter="-", dtype="int")
print("Today's temp (*C) :  ", monthly_temp[today_arr[1] - 1])


avg = np.ceil(np.mean(monthly_temp))  # average
mn = np.min(monthly_temp)
mx = np.max(monthly_temp)
rng = mx - mn
line()
print(
    "Average : ",
    avg,
    "Max (*C) : ",
    mx,
    "Min (*C) : ",
    mn,
    "Range (*C) : ",
    rng,
)
line()


# weekly comparison

pref_temp = np.cumsum(monthly_temp)
print(pref_temp)
pf = pref_temp.tolist()

week = [6, 13, 20, 27]
cw = 1
for i in week:
    val = pf[i] - pf[i - 6]
    avg_week = np.ceil(val / 7)
    print("week", cw, "avg. == ", avg_week, "* C")
    cw += 1

line()
# heat / cold waves
heat_arr = heatwaves(monthly_temp, pf)
print(heat_arr)
line()
# heat / cold waves

#precipitation
t = np.random.randint(10,30)
h = np.random.randint(1,100)
p = np.random.randint(900,1010)
w = np.random.randint(3,30)
cx = np.random.randint(10,100)
chance = Approximate_precipitation(t,h,p,w,cx)
print("Chance of Rain : ",chance, " %")
line()

#anomaly algorithms (max of |v1-v2| + |v2-v3|)
anom = anomaly(monthly_temp)
print("Anomaly temp : " , anom)
line()

#create a dataset of weather of different cities and try to create a 2D map
#Matplotlib parts

manchester = np.random.randint(10,20,30)
lancashire = np.random.randint(15,25,30)

days = np.arange(1,31)
# plt.plot(days,manchester)
plt.plot(days,lancashire)

plt.scatter(days,lancashire,color='orange')

plt.title("Temparature Data - Manchester ")
plt.xlabel("Days")

plt.ylabel("Temparature (*C)")
plt.grid(True)

plt.show()




