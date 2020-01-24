import numpy as np
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from sklearn import preprocessing
from scipy.signal import butter,filtfilt, lfilter, freqz
from scipy.fftpack import fftfreq

f = open('repeatability - Sheet1.csv','r')
F = f.read()
G = F.split('\n')
G = [elements.split(',') for elements in G]
G1 = G[2:2000]
H = np.transpose(G1)
H = np.reshape(H,(60,-1))
H = [[float(x) for x in row] for row in H]
"""
#opening Maxim file
r = open('PPG_motion.csv','r')
R = r.read()
R1 = R.split('\n')
R2 = [elements.split(',') for elements in R1]
r = R2[1:1100]
r = np.transpose(r)
r = np.reshape(r,(12,-1))
S = [[float(x) for x in row] for row in r]
"""
"""
rolling mean
H[0] = np.array(H[0])
AC_Red = H[0] - np.mean(H[1])
plt.plot(AC_Red)
plt.show()

H0 = H[0].rolling(window=10).mean()

plt.plot(H0)
plt.plot(H[0])
plt.show

h0 = np.array(H[0])
h1 = np.array(H[1])
#h2 = np.array(H[2])
q = np.fft.fft(h0)
q = abs(q)
plt.plot(q)
plt.show()
#freq = fftfreq(q)

w = np.fft.fft(h1)
w = abs(w)
plt.plot(w)
plt.show()

r = np.fft.fft(h2)
r = abs(r)
plt.plot(r)
plt.show()"""

plt.plot(H[3][100:250])
plt.show()

#mean centering
"""g = int(len(H[3])/2)
if g%2 == 0:
    t = g+1
else:
    if g%2 == 1:
        t = g
t = np.array(range(-g,g+1))"""
t = np.arange(len(H[0]))
X = preprocessing.scale(H[3])
plt.plot(t[100:250],X[100:250])
#plt.show()
Y = preprocessing.scale(H[4])
#plt.plot(t,Y)
#plt.show()
Z = preprocessing.scale(H[5])
#plt.plot(t,Z)
plt.xlabel("Sample points")
plt.ylabel("Normalized Amplitude")
plt.title("Comparison of PPG signal (IR) with ECG")
plt.show()

#SNR calculation
x = np.array(H[3])
y = np.array(H[4])
z = np.array(H[5])

DC_Red0 = x.mean(axis=0)
DC_IR0 = y.mean(axis=0)
DC_ECG = z.mean(axis=0)

AC_Red0 = (x - DC_Red0)**2
AC_IR0 = (y - DC_IR0)**2
AC_ECG = (z - DC_ECG)**2

ACDC10 = np.sqrt(sum(AC_Red0))/(DC_Red0)
ACDC20 = np.sqrt(sum(AC_IR0)/(DC_IR0))
ACDCECG = np.sqrt(sum(AC_ECG)/(DC_ECG))
#print(ACDC10, ACDC20,ACDCECG)    

SNR_Red0 = 20 * np.log10(ACDC10)
SNR_IR0 = 20 * np.log10(ACDC20)
SNR_ECG = 20 * np.log10(ACDCECG)
print(SNR_Red0)
print(SNR_IR0)
print(SNR_ECG)


"""m = int(len(S[0])/2)
if m%2 == 0:
    m = m+1
else:
    if m%2 == 1:
        m = m
t_1 = np.array(range(-m-1,m))

s1 = preprocessing.scale(S[7])
plt.plot(t_1,s1)
plt.show()

s2 = preprocessing.scale(S[8])
plt.plot(t_1,s2)
plt.show()"""

"""
def detect(X):
    X1 = []
    for i in range(len(X)):
        if (X[i] < 0.4) or (X[i] > 1):
            np.delete(X,i)
        else:
            X1.append(X[i])  

detect(X)
plt.plot(X1)
plt.show()

detect(Y)
plt.plot(X1)
plt.show"""

# baseline removed
sum1 = 0
t1 = np.arange(len(H[3]))
t2 = [(i**2) for i in t]
for i in range(0,len(t1)):
    sum1 = sum1 + t2[i]
#print(sum1)

sum2 = 0
sum3 = 0
sum4 = 0
for i in range(0,len(t)):
    sum2 = sum2 + t1[i]*X[i]
    sum3 = sum3 + t1[i]*Y[i]
    sum4 = sum4 + t1[i]*Z[i]

b1 = sum2/sum1
b2 = sum3/sum1
b3 = sum4/sum1
y = []
x = []
z = []
for i in range(len(t)):
    b4 = b1 * t1[i]
    b5 = b2 * t1[i]
    b6 = b3 * t1[i]
    b7 = X[i] - b4
    b8 = Y[i] - b5
    b9 = Z[i] - b6
    x.append(b7)
    y.append(b8)
    z.append(b9)
plt.plot(t1[100:250],x[100:250])
plt.show()

#plt.plot(t,y)
#plt.show()

#plt.plot(t,z)
#plt.show()


# moving averages
def avg(x):
    avg = []
    for i in range(6,len(x)):
        A = np.mean(x[i-6:i])
        avg.append(A)
    return avg

x1 = avg(X)
plt.plot(t[100:250],x1[100:250])
plt.show()

y1 = avg(Y)
#plt.plot(y)
#plt.show()

z1 = avg(Z)
#plt.plot(z)
#plt.show()

#filtering high noise data
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    c, a = butter(order, normal_cutoff, btype='low', analog=False)
    return c, a

def butter_lowpass_filter(y, cutoff, fs, order=5):
    c, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(c, a, y)
    return y

fs = 25
order = 2
cutoff = 4
c, a = butter_lowpass(cutoff, fs, order)
r = butter_lowpass_filter(x1, cutoff, fs, order)
plt.plot(t[100:250],x1[100:250])
plt.plot(t[100:250],r[100:250])
plt.show()
c1, a1 = butter_lowpass(cutoff, fs, order)
r1 = butter_lowpass_filter(y1, cutoff, fs, order)
#plt.plot(y)
#plt.plot(r1)
#plt.show()
c2, a2 = butter_lowpass(cutoff, fs, order)
r2 = butter_lowpass_filter(z1, cutoff, fs, order)
#plt.plot(z1)
#plt.plot(r2)
#plt.show()
"""
"""#SNR calculation
r = np.array(r)
r1 = np.array(r1)
DC_Red1 = r.mean(axis=0)
DC_IR1 = r1.mean(axis=0)

AC_Red1 = (r - DC_Red1)**2
AC_IR1 = (r1 - DC_IR1)**2

ACDC11 = np.sqrt(sum(AC_Red1))/(DC_Red1)
ACDC22= np.sqrt(sum(AC_IR1))/(DC_IR1)
print(ACDC11, ACDC22)
   
SNR_Red1 = 10*np.log(ACDC11)
SNR_IR1 = 10*np.log(ACDC22)
print(SNR_Red1)
print(SNR_IR1)
"""
#peak finding algo 
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
z = np.array(Z, dtype=float)"""

peaks, properties = find_peaks(r, distance = 15)
#properties["widths"]
plt.plot(r)
plt.plot(peaks, r[peaks], "x")
plt.show()

peaks1, properties = find_peaks(r1, distance = 15)
#properties["widths"]
#plt.plot(r1)
#plt.plot(peaks1, r1[peaks1], "x")
#plt.show()

peaks2, properties = find_peaks(r2, distance = 15)
#properties["widths"]
#plt.plot(r2)
#plt.plot(peaks2, r2[peaks2], "x")
#plt.show()

"""
peaks3, properties = find_peaks(s1, distance = 15)
#properties["widths"]
plt.plot(s1)
plt.plot(peaks3, s1[peaks3], "x")
plt.show()

peaks4, properties = find_peaks(s2, distance = 15)
#properties["widths"]
plt.plot(s2)
plt.plot(peaks4, s2[peaks4], "x")
plt.show()

Q = np.fft.fft(x)
Q = abs(Q)
plt.plot(Q)
plt.show()

W = np.fft.fft(y)
W = abs(W)
plt.plot(W)
plt.show()

R = np.fft.fft(z)
R = abs(R)
plt.plot(R)
plt.show()
"""

#print(len(t))
HR = (len(peaks)/(len(t)*0.04))*60
print(HR)
HR1 = (len(peaks1)/(len(t)*0.04))*60
print(HR1)
HR2 = (len(peaks2)/(len(t)*0.04))*60
print(HR2)
"""
HR3 = (len(peaks3)/(len(t_1)*0.04))*60
print(HR3)
HR4 = (len(peaks4)/(len(t_1)*0.04))*60
print(HR4)
PPI = 0
P_sum = []
for i in range(1,len(peaks)):
    p = peaks[i]-peaks[i-1]
    PPI = PPI + p**2
    P_sum.append(PPI)
PPIm = (sum(P_sum))**(1/2) 

   
PPI1 = 0
P1_sum = []
for i in range(1,len(peaks1)):
    p = peaks1[i]-peaks1[i-1]
    PPI1 = PPI1 + p**2
    P1_sum.append(PPI)
sum(P1_sum)   
PPI1m = (sum(P1_sum))**(1/2)
 
PPI2 = 0
P2_sum = []
for i in range(1,len(peaks2)):
    p = peaks2[i]-peaks2[i-1]
    PPI2 = PPI2 + p**2
    P2_sum.append(PPI)
sum(P2_sum)
PPI2m = (sum(P2_sum))**(1/2)
"""

 

