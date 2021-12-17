---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

*user note, T is not current temperature but actually intitial temperature

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
T_t = 85 # Temperature in fahrenheit at time t which is 11:00 a.m,
T_dt = 74 # Change in temperature after change in time in fahrenheit
dt = 2 # Change in time in hours
Ta = 65 # Ambient temperature in fahrenheit
DT = (T_dt - T_t)/dt
k = -DT/(T_t-Ta)
print(k)
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def K_function(T_1, T_2, T_ambient, t_elapsed):
    '''Where T_1 is the initial temperature, T_2 is the temperature after change in time, T_ambient is ambient temperature
    and t_elapsed is time elapsed a.k.a change in time. The function then returns K, the empirical constant'''
    
    k = -((T_2 -T_1)/t_elapsed)/(T_1-T_ambient)
    return k
K_function(85, 74, 65, 2)
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
# 3 a.
#Analytical
def T_analytical(t):
    Ta = 65
    To = 85
    K = 0.275

    T = Ta + (To-Ta)*np.exp(-K*t)
    return T
tA = np.arange(0, 20+1, 1)
print(tA)
plt.plot(tA, T_analytical(tA), '-', label= 'analytical');

#Numerical
Ta = 65
k = 0.275

t = np.linspace(0, 20, 10)
dt = t[1] - t[0]
T_num = np.zeros(len(t))
T_num[0] = 85

for i in range(1, len(t)):
    T_num[i] = T_num[i-1] - k*(T_num[i-1] - Ta)*dt
plt.plot(t, T_num, 'o-', label='numerical');

#Numerical
Ta = 65
k = 0.275

t2 = np.linspace(0, 20, 50)
dt = t2[1] - t2[0]
T_num2 = np.zeros(len(t2))
T_num2[0] = 85

for i in range(1, len(t2)):
    T_num2[i] = T_num2[i-1] - k*(T_num2[i-1] - Ta)*dt
plt.plot(t2, T_num2, 'o-', label='numerical increased time step');



plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
# 3 b.
print('for 3b. it looks like as time approaches infinity, the temperature goes to the ambient temperature 65')
```

```{code-cell} ipython3
# 3 c.
t = np.linspace(-1.845, 2)
dt = t[1] - t[0]
T = np.zeros(len(t))
T[0] = 98.6

for i in range(0, len(t)-1):
    T[i+1] = T[i] - 0.275*(T[i] - 65)*dt

plt.plot(t, T, label = 'numerical estimation');
plt.plot(0, 85, 's', label ='T at 11:00 am',markersize = 10);
print('for 3c., I estimate that the time of death is at around 1.845 hours before 11:00 am because setting T[0] = 98.6 allows me to change the hours before 11:00am until the blue line passes the dot which is where temperature is 85 at 11:00 am, t=0')

plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
plt.plot?
```

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
# 4 a.
import numpy as np
import matplotlib.pyplot as plt

time_vals = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
temp_vals = np.array([50, 51, 55, 60, 65, 70, 75, 80])

def ambient_temp(t):
    '''Gives ambient temperature at time 't' where t=0 is 11 am '''
    if t >=-5 and t<=-4:
        return temp_vals[0]
    elif t>-4 and t<=-3:
        return temp_vals[1]
    elif t>-3 and t<=-2:
        return temp_vals[2]
    elif t>-2 and t<=-1:
        return temp_vals[3]
    elif t>-1 and t<=0:
        return temp_vals[4]
    elif t>0 and t<=1:
        return temp_vals[5]
    elif t>1 and t<=2:
        return temp_vals[6]


print(ambient_temp(0)) # (0 hours=11am, 65'F)
print()

total_time = np.linspace(-5,2) # np.linspace(start, stop, steps)
total_temp = np.array([ambient_temp(t) for t in total_time])
plt.plot(total_time, total_temp)
plt.xlabel('Time (Hours)');
plt.ylabel('Ambient Temperature (F)');
print()
print()
print('This is what I did for 4 a. I would say it looks correct according to the livestreams. A better way to get T_a(T) could be adding linear interpolation between the times so the line becomes smooth.')
```

```{code-cell} ipython3
# 4b Part 1. Comparing non-linear Euler with linear analytical

#Analytical
def T_analytical(t):
    Ta = 65
    To = 85
    K = 0.275

    T = Ta + (To-Ta)*np.exp(-K*t)
    return T
step = .25
tA = np.arange(0, 2+step, step)
print(tA)
plt.plot(tA, T_analytical(tA), '-', label= 'linear analytical');


# Non-linear Euler Numerical

t = np.linspace(0, 2, 50) #step forward 2 hours
dt = t[1] - t[0]
T_num = np.zeros(len(t))
T_num[0] = 85
k = 0.275
for i in range(1, len(t)):
    T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt

    



plt.plot(t, T_num, 'o-', label='nonlinear numerical');
plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
# 4b Part 2 Non-linear Euler Numerical to Guess time at T=98.6 Fahrenheit

t = np.linspace(-1.75, 2, 100)
dt = t[1] - t[0]
T_num = np.zeros(len(t))
T_num[0] = 98.6
k = 0.275
for i in range(1, len(t)):
    T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt

plt.plot(t, T_num, 'o-', label='nonlinear numerical');
plt.plot(0, 85, 's', label ='T at 11:00 am', markersize = 10)
plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');

print('for 4b., using the nonlinear numerical and using the same guessing method for 3c, the time the body is at 98.6 fahrenheit os -1.75 hours before 11:00 a.m.')
```

```{code-cell} ipython3
'''BELOW ARE THE TRIAL AND ERRORS I DID TO ATTEMPT TO MAKE 4b PART ONE GRAPH EXTEND FROM 6AM TO 1PM INSTEAD OF 11AM TO 1PM'''
```

```{code-cell} ipython3
#Analytical
def T_analytical(t):
    Ta = 65
    To = 85
    K = 0.275

    T = Ta + (To-Ta)*np.exp(-K*t)
    return T
step = .25
tA = np.arange(-5, 0+step, step)
print(tA)
plt.plot(tA, T_analytical(tA), '-', label= 'linear analytical');

#Numerical
t = np.linspace(0, 5) # step backwards 5 hours
'''Saw this on gitter but I thought we needed to account for changing ambient temperature'''
T = np.zeros(len(t))
T[0] = 85
K = 0.275

for i in range(0, len(t)-1):
    T[i+1] = T[i] + k * (T[i] - 65) * (t[i+1] - t[i])

plt.plot(- t, T, 'o-', label='nonlinear numerical backwards')


plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
time_vals = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
temp_vals = np.array([50, 51, 55, 60, 65, 70, 75, 80])

def ambient_tempBACKWARDS(t):
    if t>=0 and t<=1:
        return temp_vals[5]
    elif t>1 and t<=2:
        return temp_vals[4]
    elif t>2 and t<=3:
        return temp_vals[3]
    elif t>3 and t<=4:
        return temp_vals[2]
    elif t>4 and t<=5:
        return temp_vals[1]
    
```

```{code-cell} ipython3
#Analytical
def T_analytical(t):
    Ta = 65
    To = 85
    K = 0.275

    T = Ta + (To-Ta)*np.exp(-K*t)
    return T
step = .25
tA = np.arange(-5, 0+step, step)
print(tA)
plt.plot(tA, T_analytical(tA), '-', label= 'linear analytical');

#Numerical
t = np.linspace(0, 5) # step backwards 5 hours
T = np.zeros(len(t))
T[0] = 85
K = 0.275

for i in range(0, len(t)-1):
    T[i+1] = T[i] + k * (T[i] -  ambient_tempBACKWARDS(t[i])) * (t[i+1] - t[i])

plt.plot(- t, T, 'o-', label='nonlinear numerical backwards')


plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
#Analytical
def T_analytical(t):
    Ta = 65
    To = 85
    K = 0.275

    T = Ta + (To-Ta)*np.exp(-K*t)
    return T
step = .25
tA = np.arange(-5, 2+step, step)
print(tA)
plt.plot(tA, T_analytical(tA), '-', label= 'linear analytical');

#Numerical

t = np.linspace(-5, 2, 50)
dt = t[1] - t[0]
T_num = np.zeros(len(t))
T_num[0] = 85
k = 0.275
for i in range(1, len(t)):
    if t[i]<0:
        T_num[i] = T_num[i-1] + k * (T_num[i-1] -  ambient_tempBACKWARDS(abs(t[i-1])) * dt
    else t[i]>0:
        T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt
    elif t[i] = 0:
        T_num[i] = 85
    


plt.plot(t, T_num, 'o-', label='nonlinear numerical');
plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
np.linspace(0, 5)
```

```{code-cell} ipython3
# Non-linear Euler Numerical

t = np.linspace(-5, 2)
dt = t[1] - t[0]
T_num = np.zeros(len(t))
k = 0.275
for i in range(1, len(t)):
    if  T_num[i]<0:
        T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt
    elif T_num[i]>0:
        T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt


T_num[0] = 85

if t[i]<0:
    


print(t)
plt.plot(t, T_num, 'o-', label='nonlinear numerical');
plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3

t = np.linspace(-5, 2, 50) #step forward 2 hours
dt = t[1] - t[0]
T_num = np.zeros(len(t))
T_num[0] = 85
k = 0.275
for i in range(1, len(t)):
    T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt

    



plt.plot(t, T_num, 'o-', label='nonlinear numerical');
plt.legend(loc='best');
plt.xlabel('Time (Hours)');
plt.ylabel('Temperature (F)');
```

```{code-cell} ipython3
t = np.linspace(-5, 2)
print(t)
```

```{code-cell} ipython3
total_time = np.linspace(-5,2) # np.linspace(start, stop, steps)
total_temp = np.array([ambient_temp(t) for t in total_time])
print(total_time)
print(total_temp)
```

```{code-cell} ipython3

```
