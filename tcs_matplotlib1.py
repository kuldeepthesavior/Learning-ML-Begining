import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.testing.decorators import image_comparison


# @image_comparison(baseline_images=['Sine_Wave_Plot'], extensions=['png'])
def test_sine_wave_plot():
    # Write your functionality below
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    t = np.linspace(0.0, 2.0, num=200)
    v = np.sin(2.5 * np.pi * t)
    ax.plot(t, v, color='red', label='sin(t)', linestyle='-')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Sine Wave')
    ax.set_xlim(0, 2)
    ax.set_ylim(-1, 1)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], minor=False)
    ax.set_yticks([-1, 0, 1], minor=False)
    ax.legend()
    ax.grid(linestyle='--')
    plt.show()


# @image_comparison(baseline_images=['Multi_Curve_Plot'], extensions=['png'])
def test_multi_curve_plot():
    # Write your functionality below
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    x = np.linspace(0.0, 5.0, num=20)
    y1 = x
    y2 = x ** 2
    y3 = x ** 3
    ax.plot(x, y1, color='red', marker='o', label='y = x')
    ax.plot(x, y2, color='green', marker='s', label='y = x**2')
    ax.plot(x, y3, color='blue', marker='^', label='y = x**3')
    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_title('Linear, Quadratic, & Cubic Equations')
    ax.legend()
    plt.show()


# @image_comparison(baseline_images=['Scatter_Plot'], extensions=['png'])
def test_scatter_plot():
    # Write your functionality below
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    s = [50, 60, 55, 50, 70, 65, 75, 65, 80, 90, 93, 95]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ax.set_xlim(0, 13)
    ax.set_ylim(20, 100)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
    ax.set_xlabel('Months')
    ax.set_ylabel('No. of Cars Sold')
    ax.set_title('Cars Sold by Company \'X\' in 2017')
    ax.scatter(months, s, color='r', edgecolor='black')
    plt.show()

test_sine_wave_plot()
test_multi_curve_plot()
test_scatter_plot()