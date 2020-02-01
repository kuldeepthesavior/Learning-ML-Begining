# import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
# from matplotlib.testing.decorators import image_comparison


# @image_comparison(baseline_images=['Multiple_Plots_Figure1'], extensions=['png'])
def test_generate_figure1():
    # Write your functionality below
    t = np.arange(0.0, 5.0, 0.01)
    s1 = np.sin(2 * np.pi * t)
    s2 = np.sin(4 * np.pi * t)
    fig = plt.figure(figsize=(8, 6))
    axes1 = plt.subplot(2, 1, 1, title='Sin(2pix)')
    axes1.plot(t, s1)
    axes2 = plt.subplot(2, 1, 2, title='Sin(4pix)', sharex=axes1, sharey=axes1)
    axes2.plot(t, s2)
    plt.show()


# @image_comparison(baseline_images=['Multiple_Plots_Figure2'], extensions=['png'])
def test_generate_figure2():
    # Write your functionality below
    np.random.seed(1000)
    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.sqrt(x ** 2 + y ** 2)
    fig = plt.figure(figsize=(8, 6))
    axes1 = plt.subplot(2, 2, 1, title='Scatter plot with Upper Traingle Markers')
    axes1.scatter(x, y, c=z, s=80, marker='^')
    axes1.set_xticks([0.0, 0.4, 0.8, 1.2])
    axes1.set_yticks([-0.2, 0.2, 0.6, 1.0])
    axes2 = plt.subplot(2, 2, 2, title='Scatter plot with Plus Markers')
    axes2.scatter(x, y, s=80, c=z, marker='+')
    axes2.set_xticks([0.0, 0.4, 0.8, 1.2])
    axes2.set_yticks([-0.2, 0.2, 0.6, 1.0])
    axes3 = plt.subplot(2, 2, 3, title='Scatter plot with Circle Markers')
    axes3.scatter(x, y, s=80, c=z, marker='o')
    axes3.set_xticks([0.0, 0.4, 0.8, 1.2])
    axes3.set_yticks([-0.2, 0.2, 0.6, 1.0])
    axes4 = plt.subplot(2, 2, 4, title='Scatter plot with Diamond Markers')
    axes4.scatter(x, y, s=80, c=z, marker='d')
    axes4.set_xticks([0.0, 0.4, 0.8, 1.2])
    axes4.set_yticks([-0.2, 0.2, 0.6, 1.0])
    plt.tight_layout()
    plt.show()


# @image_comparison(baseline_images=['Multiple_Plots_Figure3'], extensions=['png'])
def test_generate_figure3():
    # Write your functionality below
    x = np.arange(1, 101)
    y1 = x
    y2 = x ** 2
    y3 = x ** 3
    fig = plt.figure(figsize=(8, 6))
    g = gridspec.GridSpec(2, 2)
    axes1 = plt.subplot(g[-2,0], title='y = x')
    axes1.plot(x, y1)
    axes2 = plt.subplot(g[-1,0], title='y = x**2')
    axes2.plot(x, y2)
    axes3 = plt.subplot(g[:, 1], title='y = x**3')
    axes3.plot(x, y3)
    plt.tight_layout()
    plt.show()


test_generate_figure1()
test_generate_figure2()
test_generate_figure3()