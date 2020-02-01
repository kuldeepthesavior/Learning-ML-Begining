# import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.testing.decorators import image_comparison


# @image_comparison(baseline_images=['Plot_with_Style1'], extensions=['png'])
def test_generate_plot_with_style1():
    # Write your functionality below
    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sepal_len = [5.01, 5.94, 6.59]
        sepal_wd = [3.42, 2.77, 2.97]
        petal_len = [1.46, 4.26, 5.55]
        petal_wd = [0.24, 1.33, 2.03]
        species = ['setosa', 'versicolor', 'viriginica']
        species_index1 = [0.7, 1.7, 2.7]
        species_index2 = [0.9, 1.9, 2.9]
        species_index3 = [1.1, 2.1, 3.1]
        species_index4 = [1.3, 2.3, 3.3]
        ax.bar(species_index1, sepal_len, width=0.2, label='Sepal Length')
        ax.bar(species_index2, sepal_wd, width=0.2, label='Sepal Width')
        ax.bar(species_index3, petal_len, width=0.2, label='Petal Length')
        ax.bar(species_index4, petal_wd, width=0.2, label='Petal Width')
        ax.set_xlabel('Species')
        ax.set_ylabel('Iris Measurements (cm)')
        ax.set_title('Mean Measurements of Iris Species')
        ax.set_xticks([1.1, 2.1, 3.1], minor=False)
        ax.set_xticklabels(['setosa', 'versicolor', 'viriginica'])
        ax.set_xlim(0.5, 3.7)
        ax.set_ylim(0, 10)
        ax.legend()
        plt.show()


# @image_comparison(baseline_images=['Plot_with_Style2'], extensions=['png'])
def test_generate_plot_with_style2():
    # Write your functionality below
    with plt.style.context(['seaborn-colorblind']):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sepal_len = [5.01, 5.94, 6.59]
        sepal_wd = [3.42, 2.77, 2.97]
        petal_len = [1.46, 4.26, 5.55]
        petal_wd = [0.24, 1.33, 2.03]
        species = ['setosa', 'versicolor', 'viriginica']
        species_index1 = [0.7, 1.7, 2.7]
        species_index2 = [0.9, 1.9, 2.9]
        species_index3 = [1.1, 2.1, 3.1]
        species_index4 = [1.3, 2.3, 3.3]
        ax.bar(species_index1, sepal_len, width=0.2, label='Sepal Length')
        ax.bar(species_index2, sepal_wd, width=0.2, label='Sepal Width')
        ax.bar(species_index3, petal_len, width=0.2, label='Petal Length')
        ax.bar(species_index4, petal_wd, width=0.2, label='Petal Width')
        ax.set_xlabel('Species')
        ax.set_ylabel('Iris Measurements (cm)')
        ax.set_title('Mean Measurements of Iris Species')
        ax.set_xticks([1.1, 2.1, 3.1], minor=False)
        ax.set_xticklabels(['setosa', 'versicolor', 'viriginica'])
        ax.set_xlim(0.5, 3.7)
        ax.set_ylim(0, 10)
        ax.legend()
        plt.show()


# @image_comparison(baseline_images=['Plot_with_Style3'], extensions=['png'])
def test_generate_plot_with_style3():
    # Write your functionality below
    with plt.style.context(['grayscale']):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sepal_len = [5.01, 5.94, 6.59]
        sepal_wd = [3.42, 2.77, 2.97]
        petal_len = [1.46, 4.26, 5.55]
        petal_wd = [0.24, 1.33, 2.03]
        species = ['setosa', 'versicolor', 'viriginica']
        species_index1 = [0.7, 1.7, 2.7]
        species_index2 = [0.9, 1.9, 2.9]
        species_index3 = [1.1, 2.1, 3.1]
        species_index4 = [1.3, 2.3, 3.3]
        ax.bar(species_index1, sepal_len, width=0.2, label='Sepal Length')
        ax.bar(species_index2, sepal_wd, width=0.2, label='Sepal Width')
        ax.bar(species_index3, petal_len, width=0.2, label='Petal Length')
        ax.bar(species_index4, petal_wd, width=0.2, label='Petal Width')
        ax.set_xlabel('Species')
        ax.set_ylabel('Iris Measurements (cm)')
        ax.set_title('Mean Measurements of Iris Species')
        ax.set_xticks([1.1, 2.1, 3.1])
        ax.set_xticklabels(['setosa', 'versicolor', 'viriginica'])
        ax.set_xlim(0.5, 3.7)
        ax.set_ylim(0, 10)
        ax.legend()
        plt.show()


test_generate_plot_with_style1()
test_generate_plot_with_style2()
test_generate_plot_with_style3()