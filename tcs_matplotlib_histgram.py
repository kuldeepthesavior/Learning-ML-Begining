# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

# @image_comparison(baseline_images=['Histogram'],extensions=['png'])
def test_hist_of_a_sample_normal_distribution():

    # Write your functionality below
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    np.random.seed(100)
    x1=25+3.0*np.random.randn(1000)
    ax.hist(x1,bins=30)
    ax.set_xlabel('X1')
    ax.set_ylabel('Bin Count')
    ax.set_title('Histogram of a Single Dataset')
    plt.show()


# @image_comparison(baseline_images=['Boxplot'],extensions=['png'])
def test_boxplot_of_four_normal_distribution():

    # Write your functionality below
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    np.random.seed(100)
    x1=25+3.0*np.random.randn(1000)
    x2=35+5.0*np.random.randn(1000)
    x3=55+10.0*np.random.randn(1000)
    x4=45+3.0*np.random.randn(1000)
    labels=['X1','X2','X3','X4']
    ax.boxplot([x1,x2,x3,x4],labels=labels,notch=True,sym='+',patch_artist=True)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Value')
    ax.set_title('Box plot of Multiple Datasets')
    plt.show()

test_boxplot_of_four_normal_distribution()
test_hist_of_a_sample_normal_distribution()
